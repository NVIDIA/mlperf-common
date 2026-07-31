#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###############################################################################
# End-to-end check of datastage on real hardware.
#
# The suite in this directory is stdlib-only and single-process: it fakes torch
# before importing datastage, so it never loads CUDA or NCCL and never talks to
# another rank.  Running it on a GPU node changes nothing -- launching it under
# srun just gets you N independent copies of the same CPU test.  Nothing in it
# exercises a real collective, real CUDA events, pinned memory against a real
# block size, or O_DIRECT.
#
# This script is the other half: one real staging job, verified by comparing
# checksums of what came out against what went in, on every node.
#
# Usage, from inside an allocation:
#
#     salloc -N2 --ntasks-per-node=4 ...
#     tests/cluster-selftest.sh /lustre/scratch/me/selftest /raid/scratch/me/selftest
#
# arg 1 is a directory on the shared filesystem (the source, and where results
#       are collected)
# arg 2 is a directory on node-local storage (the destination)
#
# Everything under both is deleted and rebuilt unless --keep-dataset is passed.
###############################################################################

set -euo pipefail

usage() {
    sed -n '18,40p' "$0" | sed 's/^# \?//' | grep -v '^#*$'
    exit "${1:-1}"
}

keep_dataset=0
args=()
for arg in "$@"; do
    case "${arg}" in
        --keep-dataset) keep_dataset=1 ;;
        -h|--help)      usage 0 ;;
        -*)             echo "unknown option ${arg}" >&2; usage ;;
        *)              args+=("${arg}") ;;
    esac
done
[[ "${#args[@]}" -eq 2 ]] || usage

readonly SHARED="${args[0]}"
readonly LOCAL="${args[1]}"
readonly REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DATASET="${SHARED}/dataset"
readonly RESULTS="${SHARED}/results"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: no SLURM_JOB_ID; run this inside an salloc or sbatch" >&2
    exit 1
fi

# One task per GPU is how datastage is meant to run.  DGXNGPU if the site sets
# it, otherwise count what this node has.
readonly NGPUS="${DGXNGPU:-$(nvidia-smi -L | wc -l)}"
readonly NNODES="${SLURM_JOB_NUM_NODES:-1}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=============================================================="
echo "datastage cluster self-test"
echo "  repo        ${REPO}"
echo "  nodes       ${NNODES} x ${NGPUS} ranks"
echo "  source      ${DATASET}   (shared)"
echo "  destination ${LOCAL}     (node-local)"
echo "=============================================================="

###############################################################################
# 1. Build a dataset with the shapes that have historically broken things.
###############################################################################
if [[ "${keep_dataset}" -eq 1 && -d "${DATASET}" ]]; then
    echo "--- 1. reusing existing dataset"
else
    echo "--- 1. building dataset"
    rm -rf "${DATASET}"
    mkdir -p "${DATASET}"
    python3 - "${DATASET}" <<'PY'
import os, sys
root = sys.argv[1]
MiB = 1024 ** 2

# Sizes clustered around the 2 MiB alignment boundary, where the O_DIRECT write
# padding and the closing ftruncate interact, plus two large enough to need
# several rounds through the pipeline at any plausible node count.
sizes = {
    "empty.bin":        0,
    "one.bin":          1,
    "small.bin":        1000,
    "align-1.bin":      2 * MiB - 1,
    "align.bin":        2 * MiB,
    "align+1.bin":      2 * MiB + 1,
    "ragged.bin":       5 * MiB + 12345,
    "sub/medium.bin":   64 * MiB + 7,
    "sub/deep/big.bin": 200 * MiB,
}
for name, size in sizes.items():
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        remaining = size
        while remaining:
            block = min(remaining, 8 * MiB)
            handle.write(os.urandom(block))
            remaining -= block

# A pile of small files: one metadata operation each, and at high node counts
# each costs a whole window of collective traffic regardless of its size.
os.makedirs(os.path.join(root, "many"), exist_ok=True)
for i in range(64):
    with open(os.path.join(root, "many", f"f{i:03d}.bin"), "wb") as handle:
        handle.write(os.urandom(4096 + i))

# Symlinks are dereferenced and copied as content, not recreated as links.
os.symlink("align.bin", os.path.join(root, "link-to-file"))

total = sum(sizes.values()) + sum(4096 + i for i in range(64))
print(f"    {len(sizes) + 64 + 1} files, {total / 1e6:.1f} MB")
PY
fi

###############################################################################
# 2. Checksum the source once, from the shared filesystem.
###############################################################################
echo "--- 2. checksumming source"
rm -rf "${RESULTS}"
mkdir -p "${RESULTS}"
"${REPO}/client/fastmd5" "${DATASET}" \
    | sed "s|^${DATASET}/||" | sort > "${RESULTS}/source.md5"

# Checksums alone are not enough.  fastmd5 emits one line per GB-chunk, so a
# zero-length file produces no lines at all -- and a destination missing
# empty.bin entirely would compare equal.  The size+path listing closes that,
# and catches extra files and truncations too.
#
# mtime is deliberately not compared: datastage does set it from the source,
# but timestamp granularity varies by filesystem and a false failure here on a
# first run would cost more than the check is worth.
( cd "${DATASET}" && find . -type f -printf '%s\t%p\n' | sort ) \
    > "${RESULTS}/source.list"
echo "    $(wc -l < "${RESULTS}/source.md5") chunk checksums, \
$(wc -l < "${RESULTS}/source.list") files"

###############################################################################
# 3. Clear the destination on every node.
###############################################################################
echo "--- 3. clearing destinations"
srun --ntasks-per-node=1 bash -c "rm -rf '${LOCAL}' && mkdir -p '${LOCAL}'"

###############################################################################
# 4. Stage.  No wrapper: datastage derives the rendezvous itself.
###############################################################################
echo "--- 4. staging"
staged_ok=1
srun --ntasks-per-node="${NGPUS}" \
    python3 -m mlperf_common.fileio.datastage -r "${DATASET}" "${LOCAL}" \
    || staged_ok=0
if [[ "${staged_ok}" -eq 0 ]]; then
    echo "FAIL: datastage exited nonzero" >&2
    exit 1
fi

###############################################################################
# 5. Verify on every node, against the node-local copy.
###############################################################################
echo "--- 5. verifying every node"
srun --ntasks-per-node=1 bash -c "
    set -e
    host=\$(hostname -s)
    '${REPO}/client/fastmd5' '${LOCAL}/dataset' \
        | sed 's|^${LOCAL}/dataset/||' | sort > '${RESULTS}/'\${host}.md5
    ( cd '${LOCAL}/dataset' && find . -type f -printf '%s\t%p\n' | sort ) \
        > '${RESULTS}/'\${host}.list
    # F9: nothing should be left of the temp files staging writes through.
    find '${LOCAL}' -name '*.datastage.tmp.*' > '${RESULTS}/'\${host}.temps
    # --chmod defaults to 0777; anything tighter breaks a shared scratch dir.
    find '${LOCAL}/dataset' \\! -perm -0777 > '${RESULTS}/'\${host}.modes
"

###############################################################################
# 6. Compare.
###############################################################################
echo "--- 6. results"
failures=0
for md5 in "${RESULTS}"/*.md5; do
    host="$(basename "${md5}" .md5)"
    [[ "${host}" == "source" ]] && continue

    if diff -q "${RESULTS}/source.list" "${RESULTS}/${host}.list" > /dev/null; then
        : # same files, same sizes
    else
        echo "    ${host}: FILE LIST DIFFERS (< missing, > unexpected)"
        diff "${RESULTS}/source.list" "${RESULTS}/${host}.list" \
            | head -20 | sed 's/^/        /'
        failures=$((failures + 1))
    fi

    if diff -q "${RESULTS}/source.md5" "${md5}" > /dev/null; then
        echo "    ${host}: bytes match"
    else
        echo "    ${host}: CONTENT MISMATCH"
        diff "${RESULTS}/source.md5" "${md5}" | head -20 | sed 's/^/        /'
        failures=$((failures + 1))
    fi

    if [[ -s "${RESULTS}/${host}.temps" ]]; then
        echo "    ${host}: left temp files:"
        sed 's/^/        /' "${RESULTS}/${host}.temps"
        failures=$((failures + 1))
    fi
    if [[ -s "${RESULTS}/${host}.modes" ]]; then
        echo "    ${host}: entries not mode 0777:"
        head -10 "${RESULTS}/${host}.modes" | sed 's/^/        /'
        failures=$((failures + 1))
    fi
done

nodes_checked=$(ls -1 "${RESULTS}"/*.md5 | grep -cv 'source\.md5$')
if [[ "${nodes_checked}" -ne "${NNODES}" ]]; then
    echo "    only ${nodes_checked} of ${NNODES} nodes reported" >&2
    failures=$((failures + 1))
fi

echo "=============================================================="
if [[ "${failures}" -eq 0 ]]; then
    echo "PASS: ${nodes_checked} node(s) hold byte-identical copies"
else
    echo "FAIL: ${failures} problem(s); details under ${RESULTS}"
fi
echo "=============================================================="
exit "$((failures == 0 ? 0 : 1))"
