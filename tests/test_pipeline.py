#!/usr/bin/env python3

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

"""Run stage_file end to end on the CPU and compare the bytes.

The reader thread, the collective, the draining thread and the writer pool hand
buffers between each other through four queues.  Getting that wrong does not
raise -- it produces a file of exactly the right length holding the wrong bytes,
which is why this compares content rather than just checking for exceptions.
Two such bugs were found this way: a drain queue emptied on shutdown before it
had been processed, and a missing final ftruncate that left O_DIRECT write
padding on the end of every unaligned file.

The multi-node case fakes the all-gather by filling each node's segment with
the bytes that node would have read, which is what the real collective delivers.
"""

import os
import random
import shutil
import sys
import tempfile

import stubs

MiB = 1024 ** 2
torch, dist = stubs.install()
modules = stubs.load_fileio()
ds = modules["datastage"]

# The tests run on a normal filesystem, where O_DIRECT would demand alignment
# the fake buffers cannot promise; the padding behaviour under test is
# direct_io's, and that is identical on a buffered descriptor.
ds.open_maybe_direct = lambda path, flags: os.open(path, flags)

# Sizes chosen around the alignment boundary, since that is where the write
# padding and the final truncate interact.
SIZES = [0, 1, 1000, 2 * MiB - 1, 2 * MiB, 2 * MiB + 1, 5 * MiB + 12345, 20 * MiB]
MULTINODE = [
    # (nodes, buffer_size, file size)
    (1, 2 * MiB, 5_000_000),
    (3, 2 * MiB, 5_000_000),
    (3, 2 * MiB, 100_000_000),
    (8, 2 * MiB, 100_000_000),
    (8, 8 * MiB, 100_000_000),
    (64, 2 * MiB, 100_000_000),   # several host chunks per window
    (64, 8 * MiB, 100_000_000),
    (64, 2 * MiB, 999),
]

STATE = {}
_FileLayout = ds.FileLayout


class RecordingLayout(_FileLayout):
    """Expose the layout the stager built, so the fake collective can use it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        STATE["layout"] = self
        STATE["round"] = 0


ds.FileLayout = RecordingLayout


def fake_all_gather(out, inp, group=None):
    """Deliver what a real all-gather would: every node's segment, filled in."""
    layout, round_index = STATE["layout"], STATE["round"]
    STATE["round"] += 1
    view = out.numpy()
    view[:] = b"\0" * len(view)
    for node in range(STATE["nodes"]):
        offset, length = layout.segment(node, round_index)
        if not length:
            continue
        chunk = os.pread(STATE["fd"], length, offset)
        base = node * layout.piece
        view[base:base + len(chunk)] = chunk


class Topology:
    def __init__(self, nodes):
        self.node_count = nodes
        self.local_world_size = 1
        self.local_rank = 0
        self.rank = 1          # not 0: keeps the stager's banner out of the way
        self.node_index = 0
        self.group = None


class Args:
    num_threads = 4
    chmod = 0o777

    def __init__(self, destination, buffer_size):
        self.destination = destination
        self.buffer_size = buffer_size


def stage_once(root, nodes, buffer_size, size, tag):
    """Stage one generated file and report whether it arrived intact."""
    src = os.path.join(root, f"src_{tag}")
    dst = os.path.join(root, f"dst_{tag}")
    payload = random.Random(size + nodes).randbytes(size)
    with open(src, "wb") as handle:
        handle.write(payload)

    STATE["nodes"] = nodes
    STATE["fd"] = os.open(src, os.O_RDONLY)
    try:
        stager = ds.Stager(Args(root, buffer_size), Topology(nodes))
        stager.stage_file(src, dst, size, os.stat(src).st_mtime_ns)
    finally:
        os.close(STATE["fd"])

    with open(dst, "rb") as handle:
        got = handle.read()
    leftovers = [n for n in os.listdir(root) if ".datastage.tmp." in n]
    problems = []
    if len(got) != size:
        problems.append(f"length {len(got)} != {size}")
    elif got != payload:
        first = next(i for i in range(size) if got[i] != payload[i])
        problems.append(f"bytes differ from offset {first}")
    if os.stat(dst).st_mode & 0o777 != 0o777:
        problems.append(f"mode {oct(os.stat(dst).st_mode & 0o777)}")
    if leftovers:
        problems.append(f"left {len(leftovers)} temp files")
    return problems


def main():
    root = tempfile.mkdtemp(prefix="datastage-")
    failures = 0
    checked = 0
    try:
        dist.all_gather_into_tensor = lambda out, inp, group=None: out.copy_(inp)
        STATE["nodes"] = 1
        for size in SIZES:
            checked += 1
            STATE["fd"] = -1
            problems = stage_once(root, 1, 2 * MiB, size, f"s{size}")
            if problems:
                failures += 1
                print(f"  FAIL single-node size={size}: {'; '.join(problems)}")

        dist.all_gather_into_tensor = fake_all_gather
        for nodes, buffer_size, size in MULTINODE:
            checked += 1
            problems = stage_once(root, nodes, buffer_size,
                                  size, f"m{nodes}_{buffer_size}_{size}")
            if problems:
                failures += 1
                print(f"  FAIL nodes={nodes} -b={buffer_size >> 20}M size={size}: "
                      f"{'; '.join(problems)}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"test_pipeline: {checked} staged files, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
