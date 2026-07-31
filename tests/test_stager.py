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

"""Two Stager behaviours that are not about bytes.

`_chmod_parents` walks up from a staged file towards the destination root, and
its terminating condition has to cope with a destination root of "/", where
os.path.dirname stops making progress.  A runaway there does not crash: it
chmods the container root forever while every peer rank waits in the barrier
on the next line, so the job burns its whole wall-clock allocation.

The device-memory budget check refuses a --buffer-size that will not fit and
suggests a smaller one.  If the suggestion is itself over budget the operator
follows the advice, gets the identical error, and burns another multi-node
allocation -- so what matters is not that a number is printed but that the
number works.
"""

import os
import re
import sys
import types

import stubs

GiB = 1024 ** 3
MiB = 1024 ** 2

torch, dist = stubs.install()
ds = stubs.load_fileio()["datastage"]

results = []


def check(name, condition, detail=""):
    results.append((name, condition, detail))


class Topology:
    def __init__(self, nodes):
        self.node_count = nodes
        self.local_world_size = 1
        self.local_rank = 0
        self.rank = 1          # not 0: keeps the stager's banner quiet
        self.node_index = 0
        self.group = None


class Args:
    num_threads = 4
    chmod = 0o777

    def __init__(self, destination, buffer_size):
        self.destination = destination
        self.buffer_size = buffer_size


def chmod_parents_calls(dest_root, dst, limit=64):
    """Run _chmod_parents against a stub, reporting the paths it chmod'd.

    Raises RuntimeError if it exceeds `limit` calls, which is how a
    non-terminating walk shows up as a failure instead of hanging the suite.
    """
    calls = []
    stager = types.SimpleNamespace(dest_root=dest_root,
                                   args=types.SimpleNamespace(chmod=0o777))

    def counting_chmod(path, mode):
        calls.append(path)
        if len(calls) > limit:
            raise RuntimeError(f"_chmod_parents did not terminate: {limit}+ calls")

    real_chmod = os.chmod
    os.chmod = counting_chmod
    try:
        ds.Stager._chmod_parents(stager, dst)
    finally:
        os.chmod = real_chmod
    return calls


def check_chmod_parents():
    try:
        calls = chmod_parents_calls("/", "/a.bin")
        check("a destination root of '/' terminates", True)
        check("'/' is chmod'd exactly once", calls == ["/"], f"got {calls}")
    except RuntimeError as exc:
        check("a destination root of '/' terminates", False, str(exc))
        check("'/' is chmod'd exactly once", False, "did not terminate")

    calls = chmod_parents_calls("/raid/scratch/ds", "/raid/scratch/ds/a/b/c.bin")
    check("it walks up to the destination root and stops",
          calls == ["/raid/scratch/ds/a/b", "/raid/scratch/ds/a", "/raid/scratch/ds"],
          f"got {calls}")

    calls = chmod_parents_calls("/raid/scratch/ds", "/raid/scratch/ds/c.bin")
    check("a file directly in the root chmods only the root",
          calls == ["/raid/scratch/ds"], f"got {calls}")

    # dest_root is a prefix of the path but not a parent directory of it; the
    # startswith test must not be fooled by the shared prefix.
    calls = chmod_parents_calls("/raid/scratch/ds", "/raid/scratch/ds-other/c.bin")
    check("a sibling directory sharing the prefix is not touched",
          calls == [], f"got {calls}")


def budget_error(total_memory, nodes, buffer_size):
    """Build a Stager and return the RuntimeError text, or None if it fit.

    The device size is patched into the stub in place rather than by calling
    stubs.install() again: datastage bound `torch` at import, so a fresh module
    from install() would not be the one it consults.

    Allocation is made free for the same duration.  The budget check runs
    before any buffers are reserved, and has to keep doing so -- but an
    *accepted* 64-node case would otherwise really allocate tens of GiB, since
    the fake tensors are backed by actual host memory.
    """
    saved = (ds.torch.cuda.get_device_properties, ds.torch.empty, ds.pinned_aligned)
    ds.torch.cuda.get_device_properties = lambda device: types.SimpleNamespace(
        total_memory=int(total_memory))
    ds.torch.empty = lambda n, **kwargs: stubs.FakeTensor(0)
    ds.pinned_aligned = lambda nbytes, alignment: (
        None, stubs.FakeTensor(0), memoryview(bytearray(1)))
    try:
        ds.Stager(Args("/tmp", buffer_size), Topology(nodes))
        return None
    except RuntimeError as exc:
        return str(exc)
    finally:
        (ds.torch.cuda.get_device_properties, ds.torch.empty,
         ds.pinned_aligned) = saved


def check_budget_advice():
    """The suggested --buffer-size has to be one that actually works."""
    # Device sizes against node counts that overflow them.  The requirement is
    # piece * (RECV_DEV_SLOTS * nodes + 1) against 60% of the device, so e.g.
    # 512M at 64 nodes wants 64.5 GiB of an 80 GB card's 47.8 GiB budget.
    cases = [
        (79.65 * GiB, 64, 512 * MiB),
        (79.65 * GiB, 32, 1024 * MiB),
        (79.65 * GiB, 16, 2048 * MiB),
        (94.0 * GiB, 64, 512 * MiB),
        (141.0 * GiB, 64, 1024 * MiB),
        (141.0 * GiB, 128, 1024 * MiB),
        (79.65 * GiB, 2048, 32 * MiB),
        (8.0 * GiB, 2048, 8 * MiB),      # cannot fit even at the 2 MiB minimum
    ]
    for total_memory, nodes, buffer_size in cases:
        label = f"{total_memory / GiB:.0f}GiB/{nodes}n/{buffer_size >> 20}M"
        message = budget_error(total_memory, nodes, buffer_size)
        if message is None:
            check(f"{label}: expected a refusal", False, "it was accepted")
            continue

        match = re.search(r"at most (\d+)M", message)
        if not match:
            # No suggestion means it claimed the node count cannot fit at all.
            check(f"{label}: says the node count does not fit",
                  "does not fit on this GPU" in message, message)
            minimum = budget_error(total_memory, nodes, 2 * MiB)
            check(f"{label}: and the minimum really is refused",
                  minimum is not None, "the 2M minimum was accepted")
            continue

        suggested = int(match.group(1)) * MiB
        again = budget_error(total_memory, nodes, suggested)
        check(f"{label}: the suggested {suggested >> 20}M is accepted",
              again is None, f"still refused: {again}")

        # And it should be close to the largest that fits, not needlessly small.
        bigger = budget_error(total_memory, nodes, suggested + 2 * MiB)
        check(f"{label}: {(suggested >> 20) + 2}M would not have fit",
              bigger is not None, "the suggestion was more than 2 MiB too low")


def main():
    check_chmod_parents()
    check_budget_advice()

    failures = [r for r in results if not r[1]]
    for name, _, detail in failures:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
    print(f"test_stager: {len(results)} checks, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
