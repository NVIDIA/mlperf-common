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

"""Every CUDA event must be recorded against this rank's device, not device 0.

torch.cuda's current device is per host thread.  main() sets it once, on the
main thread, so every thread datastage starts afterwards inherits device 0 --
and torch.cuda.Event binds to the *calling thread's* current device when it is
recorded.  An event recorded on the wrong device belongs to an idle stream, so
Event.synchronize() on it returns immediately: the writer pool reads a pinned
chunk the device-to-host copy has not filled, and a device window is recycled
while copies are still reading it.  Right-sized file, wrong bytes, exit 0, on
every rank except LOCAL_RANK 0.

None of that is visible in the staged bytes here, because on a CPU every
"copy" has already landed by the time anything looks -- which is exactly why
test_pipeline.py cannot catch it.  What is visible is the device each event was
recorded against, so that is what this checks.

This is not a test of CUDA semantics.  It is a test that the threads which
touch CUDA agree about which GPU they are on.
"""

import os
import shutil
import sys
import tempfile

import stubs

MiB = 1024 ** 2

# Anything but 0.  Device 0 is what a thread that never called set_device gets,
# so pinning this rank to device 0 would pass no matter what.
#
# Stager takes its device from torch.cuda.current_device(), not from
# topo.local_rank, so the two are free to differ here: local_rank stays 0 so
# that this single process is the one that creates the destination file, while
# the device it is actually on is 3.
DEVICE = 3

torch, dist = stubs.install()
modules = stubs.load_fileio()
ds = modules["datastage"]

# As in test_pipeline: O_DIRECT would demand alignment the fake buffers cannot
# promise, and nothing here depends on it.
ds.open_maybe_direct = lambda path, flags: os.open(path, flags)


class Topology:
    """One node, one rank, sitting on GPU DEVICE."""

    node_count = 1
    local_world_size = 1
    local_rank = 0    # this process creates the destination file
    rank = 1          # not 0: keeps the stager's banner out of the way
    node_index = 0
    group = None


class Args:
    num_threads = 4
    chmod = 0o777
    buffer_size = 2 * MiB

    def __init__(self, destination):
        self.destination = destination


def main():
    root = tempfile.mkdtemp(prefix="datastage-device-")
    try:
        src = os.path.join(root, "src")
        # Several rounds, with a ragged last one.
        size = 5 * MiB + 12345
        with open(src, "wb") as handle:
            handle.write(os.urandom(size))

        # What main() does, on the thread main() runs on -- and only there.
        torch.cuda.set_device(torch.device("cuda", DEVICE))
        stager = ds.Stager(Args(root), Topology())

        del stubs.EVENTS[:]
        stager.stage_file(src, os.path.join(root, "dst"), size,
                          os.stat(src).st_mtime_ns)
        events = list(stubs.EVENTS)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if not events:
        print("  FAIL no events recorded at all; this test is exercising nothing")
        return 1

    stray = [event for event in events if event.device != DEVICE]
    print(f"test_device: {len(events)} events recorded, {len(stray)} on the wrong device")
    if stray:
        wrong = sorted({event.device for event in stray})
        print(f"  FAIL {len(stray)}/{len(events)} events recorded against device "
              f"{wrong} instead of {DEVICE}: a thread that records events "
              f"never called torch.cuda.set_device, so synchronizing on them "
              f"waits for nothing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
