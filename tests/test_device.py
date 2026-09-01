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

"""CUDA bookkeeping the pipeline gets to be wrong about on a CPU.

Two properties, both invisible in the staged bytes and both able to corrupt a
real run: which *device* each event was recorded against, and which *stream*
the drainer's copies were issued on.

Every CUDA event must be recorded against this rank's device, not device 0.

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

The drainer's copy-back must go to a stream of its own.  On the default stream
it queues behind the next round's collective instead of running beside it on
the copy engine, which makes the second device window RECV_DEV_SLOTS buys pure
waste.  That much is only a throughput bug -- but a side stream that does not
first wait on the collective's event is a correctness one, because the copy
would read a window the all-gather has not filled.  Both are checkable here:
concurrency is not observable on a CPU, but which stream work was issued on,
and in what order, is.

This is not a test of CUDA semantics.  It is a test that the threads which
touch CUDA agree about which GPU they are on and which queue they are feeding.
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

# Set by _run_pipeline when it starts the drain thread; this is how the ops
# below are attributed to the drainer rather than to the main loop.
DRAIN_THREAD = "datastage-drainer"

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
        del stubs.OPS[:]
        stager.stage_file(src, os.path.join(root, "dst"), size,
                          os.stat(src).st_mtime_ns)
        events = list(stubs.EVENTS)
        ops = list(stubs.OPS)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if not events:
        print("  FAIL no events recorded at all; this test is exercising nothing")
        return 1

    problems = []

    stray = [event for event in events if event.device != DEVICE]
    if stray:
        wrong = sorted({event.device for event in stray})
        problems.append(
            f"{len(stray)}/{len(events)} events recorded against device {wrong} "
            f"instead of {DEVICE}: a thread that records events never called "
            f"torch.cuda.set_device, so synchronizing on them waits for nothing")

    # The drain thread is named where it is started, in _run_pipeline.
    drained = [op for op in ops if op.thread == DRAIN_THREAD]
    if not drained:
        problems.append(
            f"no operations issued from a thread named {DRAIN_THREAD!r}; either "
            f"the drainer was renamed or the copy-back moved somewhere else, and "
            f"this test is no longer looking at it")
    else:
        on_default = [op for op in drained if op.kind != "wait" and op.stream.default]
        if on_default:
            problems.append(
                f"{len(on_default)}/{len(drained)} drainer operations were issued "
                f"on the default stream, where they queue behind the next round's "
                f"collective instead of overlapping it")

        # Only meaningful once the copies are on a stream of their own: work
        # queued on the default stream is already ordered behind the collective
        # by the stream itself, and a host-side wait orders it too.  A side
        # stream has neither, so it has to be told explicitly.
        side = [op for op in drained if not op.stream.default]
        if side:
            copies = [i for i, op in enumerate(side) if op.kind == "copy"]
            waits = [i for i, op in enumerate(side) if op.kind == "wait"]
            if copies and (not waits or waits[0] > copies[0]):
                problems.append(
                    "the drainer copied from a device window on a side stream "
                    "without first waiting on the event that says the all-gather "
                    "filled it; nothing orders the copy behind the collective")

    print(f"test_device: {len(events)} events, {len(drained)} drainer operations "
          f"on {len({op.stream for op in drained})} stream(s), {len(problems)} problems")
    for problem in problems:
        print(f"  FAIL {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
