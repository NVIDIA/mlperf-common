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

"""Rank 0 must never leave the planning step without telling the other ranks.

build_plan has one rank walk the tree and broadcast the answer.  Every other
rank is already blocked in that broadcast, so anything rank 0 raises instead of
broadcasting does not surface as an error: rank 0 exits, its peers wait for a
message that will never come, and the job dies ten minutes later on an NCCL
watchdog timeout naming neither the file nor the reason.

So the requirement is not "handle the expected failure" but "reach the
broadcast no matter what".  These tests drive rank 0's side and check that
failures arrive as a broadcastable payload rather than an exception.
"""

import os
import shutil
import sys
import tempfile

import stubs

torch, dist = stubs.install()
modules = stubs.load_fileio()
ds = modules["datastage"]
copyplan = modules["copyplan"]

results = []


def check(name, condition, detail=""):
    results.append((name, condition, detail))


class Args:
    def __init__(self, sources, destination):
        self.sources = sources
        self.destination = destination


def build_plan_outcome(args):
    """Run rank 0's build_plan; report how it came back.

    Returns ("broadcast", message) if the failure was turned into something
    every rank raises together, ("escaped", exc) if it got out of build_plan
    before the broadcast, or ("ok", jobs).
    """
    try:
        return "ok", ds.build_plan(args)
    except RuntimeError as exc:
        # What build_plan raises after a broadcast payload carrying an error;
        # every rank reaches this together.
        return "broadcast", str(exc)
    except BaseException as exc:  # noqa: BLE001 - the failure under test
        return "escaped", exc


def main():
    root = tempfile.mkdtemp(prefix="buildplan-")
    original_plan = ds.plan_copy_operations
    try:
        source = os.path.join(root, "src")
        os.makedirs(source)
        for name in ("a.bin", "b.bin"):
            with open(os.path.join(source, name), "wb") as handle:
                handle.write(b"x" * 10)
        destination = os.path.join(root, "dst")
        os.makedirs(destination)

        kind, value = build_plan_outcome(Args([source], destination))
        check("a good tree plans", kind == "ok", f"got {kind}: {value}")
        if kind == "ok":
            check("the plan carries size and mtime",
                  len(value) == 2 and all(len(job) == 4 for job in value),
                  f"got {value}")

        # A file that disappears between plan_copy_operations' stat and the
        # mtime stat below it.  Rare, but it is exactly what a shared
        # filesystem does under a concurrent job, and os.stat raises
        # FileNotFoundError, which is not UnreadableEntries.
        vanished = os.path.join(source, "gone.bin")
        ds.plan_copy_operations = lambda sources, dest: [
            (vanished, os.path.join(destination, "gone.bin"), 10)]
        kind, value = build_plan_outcome(Args([source], destination))
        check("a file vanishing mid-plan is broadcast, not raised",
              kind == "broadcast", f"got {kind}: {value!r}")
        if kind == "broadcast":
            check("the broadcast error names the file",
                  "gone.bin" in value, f"got {value!r}")
        ds.plan_copy_operations = original_plan

        # The failure build_plan already expected still works.
        os.symlink("/nowhere", os.path.join(source, "dead"))
        kind, value = build_plan_outcome(Args([source], destination))
        check("an unreadable entry is broadcast", kind == "broadcast",
              f"got {kind}: {value!r}")
        if kind == "broadcast":
            check("the broadcast error names the unreadable entry",
                  "dead" in value, f"got {value!r}")
        os.remove(os.path.join(source, "dead"))

        # Bad arguments reach here too, now that plan_copy_operations applies
        # the cp rules itself -- CopyArgumentError is not UnreadableEntries
        # either.
        kind, value = build_plan_outcome(
            Args([source, source], os.path.join(root, "absent")))
        check("a rejected argument combination is broadcast",
              kind == "broadcast", f"got {kind}: {value!r}")
    finally:
        ds.plan_copy_operations = original_plan
        shutil.rmtree(root, ignore_errors=True)

    failures = [r for r in results if not r[1]]
    for name, _, detail in failures:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
    print(f"test_buildplan: {len(results)} checks, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
