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

"""copyplan's walk, mapping, and its refusal to plan an unreadable tree.

fastcp, fastmd5 and datastage all enumerate through here, so a disagreement
about which files a copy covers would mean verifying a staged tree against a
different set of files than was staged.
"""

import os
import shutil
import sys
import tempfile

import stubs

stubs.install()
copyplan = stubs.load_fileio()["copyplan"]

results = []


def check(name, condition, detail=""):
    results.append((name, condition, detail))


def build_tree(root):
    """A tree with each kind of entry the walk has to have an answer for."""
    os.makedirs(os.path.join(root, "sub"))
    os.makedirs(os.path.join(root, "realdir"))
    for path in ("a.bin", "sub/b.bin", "realdir/c.bin"):
        with open(os.path.join(root, path), "wb") as handle:
            handle.write(b"x" * 10)
    os.symlink("a.bin", os.path.join(root, "link_to_file"))
    os.symlink("realdir", os.path.join(root, "link_to_dir"))


def check_unlistable_directory(root):
    """A subtree we cannot list must be reported, not quietly left out.

    This is the one unreadable-entry case with no entry to report: the files
    under an unlistable directory never reach the walk, so there is nothing to
    stat and nothing to collect.  os.walk's default is to swallow the error and
    carry on, which makes such a subtree indistinguishable from an empty one --
    a partial copy that exits 0, and a partial checksum that agrees with it.
    """
    source = os.path.join(root, "blocked")
    os.makedirs(os.path.join(source, "readable"))
    os.makedirs(os.path.join(source, "secret"))
    for path in ("readable/good.bin", "secret/hidden.bin"):
        with open(os.path.join(source, path), "wb") as handle:
            handle.write(b"x" * 10)
    os.chmod(os.path.join(source, "secret"), 0o000)

    try:
        # Root ignores the mode bits, so the premise would not hold and the
        # check below would pass without testing anything.
        try:
            os.listdir(os.path.join(source, "secret"))
        except PermissionError:
            pass
        else:
            check("unlistable directory is reported", True,
                  "SKIPPED: this user can list a 0o000 directory")
            return

        try:
            found = copyplan.list_relative_files(source)
            check("unlistable directory is reported", False,
                  f"no exception; returned {found}")
        except copyplan.UnreadableEntries as exc:
            check("unlistable directory is reported", True)
            check("the message names the directory it could not list",
                  "secret" in str(exc), f"got {exc}")

        # Same hole one level up: the root of the walk itself.
        os.chmod(source, 0o000)
        try:
            found = copyplan.list_relative_files(source)
            check("unlistable root is reported", False,
                  f"no exception; returned {found}")
        except copyplan.UnreadableEntries:
            check("unlistable root is reported", True)
    finally:
        # Otherwise rmtree cannot clean up after us.
        os.chmod(source, 0o755)
        os.chmod(os.path.join(source, "secret"), 0o755)


def main():
    root = tempfile.mkdtemp(prefix="copyplan-")
    try:
        source = os.path.join(root, "src")
        build_tree(source)

        found = copyplan.list_relative_files(source)
        check("symlink to a file is followed and listed", "link_to_file" in found)
        check("symlink to a directory is descended into",
              "link_to_dir/c.bin" in found, f"got {found}")
        check("the symlinked directory is not itself listed as a file",
              "link_to_dir" not in found)
        check("result is sorted", found == sorted(found))

        destination = os.path.join(root, "dst")
        os.makedirs(destination)
        jobs = copyplan.plan_copy_operations([source], destination)
        check("every planned file lands under dest/<basename>/",
              all(dst.startswith(os.path.join(destination, "src") + os.sep)
                  for _, dst, _ in jobs))
        check("plan is sorted by destination",
              [j[1] for j in jobs] == sorted(j[1] for j in jobs))
        check("sizes come back with the plan", all(size == 10 for _, _, size in jobs))

        single = copyplan.plan_copy_operations(
            [os.path.join(source, "a.bin")], os.path.join(destination, "renamed.bin"))
        check("file-to-file copy keeps the given destination name",
              len(single) == 1 and single[0][1].endswith("renamed.bin"))

        # Dangling symlinks: os.walk lists them among the filenames because they
        # are not directories, and stat then follows them to nothing.
        os.symlink("/nowhere", os.path.join(source, "dead1"))
        os.symlink("/also/nowhere", os.path.join(source, "sub", "dead2"))
        try:
            copyplan.list_relative_files(source)
            check("dangling symlinks are refused", False, "no exception raised")
        except copyplan.UnreadableEntries as exc:
            check("dangling symlinks are refused", True)
            check("every bad entry is reported, not just the first",
                  len(exc.entries) == 2, f"reported {len(exc.entries)}")
            check("the message names the bad paths",
                  "dead1" in str(exc) and "dead2" in str(exc))
        try:
            copyplan.plan_copy_operations([source], destination)
            check("planning refuses an unreadable tree", False, "no exception raised")
        except copyplan.UnreadableEntries:
            check("planning refuses an unreadable tree", True)

        check_unlistable_directory(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    failures = [r for r in results if not r[1]]
    for name, _, detail in failures:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
    print(f"test_copyplan: {len(results)} checks, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
