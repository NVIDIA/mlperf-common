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

fastcp and fastmd5 both enumerate through here, so a disagreement about which
files a copy covers would mean checksumming a different set of files than was
copied.
"""

import os
import shutil
import sys
import tempfile

# copyplan is stdlib-only, so it imports straight out of the checkout with no
# stubbing: nothing here needs torch.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlperf_common.fileio import copyplan  # noqa: E402

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


def check_cp_semantics(root):
    """The SOURCE/DEST rules, against what GNU cp actually does.

    fastcp was written to match cp, and fastmd5 inherits the same enumeration
    through this module, so these are the cases where "what does copy mean here"
    has to have exactly one answer.  The rows marked ERROR are the ones cp refuses.
    """
    base = os.path.join(root, "sem")
    src = os.path.join(base, "src")
    os.makedirs(os.path.join(src, "sub"))
    for path in ("a.bin", "sub/b.bin"):
        with open(os.path.join(src, path), "wb") as handle:
            handle.write(b"x" * 10)
    for name in ("f1", "f2"):
        with open(os.path.join(base, name), "wb") as handle:
            handle.write(b"y" * 10)
    existing = os.path.join(base, "existingdir")
    os.makedirs(existing)

    f1, f2 = os.path.join(base, "f1"), os.path.join(base, "f2")
    absent = os.path.join(base, "newdir")

    def plan(sources, destination, **kwargs):
        plan = copyplan.plan_copy_operations(sources, destination, **kwargs)
        return sorted(os.path.relpath(dst, base) for _, dst, _ in plan.files)

    # cp -r src newdir, newdir absent: the contents land directly in newdir,
    # with no basename level.  This is the case that used to plan a single
    # 4 KiB "copy the directory as a file" job.
    check("cp -r src newdir puts contents directly under newdir",
          plan([src], absent) == ["newdir/a.bin", "newdir/sub/b.bin"],
          f"got {plan([src], absent)}")

    # cp -r src existingdir: basename level is kept.
    check("cp -r src existingdir keeps the basename level",
          plan([src], existing) == ["existingdir/src/a.bin",
                                    "existingdir/src/sub/b.bin"],
          f"got {plan([src], existing)}")

    check("cp a b existingdir places both under it",
          plan([f1, f2], existing) == ["existingdir/f1", "existingdir/f2"])

    check("cp a newname renames a single file",
          plan([f1], os.path.join(base, "renamed.bin")) == ["renamed.bin"])

    for name, sources, destination, kwargs in (
            ("cp a b newdir is refused when newdir is absent", [f1, f2], absent, {}),
            ("cp -r src file is refused", [src], f1, {}),
            ("cp -t newdir src is refused when newdir is absent",
             [src], absent, {"into_directory": True}),
            ("a directory source without -r is refused",
             [src], existing, {"recursive": False}),
    ):
        try:
            copyplan.validate_copy_args(sources, destination, **kwargs)
            check(name, False, "no exception raised")
        except copyplan.CopyArgumentError:
            check(name, True)

    # The planner must not depend on its caller having validated first.
    try:
        copyplan.plan_copy_operations([f1, f2], absent)
        check("planning applies the rules itself", False, "no exception raised")
    except copyplan.CopyArgumentError:
        check("planning applies the rules itself", True)


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


def check_empty_directories(root):
    """Directories with no files under them still have to be created.

    Nothing in `files` implies them -- that is the whole point -- so a planner
    that reported only files produced a destination tree quietly missing every
    empty directory, and for a wholly empty source produced no jobs at all and
    never created the destination.  cp -r creates it, and exits 0 either way,
    so a caller checking the exit status could not tell.
    """
    base = os.path.join(root, "emptydirs")
    source = os.path.join(base, "src")
    os.makedirs(os.path.join(source, "sub", "deep", "leaf"))
    os.makedirs(os.path.join(source, "lonely"))
    with open(os.path.join(source, "sub", "f.txt"), "w") as fh:
        fh.write("x")

    plan = copyplan.plan_copy_operations([source], os.path.join(base, "dst"))
    rel = {os.path.relpath(d, os.path.join(base, "dst")) for d in plan.directories}
    check("empty directories appear in the plan",
          {"lonely", os.path.join("sub", "deep"), os.path.join("sub", "deep", "leaf")} <= rel,
          f"got {sorted(rel)}")
    check("a directory holding a file is planned too", "sub" in rel)
    check("directories are shallowest first",
          list(plan.directories) == sorted(plan.directories))

    # A source with no files at all still has to produce the destination.
    empty_src = os.path.join(base, "wholly-empty")
    os.makedirs(empty_src)
    empty_dst = os.path.join(base, "newdir")
    plan = copyplan.plan_copy_operations([empty_src], empty_dst)
    check("an empty source plans no file jobs", plan.files == [])
    check("an empty source still plans its destination directory",
          empty_dst in plan.directories, f"got {plan.directories}")

    # list_relative_entries shares one walk; the files half must not drift
    # from what list_relative_files reports.
    entries = copyplan.list_relative_entries(source)
    check("list_relative_files agrees with list_relative_entries",
          entries.files == copyplan.list_relative_files(source))


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
        plan = copyplan.plan_copy_operations([source], destination)
        check("every planned file lands under dest/<basename>/",
              all(dst.startswith(os.path.join(destination, "src") + os.sep)
                  for _, dst, _ in plan.files))
        check("plan is sorted by destination",
              [j[1] for j in plan.files] == sorted(j[1] for j in plan.files))
        check("sizes come back with the plan",
              all(size == 10 for _, _, size in plan.files))

        single = copyplan.plan_copy_operations(
            [os.path.join(source, "a.bin")], os.path.join(destination, "renamed.bin"))
        check("file-to-file copy keeps the given destination name",
              len(single.files) == 1 and single.files[0][1].endswith("renamed.bin"))

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

        check_cp_semantics(root)
        check_unlistable_directory(root)
        check_empty_directories(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    failures = [r for r in results if not r[1]]
    for name, _, detail in failures:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
    print(f"test_copyplan: {len(results)} checks, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
