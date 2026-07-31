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

"""Source-tree walk and src->dst mapping shared by fastcp and datastage.

Both tools take the same cp/rsync-shaped arguments and need the same answer:
given SOURCE(s) and a DEST, which files get copied where.  Keeping that in one
place means the single-node and the collective stager can never disagree about
what "copy this directory" means.
"""

import os

__all__ = ["UnreadableEntries", "list_relative_files", "plan_copy_operations"]

# How many bad paths to name before summarising the rest.
_MAX_REPORTED = 20


class UnreadableEntries(Exception):
    """Some entries under a source tree could not be stat'd.

    Raised instead of letting the first bad entry throw, so a caller learns the
    full extent in one pass rather than fixing them one job at a time.  The
    usual cause is a dangling symlink: os.walk lists it among the filenames
    because it is not a directory, and stat then follows it to nothing.
    """

    def __init__(self, entries):
        self.entries = list(entries)
        super().__init__(self._describe())

    def _describe(self):
        count = len(self.entries)
        noun = "entry" if count == 1 else "entries"
        lines = [f"{count} unreadable {noun}:"]
        for path, reason in self.entries[:_MAX_REPORTED]:
            lines.append(f"    {path}: {reason}")
        if count > _MAX_REPORTED:
            lines.append(f"    ... and {count - _MAX_REPORTED} more")
        return "\n".join(lines)


def _stat_or_problem(path, problems):
    """stat(), following symlinks. On failure record it and return None."""
    try:
        return os.stat(path)
    except OSError as exc:
        problems.append((path, exc.strerror or str(exc)))
        return None


def list_relative_files(root):
    """Return all file paths under 'root' as relative paths, sorted alphabetically.

    os.walk(followlinks=True) allows following symlinked directories, and
    symlinks to files appear in filenames, so both are dereferenced and copied
    as content rather than recreated as links.

    Raises UnreadableEntries if anything under root cannot be stat'd, reporting
    every such entry rather than dying on the first one.

    FIXME: os.walk(followlinks) doesn't protect against cycles
    to fix this we'd need to write our own version that did a depth-first
    spanning tree.
    """
    file_list = []
    problems = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            if _stat_or_problem(full_path, problems) is not None:
                file_list.append(os.path.relpath(full_path, root))
    if problems:
        raise UnreadableEntries(problems)
    return sorted(file_list)


def plan_copy_operations(sources, destination):
    """Return a list of (src_abs, dst_abs, size_bytes) tuples to copy.

    If `destination` is an existing directory each source is placed inside it
    under its own basename (recursing into directories); otherwise this is a
    single file-to-file copy.  The result is sorted by destination path so that
    every rank of a collective copy walks the files in the same order.

    Raises UnreadableEntries if any source cannot be stat'd, reporting all of
    them together.
    """
    file_jobs = []
    problems = []
    dst_root = os.path.abspath(destination)

    def add(src_abs, dst_abs):
        st = _stat_or_problem(src_abs, problems)
        if st is not None:
            file_jobs.append((src_abs, dst_abs, st.st_size))

    if not os.path.isdir(dst_root):  # case 1: single file copy
        add(os.path.abspath(sources[0]), dst_root)
    else:
        for src in sources:
            src_abs = os.path.abspath(src)
            base = os.path.basename(src.rstrip("/"))
            if os.path.isdir(src):
                try:
                    relpaths = list_relative_files(src)
                except UnreadableEntries as exc:
                    problems.extend(exc.entries)
                    continue
                for relpath in relpaths:
                    add(os.path.join(src_abs, relpath),
                        os.path.join(dst_root, base, relpath))
            else:
                add(src_abs, os.path.join(dst_root, base))

    if problems:
        raise UnreadableEntries(problems)
    return sorted(file_jobs, key=lambda job: job[1])
