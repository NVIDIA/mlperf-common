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

The argument rules live here too, in validate_copy_args, rather than in each
tool's CLI.  They are not decoration: the mapping below is only well defined
once they hold, and a caller that skipped them used to get a silently wrong
plan instead of an error.  plan_copy_operations therefore applies them itself,
and the CLIs call them early only to fail before doing any other setup.

The semantics are GNU cp's, which is what fastcp set out to match:

    cp -r src newdir      newdir absent -> create it, src's *contents* inside
    cp -r src existingdir                -> existingdir/src/...
    cp a b existingdir                   -> existingdir/a, existingdir/b
    cp a b newdir         newdir absent -> error, target is not a directory
    cp -r src file                       -> error, cannot overwrite non-directory
"""

import os

__all__ = ["CopyArgumentError", "UnreadableEntries", "list_relative_files",
           "plan_copy_operations", "validate_copy_args"]

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


class CopyArgumentError(Exception):
    """The SOURCE/DEST combination is not one cp would accept.

    Carries a message in cp's wording, without a program name; callers prefix
    their own, as cp does.
    """


def validate_copy_args(sources, destination, recursive=True, into_directory=False):
    """Check a SOURCE(s)/DEST combination, raising CopyArgumentError if bad.

    `recursive` is the caller's -r flag and `into_directory` its -t.  Both only
    tighten the check, so plan_copy_operations can re-apply this with the
    defaults on arguments a CLI has already accepted and never disagree.
    """
    if not sources:
        raise CopyArgumentError(
            f"missing destination file operand after '{destination}'")

    for src in sources:
        if not os.path.exists(src):
            raise CopyArgumentError(f"cannot stat '{src}': No such file or directory")
        if os.path.isdir(src) and not recursive:
            raise CopyArgumentError(f"-r not specified; omitting directory '{src}'")

    if os.path.isdir(destination):
        return

    # Not a directory, so there is nowhere to put a second source, and no
    # basename to place anything under.  -t says the destination is meant to be
    # a directory to copy into, so it has to already be one.
    if into_directory or len(sources) > 1:
        if os.path.exists(destination):
            raise CopyArgumentError(f"target '{destination}' is not a directory")
        raise CopyArgumentError(f"target '{destination}': No such file or directory")

    if os.path.exists(destination) and os.path.isdir(sources[0]):
        raise CopyArgumentError(
            f"cannot overwrite non-directory '{destination}' "
            f"with directory '{sources[0]}'")


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

    Raises UnreadableEntries if anything under root cannot be stat'd, or if a
    directory under it cannot be listed, reporting every such entry rather than
    dying on the first one.

    FIXME: os.walk(followlinks) doesn't protect against cycles
    to fix this we'd need to write our own version that did a depth-first
    spanning tree.
    """
    file_list = []
    problems = []

    def unlistable(exc):
        # The one unreadable-entry case with no entry to report: files under a
        # directory we cannot list never reach the walk at all, so there is
        # nothing to stat and nothing to collect.  os.walk's default is to
        # swallow this and carry on, which makes an unlistable subtree
        # indistinguishable from an empty one -- a partial copy that exits 0,
        # and, since fastmd5 enumerates through here too, a partial checksum
        # that agrees with it.
        problems.append((exc.filename or root, exc.strerror or str(exc)))

    for dirpath, _, filenames in os.walk(root, followlinks=True, onerror=unlistable):
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
    under its own basename (recursing into directories).  Otherwise there is
    exactly one source and `destination` names it directly: a file is copied to
    that name, and a directory has its *contents* copied in under it, which is
    what `cp -r src newdir` does when newdir does not yet exist.  The result is
    sorted by destination path so that every rank of a collective copy walks
    the files in the same order.

    Raises CopyArgumentError if the arguments are not a combination cp would
    accept, and UnreadableEntries if any source cannot be stat'd or any
    directory under it cannot be listed, reporting all of them together.
    """
    validate_copy_args(sources, destination)

    file_jobs = []
    problems = []
    dst_root = os.path.abspath(destination)

    def add(src_abs, dst_abs):
        st = _stat_or_problem(src_abs, problems)
        if st is not None:
            file_jobs.append((src_abs, dst_abs, st.st_size))

    if not os.path.isdir(dst_root):
        # Validation has established there is exactly one source and, if it is
        # a directory, that nothing is in the way.  `destination` *is* the
        # copy, so a directory's contents go directly under it -- no basename
        # level, which is where cp -r and cp differ.
        src_abs = os.path.abspath(sources[0])
        if os.path.isdir(src_abs):
            try:
                for relpath in list_relative_files(src_abs):
                    add(os.path.join(src_abs, relpath), os.path.join(dst_root, relpath))
            except UnreadableEntries as exc:
                problems.extend(exc.entries)
        else:
            add(src_abs, dst_root)
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
