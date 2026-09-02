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

"""Source-tree walk and src->dst mapping shared by fastcp and fastmd5.

Both take the same cp/rsync-shaped arguments and need the same answer: given
SOURCE(s) and a DEST, which files get copied where.  Keeping that in one place
means a copy and a checksum of that copy can never disagree about what
"copy this directory" means.

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

import collections
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


TreeEntries = collections.namedtuple("TreeEntries", ("files", "directories"))

CopyPlan = collections.namedtuple("CopyPlan", ("files", "directories"))


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

    Nothing here detects symlink cycles (`latest -> .`, `up -> ..`), but they
    do not hang: the kernel allows 40 symlink traversals per path resolution,
    so the descent stops around 40 levels down, in milliseconds.  What that
    looks like is worth knowing, because os.walk hides it -- it wraps
    entry.is_dir() in try/except OSError and treats a failure as "not a
    directory", so the un-openable link is reclassified as a *file* and the
    walk reports no error at all, not even through onerror.  The ELOOP then
    surfaces here, from the stat, as an unreadable entry, and the copy is
    refused.

    That refusal is the safe outcome, so do NOT "fix" this by skipping the
    unreadable entry: the walk enumerates the files under the cycle once per
    level on the way down, so tolerating the ELOOP would turn a loud refusal
    into ~40 redundant copies of everything beneath it.  A real fix means
    detecting the cycle -- a depth-first walk tracking visited (st_dev, st_ino)
    -- which also drops the duplicates.  Not worth it until a dataset actually
    contains such a link.
    """
    return list_relative_entries(root).files


def list_relative_entries(root):
    """Return a TreeEntries(files, directories) of paths relative to `root`.

    One walk yields both.  Directories matter because a copy has to recreate
    an empty one -- there is no file under it to imply it -- and walking a
    second time to find them would double the metadata load on the shared
    filesystem, which for a dataset of millions of files is the expensive part.
    """
    file_list = []
    dir_list = []
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

    for dirpath, dirnames, filenames in os.walk(root, followlinks=True, onerror=unlistable):
        if dirpath != root:
            dir_list.append(os.path.relpath(dirpath, root))
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            if _stat_or_problem(full_path, problems) is not None:
                file_list.append(os.path.relpath(full_path, root))
        dirnames.sort()
    if problems:
        raise UnreadableEntries(problems)
    return TreeEntries(sorted(file_list), sorted(dir_list))


def plan_copy_operations(sources, destination):
    """Return a CopyPlan: (src_abs, dst_abs, size_bytes) jobs, and dirs to create.

    If `destination` is an existing directory each source is placed inside it
    under its own basename (recursing into directories).  Otherwise there is
    exactly one source and `destination` names it directly: a file is copied to
    that name, and a directory has its *contents* copied in under it, which is
    what `cp -r src newdir` does when newdir does not yet exist.  `files` is
    sorted by destination path so that any two tools planning the same copy
    walk it in the same order.

    `directories` holds every destination directory the copy needs, shallowest
    first, and is not derivable from `files`: an empty source directory has no
    file under it to imply its parent, but `cp -r` still creates it.  Create
    them before copying rather than calling makedirs per file.

    Raises CopyArgumentError if the arguments are not a combination cp would
    accept, and UnreadableEntries if any source cannot be stat'd or any
    directory under it cannot be listed, reporting all of them together.
    """
    validate_copy_args(sources, destination)

    file_jobs = []
    dst_dirs = set()
    problems = []
    dst_root = os.path.abspath(destination)

    def add(src_abs, dst_abs):
        st = _stat_or_problem(src_abs, problems)
        if st is not None:
            file_jobs.append((src_abs, dst_abs, st.st_size))
            # Every file implies its parent.  Collecting them here means the
            # caller can create the whole tree up front instead of calling
            # makedirs once per file.
            dst_dirs.add(os.path.dirname(dst_abs))

    if not os.path.isdir(dst_root):
        # Validation has established there is exactly one source and, if it is
        # a directory, that nothing is in the way.  `destination` *is* the
        # copy, so a directory's contents go directly under it -- no basename
        # level, which is where cp -r and cp differ.
        src_abs = os.path.abspath(sources[0])
        if os.path.isdir(src_abs):
            try:
                entries = list_relative_entries(src_abs)
            except UnreadableEntries as exc:
                problems.extend(exc.entries)
            else:
                dst_dirs.add(dst_root)
                for relpath in entries.files:
                    add(os.path.join(src_abs, relpath), os.path.join(dst_root, relpath))
                for relpath in entries.directories:
                    dst_dirs.add(os.path.join(dst_root, relpath))
        else:
            add(src_abs, dst_root)
    else:
        for src in sources:
            src_abs = os.path.abspath(src)
            base = os.path.basename(src.rstrip("/"))
            if os.path.isdir(src):
                try:
                    entries = list_relative_entries(src)
                except UnreadableEntries as exc:
                    problems.extend(exc.entries)
                    continue
                dst_dirs.add(os.path.join(dst_root, base))
                for relpath in entries.files:
                    add(os.path.join(src_abs, relpath),
                        os.path.join(dst_root, base, relpath))
                for relpath in entries.directories:
                    dst_dirs.add(os.path.join(dst_root, base, relpath))
            else:
                add(src_abs, os.path.join(dst_root, base))

    if problems:
        raise UnreadableEntries(problems)
    # Shallowest first, so creating them in order never needs a parent that
    # does not exist yet.
    return CopyPlan(sorted(file_jobs, key=lambda job: job[1]), sorted(dst_dirs))
