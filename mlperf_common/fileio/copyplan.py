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

__all__ = ["list_relative_files", "plan_copy_operations"]


def list_relative_files(root):
    """Return all file paths under 'root' as relative paths, sorted alphabetically.

    os.walk(followlinks=True) allows following symlinked directories, and
    symlinks to files appear in filenames, so both are dereferenced and copied
    as content rather than recreated as links.

    FIXME: os.walk(followlinks=True) doesn't protect against cycles.  To fix
    this we'd need to write our own version that did a depth-first spanning
    tree.
    """
    file_list = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root)
            file_list.append(rel_path)
    return sorted(file_list)


def plan_copy_operations(sources, destination):
    """Return a list of (src_abs, dst_abs, size_bytes) tuples to copy.

    If `destination` is an existing directory each source is placed inside it
    under its own basename (recursing into directories); otherwise this is a
    single file-to-file copy.  The result is sorted by destination path so that
    every rank of a collective copy walks the files in the same order.
    """
    file_jobs = []
    dst_root = os.path.abspath(destination)

    if not os.path.isdir(dst_root):  # case 1: single file copy
        src_abs = os.path.abspath(sources[0])
        size = os.path.getsize(src_abs)
        file_jobs.append((src_abs, dst_root, size))
    else:
        for src in sources:
            src_abs = os.path.abspath(src)
            base = os.path.basename(src.rstrip("/"))
            if os.path.isdir(src):
                for relpath in list_relative_files(src):
                    full_src = os.path.join(src_abs, relpath)
                    full_dst = os.path.join(dst_root, base, relpath)
                    file_jobs.append((full_src, full_dst, os.path.getsize(full_src)))
            else:
                dst_path = os.path.join(dst_root, base)
                file_jobs.append((src_abs, dst_path, os.path.getsize(src_abs)))

    return sorted(file_jobs, key=lambda job: job[1])
