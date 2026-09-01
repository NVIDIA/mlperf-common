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

"""FileLayout must tile every file exactly once, with aligned boundaries.

Two invariants, both of which corrupt data silently if broken:

  * every byte of the file is claimed by exactly one (local_rank, node, round),
    with no gap and no overlap -- a gap leaves stale bytes, an overlap means two
    ranks writing the same range;
  * every segment boundary is alignment-aligned except the file's own end.
    direct_io.pwrite pads writes up to the block size, so an unaligned interior
    boundary would let one segment's padding overwrite the next segment's bytes.
"""

import sys

import stubs

stubs.install()
ds = stubs.load_fileio()["datastage"]

ALIGN = ds.BUFFER_ALIGN
MiB = 1024 ** 2
GiB = 1024 ** 3

SIZES = [0, 1, 4096, ALIGN - 1, ALIGN, ALIGN + 1, 3 * ALIGN,
         100 * MiB, GiB, 7 * GiB + 12345, 137 * GiB + 999]
NODE_COUNTS = [1, 2, 3, 8, 64, 2048]
RANKS_PER_NODE = [1, 4, 8]
BUFFER_SIZES = [2 * MiB, 8 * MiB, 64 * MiB]


class Topology:
    def __init__(self, node_count, local_world_size, local_rank):
        self.node_count = node_count
        self.local_world_size = local_world_size
        self.local_rank = local_rank


def check(size, nodes, ranks_per_node, buffer_size):
    """Return an error string, or None if this configuration tiles correctly."""
    covered = []
    for local_rank in range(ranks_per_node):
        topology = Topology(nodes, ranks_per_node, local_rank)
        layout = ds.FileLayout(size, topology, buffer_size, ALIGN)
        for round_index in range(layout.rounds):
            for node in range(nodes):
                offset, length = layout.segment(node, round_index)
                if length:
                    covered.append((offset, length))

    covered.sort()
    cursor = 0
    for offset, length in covered:
        if offset != cursor:
            kind = "gap" if offset > cursor else "overlap"
            return f"{kind} at {cursor} (next segment starts at {offset})"
        cursor += length
    if cursor != size:
        return f"covered {cursor} of {size} bytes"

    for offset, length in covered:
        end = offset + length
        if offset % ALIGN:
            return f"segment starts unaligned at {offset}"
        if end % ALIGN and end != size:
            return f"interior segment ends unaligned at {end}"
    return None


def main():
    checked = 0
    failures = []
    for size in SIZES:
        for nodes in NODE_COUNTS:
            for ranks_per_node in RANKS_PER_NODE:
                for buffer_size in BUFFER_SIZES:
                    checked += 1
                    problem = check(size, nodes, ranks_per_node, buffer_size)
                    if problem:
                        failures.append(
                            f"size={size} nodes={nodes} ranks/node={ranks_per_node} "
                            f"-b={buffer_size >> 20}M: {problem}")
    for failure in failures:
        print(f"  FAIL {failure}")
    print(f"test_layout: {checked} configurations, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
