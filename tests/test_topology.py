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

"""Topology's node grouping, which nothing could construct before.

datastage assumes slurm's default block distribution: node i holds ranks
i*L .. i*L+L-1.  Getting the grouping wrong does not raise -- it builds groups
whose all-gather assembles the right bytes in the wrong order, so every file
comes out the right length with its slices permuted.

Two things are checked.  That the grouping is right: every rank agrees with
every other about which global ranks make up each group, each group holds one
rank per node, and every rank appears exactly once across the groups.  And
that the group lists are ascending -- dist.new_group sorts what it is given
and derives each member's position from that order, while the drainer maps
all-gather output position to node_index, so those two agree only while the
lists are sorted to begin with.
"""

import os
import sys

import stubs

torch, dist = stubs.install()
ds = stubs.load_fileio()["datastage"]

results = []


def check(name, condition, detail=""):
    results.append((name, condition, detail))


def topology_for(rank, nodes, ranks_per_node, **overrides):
    """Build Topology as `rank` would see it in a nodes x ranks_per_node job."""
    env = {
        "RANK": str(rank),
        "WORLD_SIZE": str(nodes * ranks_per_node),
        "LOCAL_RANK": str(rank % ranks_per_node),
        "LOCAL_WORLD_SIZE": str(ranks_per_node),
    }
    env.update({k: str(v) for k, v in overrides.items()})
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return ds.Topology()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def check_layout(nodes, ranks_per_node):
    world = nodes * ranks_per_node
    label = f"{nodes}x{ranks_per_node}"
    views = [topology_for(rank, nodes, ranks_per_node) for rank in range(world)]

    check(f"{label}: every rank sees the same groups",
          len({tuple(tuple(g) for g in view.groups) for view in views}) == 1)

    groups = views[0].groups
    check(f"{label}: one group per local rank", len(groups) == ranks_per_node)
    check(f"{label}: each group has one rank per node",
          all(len(g) == nodes for g in groups))
    check(f"{label}: every rank appears exactly once",
          sorted(r for g in groups for r in g) == list(range(world)))
    check(f"{label}: group lists are ascending",
          all(list(g) == sorted(g) for g in groups),
          f"got {groups}")

    for rank, view in enumerate(views):
        if view.node_index != rank // ranks_per_node:
            check(f"{label}: rank {rank} knows its node", False,
                  f"node_index {view.node_index}")
            return
        if view.rank not in view.group:
            check(f"{label}: rank {rank} is a member of its own group", False)
            return
        # The position the all-gather will place this rank's segment at has to
        # be its node_index, or the drainer writes segments to other nodes'
        # offsets.
        if list(view.group).index(rank) != view.node_index:
            check(f"{label}: rank {rank} sits at its node_index in its group",
                  False, f"position {list(view.group).index(rank)}")
            return
    check(f"{label}: every rank sits at its node_index in its own group", True)


def check_rejected(name, **kwargs):
    try:
        topology_for(**kwargs)
        check(name, False, "no exception raised")
    except RuntimeError:
        check(name, True)


def main():
    for nodes, ranks_per_node in ((1, 1), (1, 8), (2, 8), (3, 4), (64, 8), (17, 3)):
        check_layout(nodes, ranks_per_node)

    # A cyclic distribution reports a LOCAL_RANK that disagrees with RANK % L.
    # Rank 1 of a 2x2 job is local rank 0 on node 1 under cyclic, but block
    # arithmetic puts it at local rank 1.
    check_rejected("a cyclic distribution is refused",
                   rank=1, nodes=2, ranks_per_node=2, LOCAL_RANK=0)
    check_rejected("a ragged --ntasks-per-node is refused",
                   rank=0, nodes=2, ranks_per_node=8, WORLD_SIZE=15)
    check_rejected("LOCAL_WORLD_SIZE of 0 is refused",
                   rank=0, nodes=1, ranks_per_node=1, LOCAL_WORLD_SIZE=0)

    # A missing LOCAL_WORLD_SIZE used to default to 1, silently turning an
    # 8-GPU node into 8 single-rank nodes.
    try:
        saved = os.environ.pop("LOCAL_WORLD_SIZE", None)
        os.environ.update({"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "3"})
        ds.Topology()
        check("a missing LOCAL_WORLD_SIZE is refused", False, "no exception")
    except KeyError:
        check("a missing LOCAL_WORLD_SIZE is refused", True)
    except RuntimeError:
        check("a missing LOCAL_WORLD_SIZE is refused", True)
    finally:
        if saved is not None:
            os.environ["LOCAL_WORLD_SIZE"] = saved

    failures = [r for r in results if not r[1]]
    for name, _, detail in failures:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
    print(f"test_topology: {len(results)} checks, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
