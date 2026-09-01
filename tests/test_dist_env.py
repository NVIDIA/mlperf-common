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

"""The slurm/mpirun -> torch env:// translation datastage does for itself.

Two things are worth pinning down here.  The first is that this agrees with
`client/slurm2pytorch`, which other benchmarks still use: a program launched
under the wrapper must take the wrapper's values as given, not recompute them
into something different.

The second is the failure modes.  Everything this module derives feeds a
rendezvous, and a rendezvous that is wrong does not return an error -- it
hangs until the job's wall clock runs out, which is the worst way to find out
that --ntasks-per-node was missing.  So the cases that must raise get as much
attention as the cases that must work.

This is the one test file that does not use stubs.py.  dist_env is stdlib
only, so it can be imported directly, and detect() takes the environment as an
argument -- meaning almost everything here is a plain dict rather than a dance
around os.environ.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlperf_common import dist_env  # noqa: E402

results = []


def check(name, condition, detail=""):
    results.append((name, condition, detail))


def check_raises(name, environ):
    try:
        env = dist_env.detect(environ)
        check(name, False, f"no exception; got {env}")
        return None
    except dist_env.DistEnvError as exc:
        check(name, True)
        return str(exc)


def check_hostnames():
    """Slurm's compressed hostlist, of which we only ever want the first name."""
    cases = [
        ("dgx001", "dgx001"),
        ("dgx[001-004,007]", "dgx001"),      # comma inside brackets
        ("dgx[007,001-004]", "dgx007"),      # literal first, not sorted
        ("dgx[001]", "dgx001"),
        ("a,b", "a"),
        ("node[01-04],other[7-9]", "node01"),
        ("nid[1-2]-ib", "nid1-ib"),          # suffix after the bracket
        ("rack[1-2]node[3-4]", "rack1node3"),  # two groups, one name
        (" dgx001 ", "dgx001"),
        ("dgx001,", "dgx001"),
        ("", None),
        ("   ", None),
        (None, None),
        ("dgx[001-004", None),               # unbalanced
        ("dgx[]", None),
        ("dgx[-4]", None),
        ("dgx[a-b]", None),
        ("dgx001]", None),                   # stray close
    ]
    for nodelist, expected in cases:
        got = dist_env.first_hostname(nodelist)
        check(f"first_hostname({nodelist!r}) == {expected!r}", got == expected,
              f"got {got!r}")

    # Zero padding has to survive: dgx1 does not resolve.
    check("zero padding is preserved",
          dist_env.first_hostname("dgx[0007-0009]") == "dgx0007")


def check_srun():
    """A plain srun launch, no wrapper -- the case this module exists for."""
    environ = {
        "SLURM_PROCID": "9",
        "SLURM_NTASKS": "16",
        "SLURM_LOCALID": "1",
        "SLURM_NTASKS_PER_NODE": "8",
        "SLURM_NNODES": "2",
        "SLURM_JOB_NODELIST": "dgx[001-002]",
    }
    env = dist_env.detect(environ)
    check("srun: rank", env.rank == 9, f"got {env.rank}")
    check("srun: world size", env.world_size == 16)
    check("srun: local rank", env.local_rank == 1)
    check("srun: local world size", env.local_world_size == 8)
    check("srun: master addr from the nodelist", env.master_addr == "dgx001",
          f"got {env.master_addr}")
    check("srun: default port", env.master_port == 29500)
    check("srun: source", env.source == "slurm", f"got {env.source}")
    check("srun: describe names the rendezvous",
          "dgx001:29500" in dist_env.describe(env),
          dist_env.describe(env))


def check_precedence():
    base = {
        "SLURM_PROCID": "0", "SLURM_NTASKS": "16", "SLURM_LOCALID": "0",
        "SLURM_NTASKS_PER_NODE": "8", "SLURM_JOB_NODELIST": "job[001-002]",
    }
    check("job nodelist is used when nothing better exists",
          dist_env.detect(base).master_addr == "job001")

    with_step = dict(base, SLURM_STEP_NODELIST="step[005-006]")
    check("step nodelist beats job nodelist",
          dist_env.detect(with_step).master_addr == "step005")

    with_mlperf = dict(with_step, MLPERF_SLURM_FIRSTNODE="fromhost")
    check("MLPERF_SLURM_FIRSTNODE beats both nodelists",
          dist_env.detect(with_mlperf).master_addr == "fromhost")

    with_master = dict(with_mlperf, MASTER_ADDR="explicit")
    check("MASTER_ADDR beats everything",
          dist_env.detect(with_master).master_addr == "explicit")


def check_mpirun():
    environ = {
        "OMPI_COMM_WORLD_RANK": "3",
        "OMPI_COMM_WORLD_SIZE": "8",
        "OMPI_COMM_WORLD_LOCAL_RANK": "3",
        "OMPI_COMM_WORLD_LOCAL_SIZE": "8",
    }
    env = dist_env.detect(environ)
    check("mpirun: source", env.source == "ompi", f"got {env.source}")
    check("mpirun: rank", env.rank == 3)
    check("mpirun: single node gets localhost", env.master_addr == "127.0.0.1")


def check_under_slurm2pytorch():
    """The wrapper still works: its values are taken as given, not recomputed.

    The slurm variables here deliberately disagree with the preset ones.  If
    anything recomputed rather than deferred, this is where it would show.
    """
    environ = {
        "RANK": "5", "WORLD_SIZE": "16", "LOCAL_RANK": "5",
        "LOCAL_WORLD_SIZE": "8", "MASTER_ADDR": "wrapper-host",
        "MASTER_PORT": "29500",
        "SLURM_PROCID": "11", "SLURM_NTASKS": "32", "SLURM_LOCALID": "3",
        "SLURM_NTASKS_PER_NODE": "4", "SLURM_JOB_NODELIST": "other[001-008]",
    }
    env = dist_env.detect(environ)
    check("wrapper: source is preset", env.source == "preset", f"got {env.source}")
    check("wrapper: rank taken as given", env.rank == 5, f"got {env.rank}")
    check("wrapper: world size taken as given", env.world_size == 16)
    check("wrapper: local rank taken as given", env.local_rank == 5)
    check("wrapper: local world size taken as given", env.local_world_size == 8)
    check("wrapper: master addr taken as given",
          env.master_addr == "wrapper-host")


def check_empty_and_single():
    check("an exported-but-empty variable counts as unset",
          dist_env.detect({"RANK": "", "SLURM_PROCID": "3",
                           "SLURM_NTASKS": "8", "SLURM_LOCALID": "3",
                           "SLURM_NTASKS_PER_NODE": "8"}).rank == 3)

    env = dist_env.detect({})
    check("no launcher at all: source is single", env.source == "single",
          f"got {env.source}")
    check("no launcher at all: rank 0 of 1", (env.rank, env.world_size) == (0, 1))
    check("no launcher at all: localhost", env.master_addr == "127.0.0.1")

    env = dist_env.detect({"SLURM_PROCID": "2", "SLURM_NTASKS": "8",
                           "SLURM_LOCALID": "2", "SLURM_NTASKS_PER_NODE": "8"})
    check("single node with no nodelist falls back to localhost",
          env.master_addr == "127.0.0.1")


def check_refusals():
    """The launches we cannot serve, which must not turn into a hung rendezvous."""
    message = check_raises(
        "a multi-node job with no derivable master is refused",
        {"SLURM_PROCID": "0", "SLURM_NTASKS": "16", "SLURM_LOCALID": "0",
         "SLURM_NTASKS_PER_NODE": "8", "SLURM_JOB_NODELIST": "dgx[001"})
    if message:
        check("that message says how to fix it",
              "MLPERF_SLURM_FIRSTNODE" in message, message)

    message = check_raises(
        "srun -N2 -n16 with no --ntasks-per-node is refused",
        {"SLURM_PROCID": "0", "SLURM_NTASKS": "16", "SLURM_LOCALID": "0",
         "SLURM_JOB_NODELIST": "dgx[001-002]"})
    if message:
        check("that message names --ntasks-per-node",
              "--ntasks-per-node" in message, message)

    # Guarded, because the interesting way for this to break is for detect()
    # to start raising -- and an exception thrown inside a check() argument
    # takes the whole file down with a traceback instead of naming itself.
    try:
        supplied = dist_env.detect(
            {"SLURM_PROCID": "0", "SLURM_NTASKS": "16", "SLURM_LOCALID": "0",
             "SLURM_TASKS_PER_NODE": "8(x2)",
             "SLURM_JOB_NODELIST": "dgx[001-002]"}).local_world_size
        check("SLURM_TASKS_PER_NODE supplies the missing ranks-per-node",
              supplied == 8, f"got {supplied}")
    except dist_env.DistEnvError as exc:
        check("SLURM_TASKS_PER_NODE supplies the missing ranks-per-node",
              False, f"raised {exc}")

    check_raises(
        "a ragged SLURM_TASKS_PER_NODE is refused",
        {"SLURM_PROCID": "0", "SLURM_NTASKS": "12", "SLURM_LOCALID": "0",
         "SLURM_TASKS_PER_NODE": "4,8", "SLURM_JOB_NODELIST": "dgx[001-002]"})

    check_raises(
        "slurm's own numbers not multiplying out is refused",
        {"SLURM_PROCID": "0", "SLURM_NTASKS": "15", "SLURM_LOCALID": "0",
         "SLURM_NNODES": "2", "SLURM_NTASKS_PER_NODE": "8",
         "SLURM_JOB_NODELIST": "dgx[001-002]"})

    check_raises(
        "slurm and mpirun disagreeing is refused",
        {"SLURM_PROCID": "0", "SLURM_NTASKS": "16", "SLURM_LOCALID": "0",
         "SLURM_NTASKS_PER_NODE": "8", "SLURM_JOB_NODELIST": "dgx[001-002]",
         "OMPI_COMM_WORLD_RANK": "4", "OMPI_COMM_WORLD_SIZE": "16"})

    check_raises("a non-integer count is refused",
                 {"SLURM_PROCID": "0", "SLURM_NTASKS": "x"})

    check_raises("a rank outside the world is refused",
                 {"RANK": "9", "WORLD_SIZE": "8", "LOCAL_RANK": "1",
                  "LOCAL_WORLD_SIZE": "8"})

    check_raises("a local rank outside its node is refused",
                 {"RANK": "1", "WORLD_SIZE": "8", "LOCAL_RANK": "9",
                  "LOCAL_WORLD_SIZE": "8"})

    check_raises("a world that is not a multiple of the node size is refused",
                 {"RANK": "1", "WORLD_SIZE": "15", "LOCAL_RANK": "1",
                  "LOCAL_WORLD_SIZE": "8", "MASTER_ADDR": "h"})

    check_raises("an impossible port is refused",
                 {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
                  "LOCAL_WORLD_SIZE": "1", "MASTER_PORT": "99999"})


def check_configure():
    environ = {
        "SLURM_PROCID": "9", "SLURM_NTASKS": "16", "SLURM_LOCALID": "1",
        "SLURM_NTASKS_PER_NODE": "8", "SLURM_JOB_NODELIST": "dgx[001-002]",
    }
    dist_env.configure(environ)
    expected = {"RANK": "9", "WORLD_SIZE": "16", "LOCAL_RANK": "1",
                "LOCAL_WORLD_SIZE": "8", "MASTER_ADDR": "dgx001",
                "MASTER_PORT": "29500"}
    for key, value in expected.items():
        check(f"configure sets {key}", environ.get(key) == value,
              f"got {environ.get(key)!r}")
    check("configure sets OMP_NUM_THREADS on a multi-rank node",
          environ.get("OMP_NUM_THREADS") == "1")

    kept = dict(environ, OMP_NUM_THREADS="4")
    dist_env.configure(kept)
    check("configure leaves an existing OMP_NUM_THREADS alone",
          kept["OMP_NUM_THREADS"] == "4")

    single = {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
              "LOCAL_WORLD_SIZE": "1"}
    dist_env.configure(single)
    check("configure leaves OMP_NUM_THREADS unset for one rank per node",
          "OMP_NUM_THREADS" not in single)


def check_topology_contract():
    """What configure() writes is what Topology reads back.

    Topology does int(os.environ[name]) for these four and validates the block
    layout itself; this only checks the handover, not the layout.
    """
    keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
            "MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS",
            "SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID",
            "SLURM_NTASKS_PER_NODE", "SLURM_JOB_NODELIST")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update({
            "SLURM_PROCID": "9", "SLURM_NTASKS": "16", "SLURM_LOCALID": "1",
            "SLURM_NTASKS_PER_NODE": "8", "SLURM_JOB_NODELIST": "dgx[001-002]",
        })
        env = dist_env.configure()
        readback = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]),
                    int(os.environ["LOCAL_RANK"]),
                    int(os.environ["LOCAL_WORLD_SIZE"]))
        check("os.environ round-trip agrees with the returned values",
              readback == (env.rank, env.world_size, env.local_rank,
                           env.local_world_size),
              f"got {readback}")
        check("the round-trip satisfies Topology's block-layout check",
              readback[0] % readback[3] == readback[2])
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    check_hostnames()
    check_srun()
    check_precedence()
    check_mpirun()
    check_under_slurm2pytorch()
    check_empty_and_single()
    check_refusals()
    check_configure()
    check_topology_contract()

    failures = [r for r in results if not r[1]]
    for name, _, detail in failures:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
    print(f"test_dist_env: {len(results)} checks, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
