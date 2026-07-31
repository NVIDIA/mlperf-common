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

"""Derive torch.distributed's env:// variables from slurm or mpirun.

torch.distributed.init_process_group(init_method="env://") wants RANK,
WORLD_SIZE, MASTER_ADDR and MASTER_PORT in the environment, and by convention
LOCAL_RANK and LOCAL_WORLD_SIZE alongside them.  Neither slurm nor mpirun sets
those names, so something has to translate.  `client/slurm2pytorch` does it in
bash and execs the real command; this is the same translation in Python, so a
program can do it for itself instead of being wrapped.

The two must stay behaviourally compatible: a program launched under
slurm2pytorch finds the variables already set, and takes them as given.

Two deliberate divergences from the script, both in the direction of failing
loudly rather than proceeding on a guess:

  * slurm2pytorch defaults LOCAL_WORLD_SIZE to 1.  But SLURM_NTASKS_PER_NODE is
    only set when --ntasks-per-node was actually passed, so `srun -N2 -n16`
    has no source for it and that default turns two 8-GPU nodes into sixteen
    single-rank "nodes" -- which reads as a valid job right up until the copy
    comes out wrong.  We consult SLURM_TASKS_PER_NODE, which srun always sets,
    and refuse to guess on a multi-rank job.
  * slurm2pytorch falls back to MASTER_ADDR=127.0.0.1, which its own comment
    notes "will fail for multinode" -- as a rendezvous that hangs until the
    job's wall clock runs out.  We parse the address out of slurm's nodelist,
    and if that fails on a multi-node job we say so immediately.

There is deliberately no PYTORCH_VERSION gate.  slurm2pytorch has one because
it wraps arbitrary commands in arbitrary containers and must be a no-op outside
a pytorch one.  A caller importing this module is already torch code.
"""

import collections
import os

__all__ = ["DistEnv", "DistEnvError", "MASTER_PORT_DEFAULT", "configure",
           "describe", "detect", "first_hostname"]

# Static rendezvous port.  torch.distributed's TCPStore does not handle races
# for dynamic port assignment gracefully, so a fixed port is safer than an
# ephemeral one.  29500 is what torch.distributed.run uses, is unassigned by
# IANA, and sits outside both the IANA ephemeral range (49152-65535) and
# Ubuntu's default (32768-60999).  If your site puts a service on 29500, or
# includes it in the local ephemeral range (`cat
# /proc/sys/net/ipv4/ip_local_port_range`), set MASTER_PORT explicitly;
# 61000-65535 is usually a good alternative on Linux.
MASTER_PORT_DEFAULT = 29500

_LOCALHOST = "127.0.0.1"

DistEnv = collections.namedtuple(
    "DistEnv",
    "rank world_size local_rank local_world_size master_addr master_port source")


class DistEnvError(RuntimeError):
    """The environment does not describe a job we can join.

    Carries a message without a program name; callers prefix their own.
    """


def _value(environ, *names):
    """First of `names` with a non-empty value, as bash ${X:-...} would treat it.

    environ.get() alone is not enough: an exported-but-empty variable is a
    real thing to find in a job script, and bash's :- skips it.
    """
    for name in names:
        text = environ.get(name)
        if text is not None and text.strip():
            return name, text.strip()
    return None, None


def _int(environ, *names):
    """_value(), converted, naming the variable that was actually wrong."""
    name, text = _value(environ, *names)
    if name is None:
        return None, None
    try:
        return name, int(text)
    except ValueError:
        raise DistEnvError(f"{name}={text!r} is not an integer") from None


def first_hostname(nodelist):
    """First hostname of a slurm hostlist, or None if it cannot be read.

    Slurm hands the allocation over compressed -- "dgx[001-004,007]",
    "a,b", "rack[1-2]node[3-4]" -- and `scontrol show hostnames` is not
    available inside a container.  Only the first name is ever needed (rank 0
    runs there under a block distribution), and that is much less work than
    expanding the list.

    Two things a naive nodelist.split(",")[0] gets wrong:

      * commas inside brackets.  "dgx[001-004,007]" is one entry, not two, and
        splitting yields "dgx[001".
      * zero padding.  The low bound is appended as the literal token it was
        written as, never through int(), because "dgx[001-004]" is dgx001 and
        dgx1 does not resolve.

    Returns None rather than raising: an unparseable nodelist is only fatal if
    nothing else supplies an address, and that is the caller's judgement.
    """
    if not nodelist or not nodelist.strip():
        return None

    text = nodelist.strip()
    out = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == ",":          # top-level comma: end of the first entry
            break
        if char == "[":
            close = text.find("]", i)
            if close < 0:
                return None      # unbalanced
            low = text[i + 1:close].split(",")[0].split("-")[0].strip()
            if not low.isdigit():
                return None
            out.append(low)
            i = close + 1
            continue
        if char == "]":
            return None          # stray close bracket
        out.append(char)
        i += 1

    name = "".join(out).strip()
    if not name or not all(c.isalnum() or c in "-_." for c in name):
        return None
    return name


def _tasks_per_node(text):
    """Expand SLURM_TASKS_PER_NODE ("8", "8(x2)", "4,8", "2(x3),1") to a list.

    Unlike SLURM_NTASKS_PER_NODE this is set whenever srun runs, which is what
    makes it a usable fallback.  Returns None if it cannot be parsed.
    """
    counts = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            return None
        if "(" in part:
            count, _, repeat = part.partition("(")
            repeat = repeat.strip().rstrip(")").lstrip("xX")
            if not count.strip().isdigit() or not repeat.isdigit():
                return None
            counts.extend([int(count)] * int(repeat))
        else:
            if not part.isdigit():
                return None
            counts.append(int(part))
    return counts or None


def _local_world_size(environ, world_size):
    """Ranks per node, refusing to guess it on a multi-rank job."""
    name, value = _int(environ, "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE",
                       "OMPI_COMM_WORLD_LOCAL_SIZE")
    if name is not None:
        return value

    text = environ.get("SLURM_TASKS_PER_NODE", "").strip()
    if text:
        counts = _tasks_per_node(text)
        if counts and len(set(counts)) == 1:
            return counts[0]
        if counts:
            raise DistEnvError(
                f"SLURM_TASKS_PER_NODE={text!r} describes a ragged allocation "
                f"({sorted(set(counts))} tasks per node); datastage needs the "
                "same number of ranks on every node"
            )

    if world_size == 1:
        return 1
    raise DistEnvError(
        f"cannot determine LOCAL_WORLD_SIZE for a WORLD_SIZE={world_size} job; "
        "launch with a uniform --ntasks-per-node (one task per GPU), or set "
        "LOCAL_WORLD_SIZE explicitly"
    )


def _master_addr(environ, world_size, local_world_size):
    """Where rank 0 is, which is the first node of the allocation."""
    _, addr = _value(environ, "MASTER_ADDR", "MLPERF_SLURM_FIRSTNODE")
    if addr:
        return addr

    # The step's nodelist first: a step may run on a subset of the job's nodes,
    # and rank 0 of *this step* is on the first of those.  SLURM_NODELIST is
    # the older alias, kept last.
    for name in ("SLURM_STEP_NODELIST", "SLURM_JOB_NODELIST", "SLURM_NODELIST"):
        host = first_hostname(environ.get(name))
        if host:
            return host

    if world_size <= local_world_size:
        return _LOCALHOST

    seen = {name: environ.get(name) for name in
            ("SLURM_STEP_NODELIST", "SLURM_JOB_NODELIST", "SLURM_NODELIST")
            if environ.get(name)}
    raise DistEnvError(
        f"cannot determine MASTER_ADDR for a multi-node job "
        f"(WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size}); "
        f"nodelist variables are {seen or 'unset'}. Set MASTER_ADDR, or set "
        'MLPERF_SLURM_FIRSTNODE from the host with $(scontrol show hostnames '
        '"$SLURM_JOB_NODELIST" | head -n1)'
    )


def _check_sources_agree(environ, rank, world_size):
    """Catch an environment describing two different jobs at once.

    slurm and mpirun each set their own variables, and a command line that
    mixes launchers leaves both -- with only one of them true.  Preferring
    slurm silently would pick a rank ordering the other half of the job does
    not share.
    """
    _, ompi_rank = _int(environ, "OMPI_COMM_WORLD_RANK")
    _, ompi_world = _int(environ, "OMPI_COMM_WORLD_SIZE")
    _, slurm_rank = _int(environ, "SLURM_PROCID")
    _, slurm_world = _int(environ, "SLURM_NTASKS")

    if slurm_rank is None or ompi_rank is None:
        return
    if slurm_rank != ompi_rank or (
            slurm_world is not None and ompi_world is not None
            and slurm_world != ompi_world):
        raise DistEnvError(
            f"slurm and mpirun disagree about this process: "
            f"SLURM_PROCID={slurm_rank} of {slurm_world}, "
            f"OMPI_COMM_WORLD_RANK={ompi_rank} of {ompi_world}. Set RANK and "
            "WORLD_SIZE explicitly to say which is right"
        )


def _check_slurm_arithmetic(environ):
    """SLURM_NNODES * SLURM_NTASKS_PER_NODE has to be SLURM_NTASKS."""
    _, nodes = _int(environ, "SLURM_NNODES", "SLURM_JOB_NUM_NODES")
    _, per_node = _int(environ, "SLURM_NTASKS_PER_NODE")
    _, tasks = _int(environ, "SLURM_NTASKS")
    if None in (nodes, per_node, tasks):
        return
    if nodes * per_node != tasks:
        raise DistEnvError(
            f"slurm's own numbers do not agree: SLURM_NNODES={nodes} x "
            f"SLURM_NTASKS_PER_NODE={per_node} is {nodes * per_node}, but "
            f"SLURM_NTASKS={tasks}"
        )


def detect(environ=None):
    """Work out the job layout without touching the environment.

    Validation here is about the *sources*: that they are present, agree with
    each other, and describe a coherent job.  Whether the resulting layout is
    the block distribution datastage needs is Topology's business, and it
    still checks -- someone can set these variables by hand without coming
    through here.
    """
    environ = os.environ if environ is None else environ

    _check_slurm_arithmetic(environ)

    world_name, world_size = _int(environ, "WORLD_SIZE", "SLURM_NTASKS",
                                  "OMPI_COMM_WORLD_SIZE")
    if world_name is None:
        world_size = 1
    rank_name, rank = _int(environ, "RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK")
    if rank_name is None:
        rank = 0
    local_name, local_rank = _int(environ, "LOCAL_RANK", "SLURM_LOCALID",
                                  "OMPI_COMM_WORLD_LOCAL_RANK")
    if local_name is None:
        local_rank = 0

    if rank_name != "RANK":
        _check_sources_agree(environ, rank, world_size)

    source = {"RANK": "preset", "SLURM_PROCID": "slurm",
              "OMPI_COMM_WORLD_RANK": "ompi"}.get(rank_name, "single")

    local_world_size = _local_world_size(environ, world_size)

    if world_size < 1:
        raise DistEnvError(f"WORLD_SIZE={world_size} must be at least 1")
    if not 0 <= rank < world_size:
        raise DistEnvError(f"RANK={rank} is outside WORLD_SIZE={world_size}")
    if local_world_size < 1:
        raise DistEnvError(
            f"LOCAL_WORLD_SIZE={local_world_size} must be at least 1")
    if not 0 <= local_rank < local_world_size:
        raise DistEnvError(
            f"LOCAL_RANK={local_rank} is outside "
            f"LOCAL_WORLD_SIZE={local_world_size}")
    if world_size % local_world_size:
        raise DistEnvError(
            f"WORLD_SIZE={world_size} is not a multiple of "
            f"LOCAL_WORLD_SIZE={local_world_size}; launch with a uniform "
            "--ntasks-per-node"
        )

    _, master_port = _int(environ, "MASTER_PORT")
    if master_port is None:
        master_port = MASTER_PORT_DEFAULT
    if not 1 <= master_port <= 65535:
        raise DistEnvError(f"MASTER_PORT={master_port} is not a valid port")

    return DistEnv(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        master_addr=_master_addr(environ, world_size, local_world_size),
        master_port=master_port,
        source=source,
    )


def configure(environ=None):
    """detect(), then write the result where torch and Topology will read it.

    Setting the environment is the point, not a side effect:
    init_process_group(init_method="env://") reads os.environ, and so does
    Topology.
    """
    environ = os.environ if environ is None else environ
    env = detect(environ)

    environ["RANK"] = str(env.rank)
    environ["WORLD_SIZE"] = str(env.world_size)
    environ["LOCAL_RANK"] = str(env.local_rank)
    environ["LOCAL_WORLD_SIZE"] = str(env.local_world_size)
    environ["MASTER_ADDR"] = env.master_addr
    environ["MASTER_PORT"] = str(env.master_port)

    # As torch.distributed.run does: without this every rank on a node spawns
    # a full set of OpenMP threads and they fight over the same cores.
    if env.local_world_size > 1 and not environ.get("OMP_NUM_THREADS", "").strip():
        environ["OMP_NUM_THREADS"] = "1"

    return env


def describe(env):
    """One line, worth printing before the rendezvous rather than after."""
    nodes = env.world_size // env.local_world_size
    return (f"{env.source} launch: {env.world_size} ranks = {nodes} nodes x "
            f"{env.local_world_size} ranks/node, rendezvous at "
            f"{env.master_addr}:{env.master_port}")
