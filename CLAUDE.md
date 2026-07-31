# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mlperf-common` is a grab-bag of utilities shared across NVIDIA's MLPerf benchmark
submissions. Benchmarks consume it by adding
`git+https://github.com/NVIDIA/mlperf-common.git` to their `requirements.txt`.

Two mostly independent halves live here:

* **Logging / profiling** (`mlperf_common/logging.py`, `scaleoutbridge.py`,
  `frameworks/`, `callbacks/`) — wraps the official `mlperf_logging` package.
* **Fast file I/O** (`mlperf_common/fileio/`, `client/`, `src/`) — O_DIRECT copy
  and checksum tools, plus collective dataset staging onto node-local storage.

## Commands

```bash
python3 tests/run_tests.py          # whole suite (stdlib only, no pytest, no GPU)
python3 tests/test_pipeline.py      # one test file, directly

make -C src                         # build the C++ tools
make -C src install prefix=/usr/local

pip install .                       # installs the package + client/ scripts into bin/
```

There is no linter or formatter configured, and no CI in the repo.

Tests deliberately run each file in a **separate interpreter**: each installs its
own fake `torch` (`tests/stubs.py`) into `sys.modules` and patches module-level
names in `datastage`. Don't collapse them into one process. See `tests/README.md`
for what the stubs do and do not cover — notably not NCCL, not real CUDA events,
not O_DIRECT itself.

## Architecture

### Logging stack

`MLLoggerWrapper` (`mlperf_common/logging.py`) is the entry point. It takes a
`CommunicationHandler` so the same wrapper works under either `torch.distributed`
or MPI; `frameworks/base.py` defines the `CommunicationHandler` /
`ProfilerHandler` interfaces and `frameworks/{pyt,mxnet,hugectr,base_mpi}.py`
implement them per framework. Adding framework support means implementing those
two interfaces, not touching the wrapper.

`scaleoutbridge.py` layers profiling on top: `init_bridge(prof_handler,
comm_handler, mllogger)` picks a bridge implementation from env vars —
`TIME_TAGS` / `NVTX_FLAG` select `ScaleoutBridgeIterwise`, `EPOCH_PROF` selects
`ScaleoutBridgeEpochwise`, and with none set you get the no-op
`ScaleoutBridgeBase`.

`callbacks/logging.py` is the Lightning/NeMo layer (`LoggingCallback`,
`MLPerfLogger`, `StatsLogCallback`). It imports `lightning.pytorch` with a
fallback to `pytorch_lightning`, and is knob-driven by env vars
(`FORCE_SUCCESS_STATUS`, `REDUCE_TP`, `LOG_EVERY_N_BATCHES`, `RUN_N_ITERS`,
`SEED`). This is the only part that pulls in Lightning and numpy.

### fileio

Three layers, with a deliberate dependency rule stated in
`mlperf_common/fileio/__init__.py`: **only `datastage` may import torch.**
`direct_io` and `copyplan` stay dependency-free so the single-node `client/`
scripts don't drag in a training stack.

* `direct_io.py` — aligned-buffer `pread`/`pwrite` with retry loops.
* `copyplan.py` — source-tree walk and src→dst mapping. `plan_copy_operations`
  raises `UnreadableEntries` listing *every* bad entry rather than dying on the
  first.
* `datastage.py` — collective staging (below).

`BUFFER_ALIGN = 2 MiB` (the huge-page size) is the shared alignment constant
across `fastcp`, `fastmd5`, and `datastage`.

### datastage — the one thing worth reading before editing

Stages a dataset from shared storage (Lustre) to node-local storage across a
whole job. With W ranks = N nodes × L ranks/node:

* The world splits into L process groups; group `l` holds every node's
  `LOCAL_RANK == l` rank and owns slice `l` of each file.
* Within group `l`, each of the N ranks reads a disjoint 1/N of that slice, and
  an all-gather assembles the full slice on every node. Each rank writes slice
  `l` locally.
* Result: W disjoint readers on the shared FS, one fabric crossing per byte per
  node, L concurrent all-gathers.

`Topology` assumes **slurm's default block distribution**: node `i` holds ranks
`i*L .. i*L+L-1`, so a rank's node is `RANK // L` and its slot is `RANK % L`.
That is arithmetic, requiring no collective. It verifies the assumption locally
— slurm reports RANK (`SLURM_PROCID`) and LOCAL_RANK (`SLURM_LOCALID`)
independently, and they agree only under a block distribution, so a modulo
comparison on each rank rejects `--distribution=cyclic`/`=arbitrary` and ragged
`--ntasks-per-node` immediately.

An earlier version derived the layout from gathered hostnames to support
`--distribution=arbitrary`. That was never a real requirement — a mistake
carried over from the C version — and it could not have worked anyway, since
`MASTER_ADDR` is the first node in the nodelist and rank 0 must be there. Don't
reintroduce discovery here.

`group_ranks[l]` must stay **ascending**: `dist.new_group` sorts the list it is
given and derives each member's group position from that order, while the
drainer maps all-gather output position to `node_index`. Those agree only while
the lists are sorted. `tests/test_topology.py` checks it.

`FileLayout` keeps every offset and length aligned
*except* the final range of the final slice, so O_DIRECT write padding can only
ever run off the end of the file, where a closing `ftruncate` trims it.

The pipeline (`Stager._run_pipeline`) moves buffers between a reader thread, the
collective, a drain thread, and a writer pool through four queues, with CUDA
events for ordering. Getting a handoff wrong does not raise — it writes a
correctly-sized file containing wrong bytes. That is why the tests compare bytes,
not exit status. If you change the pipeline, deliberately reintroduce a bug and
confirm `test_pipeline.py` goes red.

Device and pinned-host memory scale with node count (the window is
`--buffer-size × N`), so `Stager.__init__` budget-checks against 60% of device
memory and fails with a suggested `--buffer-size` rather than OOMing inside CUDA.

Run it as one task per GPU, under `slurm2pytorch` so RANK/WORLD_SIZE/LOCAL_RANK/
MASTER_ADDR are set:

```bash
srun --ntasks-per-node=${DGXNGPU} ... slurm2pytorch \
    python3 -m mlperf_common.fileio.datastage -r "${SLOW_DATADIR}/${DATASET}" "${DATADIR}"
```

`--dry-run` without RANK set prints the copy plan on a single node.

### client/ and src/

`client/` holds scripts installed onto `PATH` by `setup.py`: `bindpcie` (NUMA/IB
affinity binding), `mgpurun`, `slurm2pytorch` (derives PyTorch rendezvous env
from SLURM), `fastcp` / `fastmd5` (threaded O_DIRECT copy and per-GB checksum),
`dropcache`, plus log/telemetry shell helpers.

These scripts import `direct_io` and `mlperf_common.fileio.copyplan` via a
`sys.path` dance that works both for a pip install and for a source tree with
`mlperf_common/` alongside `client/` (the "deploy the repo to a shared filesystem
and run in place" pattern). `client/direct_io.py` is a compatibility shim
re-exporting `mlperf_common.fileio.direct_io`. Copying individual scripts out of
`client/` on their own does not work — preserve that when touching imports, and
if you add a script remember to add it to `scripts=` in `setup.py`.

`src/` is a separate, older C++ implementation (`fastcp`, `cp-into`,
`alloc-empty-file-buffer`) sharing `cmdline.h`. It is not built or installed by
`setup.py`; use the `Makefile` directly.

## Notes

* Commit subjects follow `area: lowercase imperative summary`.
* New files carry the Apache 2.0 header with an NVIDIA copyright line.
* The README's "Mount check" section documents `get-mount-info.sh` /
  `verify-mounts.sh`, which no longer exist — `mlperf_common/mountcheck.py`
  (`--initialize` / verify against `expected-mounts.csv`, sparse SHA256
  fingerprints) replaced them. Trust the module over the README there.
