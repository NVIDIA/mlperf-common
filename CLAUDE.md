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
  and checksum tools, and the shared tree-walk and I/O primitives behind them.

## Commands

```bash
python3 tests/run_tests.py          # whole suite (stdlib only, no pytest, no GPU)
python3 tests/test_copyplan.py      # one test file, directly

make -C src                         # build the C++ tools
make -C src install prefix=/usr/local

pip install .                       # installs the package + client/ scripts into bin/
```

There is no linter or formatter configured, and no CI in the repo.

Tests run each file in a **separate interpreter** so nothing one leaves in
`sys.modules` reaches another. They are stdlib-only and need no GPU. See
`tests/README.md` for what they do and do not cover — notably not `direct_io`
and not O_DIRECT itself.

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

Two modules, with a deliberate dependency rule stated in
`mlperf_common/fileio/__init__.py`: **neither may import torch.** They stay
stdlib-only so the single-node `client/` scripts don't drag in a training stack.
Anything needing torch belongs in a module of its own.

* `direct_io.py` — aligned-buffer `pread`/`pwrite` with retry loops.
  `allocate_aligned_buffers` carves per-thread buffers out of one big ctypes
  allocation and hands back **memoryviews**, so slicing a buffer to the length
  actually read costs nothing.
* `copyplan.py` — source-tree walk and src→dst mapping. `plan_copy_operations`
  raises `UnreadableEntries` listing *every* bad entry rather than dying on the
  first, because a walk that silently skips an unlistable subtree yields a
  partial copy that exits 0 — and then a checksum run that skips the same files
  and agrees with it.

`BUFFER_ALIGN = 2 MiB` (the huge-page size) is the shared alignment constant
across `fastcp` and `fastmd5`.

This package was extracted from `fastcp` and `fastmd5`, which had been carrying
their own copies. Keep the two scripts going through it rather than reintroducing
private variants.

### client/ and src/

`client/` holds scripts installed onto `PATH` by `setup.py`: `bindpcie` (NUMA/IB
affinity binding), `mgpurun`, `slurm2pytorch` (derives PyTorch rendezvous env
from SLURM), `fastcp` / `fastmd5` (threaded copy and per-GB checksum; `fastcp` opens with
O_DIRECT, `fastmd5` does not),
`dropcache`, plus log/telemetry shell helpers.

**Don't delete `slurm2pytorch`.** Benchmarks outside this repo depend on it and
it stays installed, even though nothing in this repo calls it.

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
