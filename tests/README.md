# tests

Stdlib only, no pytest, no numpy. Run them with:

    python3 tests/run_tests.py

or run any one directly:

    python3 tests/test_pipeline.py

Each test runs in its own interpreter, because each installs its own fake
`torch` into `sys.modules` and patches module-level names in `datastage`.

## What is here

| file | covers |
| --- | --- |
| `test_layout.py` | `FileLayout` tiles each file exactly once, with aligned boundaries |
| `test_copyplan.py` | tree walk, src→dst mapping, refusal to plan an unreadable tree |
| `test_pipeline.py` | `stage_file` end to end: bytes in == bytes out |
| `stubs.py` | fake `torch` / `torch.distributed` so the above run on a CPU |

## Why the bytes are compared, not just the exit status

`datastage` moves buffers between a reader thread, a collective, a draining
thread and a writer pool through four queues. Getting that handoff wrong does
not raise — it produces a file of exactly the right length containing the wrong
bytes. Two such bugs were caught this way:

* the drain queue was emptied on shutdown before it had been processed, so
  windows already queued were silently dropped. How much of a file survived
  depended on timing.
* nothing trimmed the O_DIRECT write padding off the end of a file, so any file
  whose size was not a multiple of the filesystem block size ended up longer
  than the source, with garbage on the end.

Both pass a "did it throw?" check and both fail a "do the bytes match?" check.
If you change the pipeline, confirm a deliberately reintroduced bug still makes
`test_pipeline.py` fail — a green test that cannot go red is worth nothing.

## What these do NOT cover

`stubs.py` replaces the GPU and the job. So none of this exercises:

* real NCCL, or more than one process — the multi-node cases fake the
  all-gather by filling each node's segment with the bytes that node would have
  read, which is what the real collective delivers, but no data crosses a wire
* real CUDA events, so no ordering bug between a copy and its consumer can show
  up here; on the CPU every "copy" has already landed
* pinned memory, and whether it is aligned enough for a real filesystem block
  size
* O_DIRECT itself — the tests reopen files buffered, since the fake buffers
  cannot promise the alignment it demands
* Lustre, node-local NVMe, and anything about throughput

A green run means the arithmetic and the choreography are right. It says
nothing about whether staging works on a cluster.
