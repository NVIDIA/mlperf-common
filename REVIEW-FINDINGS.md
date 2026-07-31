# Code review findings — branch `mfrank/py-data-staging`

Working file, not part of the deliverable. Delete it when the list is worked
through. Raw verifier output is in `.review-findings.json` next to this file.

Reviewed diff: `git diff main...HEAD` (merge-base `c02c93c`, 13 files,
+1766/-212). Review ran at `max` effort: 49 agents, 53 raw findings deduplicated
to the 15 below, each independently verified. Verdicts were `CONFIRMED` across
the board, but see the premise correction immediately below — it demotes one of
them, and the verifiers did not know it.

## Premise correction (2026-07-31)

**datastage does not need to support `srun --distribution=arbitrary`.** The
review assumed it did, because `Topology`'s comments say so. That requirement
was a mistake by the original author of the C version and was carried into the
Python port unexamined; it is not a real constraint.

Consequences:

* **F2 is demoted** from live corruption to latent fragility — its repro needs a
  non-ascending `group_ranks[l]`, which block/cyclic distribution never
  produces. Worth an assert, not a panic.
* `Topology`'s `all_gather_object` hostname derivation exists to serve this
  non-requirement. Simplifying it is open for discussion; it is not in scope of
  any finding below.
* `CLAUDE.md` stated the arbitrary-distribution rationale as a design invariant.
  Corrected in the same change that created this file.

## Status legend

`[ ]` open · `[x]` fixed · `[~]` intentionally skipped · `[-]` no change needed

---

## Tier 1 — silent data corruption

- [x] **F1 · `mlperf_common/fileio/datastage.py:476` · set_device missing on drainer thread** — fixed, see "F1 fix" below

  The drain thread never calls `torch.cuda.set_device`. CUDA's current device is
  per-host-thread, so on any rank with `LOCAL_RANK != 0` the thread sits on
  device 0. The `copy_` itself lands on the right device via the tensor's device
  guard, but `torch.cuda.Event()` + `record()` resolve `current_stream()` against
  the *thread's* device, so the event records on device 0's idle default stream.
  `_write_one`'s `ready.synchronize()` (:338) and `last_copy.synchronize()`
  (:494) then return immediately.

  Effect: `pwrite` reads the pinned chunk before the D2H copy lands, and
  `recv_dev` slots are recycled while still being read. On an 8-GPU node, 7 of 8
  slices of every staged file get zeros or stale bytes. Right length, wrong
  contents, exit 0.

  Fires on every multi-GPU node — independent of the distribution correction
  above. Fix first.

  **F1 fix (2026-07-31).** `torch.cuda.set_device(self.device)` at the top of
  `drainer()`. Guarded by `tests/test_device.py`, which models a per-thread
  current device in the stubs and asserts every recorded event belongs to this
  rank's device. Confirmed red before the fix (3 of 9 events on device 0 — the
  drainer's, one per round) and green after.

  Still unverified on hardware: no GPU was available. The test checks that the
  threads agree on which device they are on, not CUDA ordering semantics. Worth
  a multi-GPU confirmation run when a node is free.

- [ ] **F2 · `mlperf_common/fileio/datastage.py:199` · new_group sorts its rank list** *(demoted — see premise correction)*

  `dist.new_group` does `ranks = sorted(ranks)` then
  `group_rank = ranks.index(global_rank)`, so a member's group rank is its
  position in *sorted* order. The drainer maps all-gather output position `i` to
  `layout.segment(i, ...)`, assuming position == `node_index`, where
  `node_index` is hostname first-appearance order.

  These agree whenever RANK is block- or cyclic-distributed, which is now the
  only supported case. Under a hypothetical `A B B A` layout `group_ranks[1] =
  [3, 2]` sorts to `[2, 3]` and two nodes' sub-shards swap.

  Recommended: assert `group_ranks[l] == sorted(group_ranks[l])` and fail loudly,
  rather than carrying the machinery that pretends to handle the general case.

## Tier 2 — silent data loss / job hangs

- [x] **F3 · `mlperf_common/fileio/copyplan.py:82` · os.walk swallows unreadable subtrees** — fixed, see "F3 fix" below

  Default `onerror=None` means a directory that can't be listed is skipped
  silently, instead of raising `UnreadableEntries` — the exact failure this
  module was written to prevent.

  Verified: `src/{good/a.bin, secret/b.bin}` with `secret/` at mode 000 returns
  `['good/a.bin']`, no exception. Unlistable root returns `[]`, also silent.

  Compounding: fastmd5 now walks through the same function, so verification of a
  partially staged tree omits the same files and passes.

  **Not a regression.** `os.walk(followlinks=True)` with no `onerror` was
  already in `client/fastcp` at `c02c93c`, before the fileio refactor; copyplan
  inherited it verbatim. `4f1c063` made *stat* failures noisy and did so
  correctly — the miss is that an unlistable directory produces no entry to
  stat, so it fails by omission and there is nothing for a stat-oriented check
  to collect. What the branch changed is blast radius: one shared walk means
  fastmd5 now inherits fastcp's blind spot.

  **F3 fix (2026-07-31).** An `onerror` callback appending to the same
  `problems` list, so an unlistable directory joins the existing report. Covers
  the root of the walk too. `tests/test_copyplan.py` grew two checks, confirmed
  red first: the subtree case returned `['readable/good.bin']` with the hidden
  file silently absent, and the unlistable-root case returned `[]`. The check
  verifies its own premise and skips if run as a user who can list a 0o000
  directory, rather than passing vacuously.

- [ ] **F4 · `mlperf_common/fileio/datastage.py:649` · no destination-is-a-directory check**

  `parse_args` validates sources but never checks the destination.
  `plan_copy_operations` (copyplan.py:112) silently falls back to a single-file
  copy of `sources[0]` when the destination isn't an existing directory.

  Reproduced: `datastage -r /lustre/dataset /raid/scratch/dataset --dry-run`
  with the destination absent emits one bogus job for the source *directory*
  (`4096 bytes`). In a real run the O_DIRECT open gets EINVAL, falls back to
  buffered, and the reader dies with `[Errno 21] Is a directory` while peers
  block in the collective until the watchdog fires. With multiple sources,
  `sources[1:]` are dropped with no message and the job exits 0.

  `client/fastcp` blocks both cases at `parse_and_validate_args:172-179`.
  Port that check.

- [ ] **F5 · `mlperf_common/fileio/datastage.py:572` · unguarded os.stat before the broadcast**

  The `try` around the payload build catches only `UnreadableEntries`. A file
  unlinked between `plan_copy_operations`'s stat and this one raises
  `FileNotFoundError`; rank 0 exits while every peer is already blocked in
  `broadcast_object_list`. Job dies on a 10+ minute watchdog timeout with an
  opaque collective message instead of `cannot stat <path>`. The comment above
  the call says this must not happen.

## Tier 3 — real, lower severity

- [ ] **F6 · `client/fastmd5:111` · directory-symlink cycles now abort the run** *(priority questioned — see note)*

  Swapping `Path.rglob('*')` (does not descend symlinked dirs) for
  `copyplan.list_relative_files` (`os.walk(followlinks=True)`) means a
  `latest -> .` or `current -> ..` symlink recurses to ELOOP, raises
  `UnreadableEntries`, and `fastmd5:113` turns that into `sys.exit`. Zero
  checksums printed — the tree can't be verified at all. Regression against the
  pre-branch behaviour.

  **Priority questioned (2026-07-31, Matt).** Not convinced this is worth
  fixing: it needs a dataset containing a self- or ancestor-referential
  directory symlink, and unlike the rest of the list it fails loudly — a
  nonzero exit with no output — rather than silently producing wrong results.
  Left open, not scheduled. The proper fix is the depth-first walk with
  visited-inode tracking that the standing `FIXME` in `list_relative_files`
  describes, which would also close the cycle hole `os.walk(followlinks=True)`
  leaves open.

- [ ] **F7 · `mlperf_common/fileio/datastage.py:406` · `_chmod_parents` infinite loop on `/`**

  `datastage a.bin /` makes `dest_root == "/"`; `os.path.dirname("/") == "/"`, so
  the loop never terminates, chmod'ing `/` forever while peers block in the
  following barrier. Hangs to wall-clock limit.

- [ ] **F8 · `mlperf_common/fileio/datastage.py:408` · `_chmod_parents` widens pre-existing dirs**

  Loop condition (`path.startswith(dest_root + os.sep) or path == dest_root`)
  doesn't distinguish directories datastage created from ones already there.
  Verified: a 0750 destination tree becomes 0777. Docstring says the function
  exists to fix the umask on *directories we created*. Under
  `--container-remap-root` the process is root, so it always succeeds silently.

- [ ] **F9 · `mlperf_common/fileio/datastage.py:343` · `.datastage.tmp` files leak**

  Preallocated to full source size via `ftruncate`, removed only by the
  success-path `os.rename`. No `unlink` anywhere in the file. Every failure
  leaves a near-full copy, and the `SLURM_JOB_ID` suffix means each resubmit
  orphans a fresh one until node-local NVMe fills — at which point attempts fail
  with ENOSPC rather than the original error. Tests assert on leftover temp
  files but only exercise the success path.

- [ ] **F10 · `mlperf_common/fileio/datastage.py:537` · all-gather ignores actual round length**

  `FileLayout` gives `shard >= 2 MiB` for any non-empty slice, so `rounds == 1`
  for any file below `buffer_size`, and `send_dev` is always `piece` wide. A
  1 KiB file at N=2048 with default `-b 8M` moves a 16 GiB window into every
  node and D2H-copies all of it (`nbytes = count * layout.piece` is
  unconditional at :474) to keep 1 KiB. For 100k small files this is petabytes
  of fabric traffic plus two world barriers per file — slower than the rsync
  path it replaces.

- [ ] **F11 · `mlperf_common/fileio/datastage.py:277` · memory-budget error suggests an over-budget value**

  Check is `piece * (2N + 1) > budget`; the suggestion divides by `2N`, dropping
  the `+1`. Reproduced at 79.65 GiB / 64 nodes: `-b 512M` rejected with "lower to
  at most 382M", and `-b 382M` reprints byte-identical advice. Correct answer is
  378M. Mis-suggests at every node count checked (16/32/64 on 80/94/141 GB), so
  an operator burns repeated allocations with no workable value ever offered.

- [x] **F12 · `mlperf_common/fileio/datastage.py:475` · no dedicated stream, so no overlap** — fixed, see "F12 fix" below

  The comment at :444 claims round k+1's collective overlaps round k's copy-back.
  It doesn't: the drainer's `copy_` is enqueued on the default stream, same as
  H2D(k+1) and all_gather(k+1), and ProcessGroupNCCL makes its internal stream
  wait on the default stream. `RECV_DEV_SLOTS=2`, the drain thread and the events
  buy only host-side write overlap. Fix is a dedicated `torch.cuda.Stream` for
  the drainer with the existing events as cross-stream waits — which also
  interacts with F1, so do them together.

  **F12 fix (2026-07-31).** `Stager.drain_stream`, with the copies and their
  events issued inside `torch.cuda.stream(self.drain_stream)` and the host-side
  `assembled.synchronize()` replaced by `drain_stream.wait_event(assembled)` —
  the drain thread now queues a round's copies while that round's collective is
  still running, instead of blocking until it finishes.

  `last_copy.synchronize()` before `recv_free_q.put` is now load-bearing for a
  second reason: the next all-gather is issued on the default stream, which has
  no ordering against `drain_stream`, so that host-side wait is the only thing
  keeping the collective off a window still being copied. Commented in place.

  `test_device.py` gained two checks: no drainer operation may be issued on the
  default stream, and a side stream must `wait_event` before its first copy.
  Both confirmed red before the change; the second was re-confirmed red by
  deleting the `wait_event` line and re-running.

  **Unmeasured.** The expected win is per-round `T_collective + T_d2h` becoming
  roughly `max(...)`, but whether the GPU side is the bottleneck at all depends
  on Lustre read and NVMe write rates. The stubs make every copy instantaneous,
  so the suite cannot see throughput. Profile on a real node before claiming a
  speedup.

- [ ] **F13 · `client/direct_io.py:35` · shim fallback can't displace a stale install**

  When an older `mlperf_common` is already importable (a container that
  pip-installed one from a benchmark's `requirements.txt`), the first import
  binds `mlperf_common` in `sys.modules` to site-packages. Inserting the repo
  root at `sys.path[0]` doesn't change the cached package's `__path__`, so the
  retry re-resolves to the same place and the error tells the user to do the one
  thing they have already done. The compatibility path the shim exists for is
  dead in exactly the case it was written for.

## Tier 4 — the tests can't catch the above

- [ ] **F14 · `tests/test_pipeline.py:90` · fake all-gather re-reads the source**

  `fake_all_gather` ignores its `inp` argument and fills the window from
  `os.pread(STATE['fd'], ...)`. Verified: making `_read_piece` overwrite
  everything it reads with `0xEE` still passes all 8 multi-node configurations
  (only the 7 single-node cases go red). So `_read_piece`'s offset math, its
  threaded split, send-slot recycling and the H2D copy are unverified in every
  multi-node case. Instrumentation shows the threaded branch is entered exactly
  once in the whole suite, in a case whose result is discarded.

- [ ] **F15 · `tests/stubs.py:106` · stubs can't construct `Topology` or `build_plan`**

  `install()` defines only barrier/get_rank/broadcast_object_list/all_reduce/
  all_gather_into_tensor. `ds.Topology` raises AttributeError on
  `all_gather_object`; `ds.build_plan` raises on `torch.tensor`. Both test files
  therefore hand-roll their own `Topology` (test_layout.py:47,
  test_pipeline.py:95) and never touch the real one. Reintroduce the node
  grouping bug and the suite still prints "all tests passed".

  Note `tests/README.md` tells you to validate pipeline changes by reintroducing
  a bug and confirming red. Per F14 that instruction does not currently hold for
  the multi-node path.

---

## Suggested order

1. **F1** — fires unconditionally, corrupts data, blocks trusting anything else.
2. **F14 + F15** — without these, an F1 fix can't be shown to work.
3. **F3, F4, F5** — silent loss and hangs; all small, independent fixes.
4. **F12** with F1 (same code, same stream reasoning).
5. Remainder in any order. **F2** likely reduces to an assert.

## Provenance

* `.review-findings.json` — verifier output, this directory (durable copy).
* Workflow run `wf_46ab6788-524`, session `5451bab2-8557-4276-b6fb-39cb3735c04e`.
  Per-agent transcripts under
  `~/.claude-nvidia-account/projects/-home-matt-work-jun-2026-mlperf-common/<session>/subagents/workflows/<run>/`.
  Session-scoped; will not survive into a new session.
* The original `/tmp/claude-1000/.../tasks/w0ajzd4ot.output` is volatile.
