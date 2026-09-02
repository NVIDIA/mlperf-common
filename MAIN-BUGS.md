# fastcp / fastmd5 / direct_io — defects present on `main`

**Status: not filed anywhere.** Recorded here because `NVIDIA/mlperf-common` has
GitHub issues disabled (`hasIssuesEnabled: false`), so there is no tracker to
file into. This file is the record until there is one; move it wherever the team
actually tracks bugs, and delete it here when that happens.

- Found: 2026-09-02, reviewing PR #43 (`mfrank/shared-fileio`).
- Verified against: `main` @ `b5014f3`, via `git show main:client/fastcp` run directly.
- **None of these is caused by PR #43.** Each reproduces identically before and
  after that refactor; `client/direct_io.py` is byte-identical across the two
  modulo comments. The three defects that PR *did* introduce are tracked in
  `FASTCP-BUGS.md` on that branch and are not repeated here.
- Deliberately excluded: anything only reachable on the `mfrank/py-data-staging`
  branch.

To re-run any of these against main:

```bash
cd /home/matt/work/jun-2026/mlperf-common
mkdir -p /tmp/mv && git show main:client/fastcp > /tmp/mv/fastcp
git show main:client/direct_io.py > /tmp/mv/direct_io.py
python3 /tmp/mv/fastcp ...
```

Ordered by severity.

---

# DATA LOSS

## 1. `fastcp -f SRC .` truncates the source to zero

```bash
mkdir -p /tmp/t1 && cd /tmp/t1 && head -c 50000 /dev/urandom > a.bin
python3 /tmp/mv/fastcp -f a.bin .
ls -l a.bin        # 0 bytes.  exit was 0.
```

DEST resolves to `/tmp/t1/a.bin`; `fastcp` opens it `O_TRUNC` *before* fstat'ing
the source, so the size it then copies is 0. GNU cp refuses outright:
`'a.bin' and 'a.bin' are the same file`, exit 1.

`validate_copy_args` has no same-file guard. Without `-f` the pre-existence
check saves you; with `-f` — which any re-runnable staging script needs — the
source is unrecoverable.

**Fix:** compare `(st_dev, st_ino)` of source and resolved destination in
`validate_copy_args`, and refuse.

## 2. `fastcp -r -f src/.. dst` destroys files outside the copy

```bash
mkdir -p /tmp/t2/src /tmp/t2/dst && cd /tmp/t2
echo SRC-DATA > src/a.txt && echo VICTIM-DATA > victim.txt
python3 /tmp/mv/fastcp -r -f src/.. dst
wc -c victim.txt src/a.txt   # both 0.  dst/ is empty.  exit was 0.
```

Destinations are built from `os.path.basename(src.rstrip("/"))`, which returns
`".."`, so every planned destination is literally `dst/../<name>` — i.e. back
outside DEST, onto the sources themselves. GNU cp refuses:
`cannot copy a directory, 'src/..', into itself`.

`basename('/') == ''` collapses `/` into DEST the same way. Without `-f` the
same command writes files outside DEST rather than destroying them.

**Fix:** normalise and reject any source whose basename is `.`, `..` or empty;
or refuse when the resolved destination is not strictly under DEST.

---

# SILENT WRONG RESULTS

## 3. `-n 0` writes a full-size file of zeros and exits 0

```bash
mkdir -p /tmp/t3 && cd /tmp/t3 && head -c 50000 /dev/urandom > a.bin
python3 /tmp/mv/fastcp -n 0 a.bin out.bin
ls -l out.bin                                    # 50000 bytes
python3 -c "d=open('out.bin','rb').read(); print(d == b'\x00'*len(d))"   # True
```

`--num-threads` is never validated. `range(0)` makes both the thread-start and
the join loops no-ops, then `os.ftruncate(fd_dst, file_size)` inflates the empty
destination to the source's size.

The verification half fails the same way: `fastmd5 -n 0 <tree>` emits **zero
lines** and exits 0, so diffing a source checksum run against a destination one
reports them **identical**. The two failures compound into a clean-looking
staging job that copied nothing.

Reachable without anyone typing `0`: `-n $(nproc)` inside a constrained cgroup,
or an unset shell variable arithmetic-expanded to 0.

Same missing guard on `-b`: `fastmd5 -b 0` dies with `ZeroDivisionError` instead
of its intended message.

**Fix:** require `--num-threads >= 1` and `--buffer-size > 0` in both tools.

## 4. A dead worker thread yields a same-size, wrong-bytes destination

An exception inside `copy_worker` kills only that thread. `join()` returns
normally, `ftruncate` sets the correct size, and `fastcp` exits 0. Source and
destination sizes match; contents do not.

`fastmd5`'s `checksum_worker` docstring states the rule verbatim — *"an exception
escaping a thread does not affect the process exit status"* — and `fastmd5` was
fixed for exactly this. `fastcp`, the tool that *writes* data, was not.

Related: `fastcp`'s bare `except Exception` around `workpile.get_nowait()`
swallows non-`Empty` errors, where `fastmd5` correctly narrows to `queue.Empty`.

**Fix:** port `fastmd5`'s pattern — collect worker exceptions into a shared list
and exit nonzero if it is non-empty.

## 5. Empty directories are silently omitted

```bash
mkdir -p /tmp/t5/src/hollow /tmp/t5/dst && echo x > /tmp/t5/src/plain
python3 /tmp/mv/fastcp -r /tmp/t5/src /tmp/t5/dst
ls /tmp/t5/dst/src           # 'plain' only; 'hollow' is missing.  exit was 0.
```

`cp -r` recreates it. Fixed on the PR #43 branch (commit `01f4634`) by planning
directories alongside files; `main` is still affected.

**Correcting `01f4634`'s commit message,** which calls this "a regression the
fileio extraction introduced". That is only half true, and the halves differ:

| case | `main` | before `01f4634` | after |
| --- | --- | --- | --- |
| DEST exists, source has an empty subdir | exit 0, subdir missing | exit 0, subdir missing | exit 0, created |
| DEST absent, source wholly empty | **exit 1**, loud | exit 0, DEST never created | exit 0, created |

So the first row is a defect `main` already had and the extraction carried
across unchanged — it belongs in this file, which is why it is here. Only the
second row regressed, and it regressed in the way that matters most: `main`
failed loudly enough for a caller checking the exit status to notice, and the
refactor turned that into a silent success with an incomplete tree.

The fix is correct for both rows. Only the commit message overstated its scope,
and it is not worth a force-push to reword.

## 6. Duplicate destination names clobber without `--force`

`fastcp x/same y/same dst` writes both to `dst/same`, last writer wins, exit 0.
GNU cp refuses with `will not overwrite just-created`.

---

# HANGS

## 7. A FIFO anywhere in the tree hangs the copy forever

```bash
mkdir -p /tmp/t7/src /tmp/t7/dst && echo hi > /tmp/t7/src/a.txt
mkfifo /tmp/t7/src/pipe
timeout 10 python3 /tmp/mv/fastcp -r /tmp/t7/src /tmp/t7/dst   # exit 124
```

Blocks in `open(O_RDONLY)` with no diagnostic and no timeout. `cp -r` completes
in milliseconds and recreates it as a FIFO. `fastmd5` silently skips it, so the
checksum reports a clean tree the copier can never finish. In a multi-rank
staging job this is a hang on one rank while the others wait.

**Fix:** `list_relative_files` should classify non-regular files and either skip
them with a warning or refuse the copy.

## 8. Unbounded retry loops in `direct_io`

`pread` and `pwrite` re-issue the identical syscall at the identical offset on a
short transfer, with no iteration bound and no progress check. A *permanent*
short read/write therefore spins at 100% CPU forever instead of raising.

- Read side: a file that shrinks between `fastmd5`'s `os.path.getsize` at enqueue
  and the worker's `pread` minutes later hangs that worker; `main()`'s `join()`
  never returns.
- Write side: this is the ENOSPC shape. A destination filling up makes `fastcp`
  spin forever rather than report `No space left on device`.

Same area, two more:

- The retry guards compare against `count` where they need `padded_count`, so a
  legitimate partial transfer raises an assertion describing the opposite of
  what happened, instead of reaching the retry that would have worked.
- Under `python -O` every `assert` in the module vanishes. They are the module's
  **only** validation, and it has no test coverage at all — including
  `allocate_aligned_buffers`' alignment check, whose failure leaves buffers
  misaligned and every O_DIRECT call returning EINVAL.

**Fix:** bound the retries, require forward progress between iterations, and
convert the asserts to real exceptions.

---

# PERMISSIONS

## 9. Destination mode is never preserved

```bash
umask 002
mkdir -p /tmp/t9/src /tmp/t9/dst && head -c 100 /dev/urandom > /tmp/t9/src/secret
chmod 600 /tmp/t9/src/secret
python3 /tmp/mv/fastcp -r /tmp/t9/src /tmp/t9/dst
stat -c '%a' /tmp/t9/dst/src/secret     # 775, was 600
```

`os.open` is called with no mode argument, so destinations are created
`0o777 & ~umask`. `os.makedirs` does the same for directories — a `0o700` source
subdirectory becomes `0o775`. GNU cp preserves the mode in both cases.

Staging a restricted-license dataset onto shared node-local storage therefore
widens its permissions to every other user on the node. Worth treating as a
security issue, not a cosmetic one.

---

# COSMETIC

## 10. `fastmd5`'s tab-separated output is unescaped

A filename containing a tab yields five fields instead of four, so any consumer
splitting on tabs mis-parses the line. An undecodable filename additionally makes
the error handler re-raise `UnicodeEncodeError` from its own diagnostic print.
