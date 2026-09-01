# tests

Stdlib only, no pytest, no numpy, no GPU. Run them with:

    python3 tests/run_tests.py

or run any one directly:

    python3 tests/test_copyplan.py

Each test runs in its own interpreter, so nothing one test leaves in
`sys.modules` can reach another.

## What is here

| file | covers |
| --- | --- |
| `test_copyplan.py` | tree walk, src→dst mapping, `cp` argument semantics, and the refusal to plan an unreadable tree |

`mlperf_common.fileio.copyplan` is stdlib-only, so the test imports the real
module straight out of the checkout. There is nothing to stub.

## What these do NOT cover

`direct_io` has no test here. Its `pread`/`pwrite` retry loops and block-size
padding are exercised only indirectly, by running `fastcp` and `fastmd5` against
a real filesystem — and O_DIRECT's alignment demands are not reproducible
against a temp directory on every filesystem. That is a real gap, not a
deliberate omission.

Nothing here covers throughput, Lustre, or node-local NVMe.

## Why `copyplan` is worth testing at all

`fastcp` and `fastmd5` both enumerate a source tree through `list_relative_files`,
so the two agree about which files exist only because they share this code. If
the walk silently skipped a subtree — which is `os.walk`'s default when it
cannot list a directory — a partial copy would exit 0 and then be "verified" by
a checksum run that skipped exactly the same files. `plan_copy_operations`
raises `UnreadableEntries` listing every bad entry instead, and
`check_unlistable_directory` is what holds that behaviour in place.
