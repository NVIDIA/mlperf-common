# Known defects in fastcp / fastmd5 / fileio

Found reviewing the `mlperf_common.fileio` extraction (PR #43).

Everything here was **introduced by that extraction** — each item was checked
against `main`'s pre-refactor `client/fastcp` and does not occur there. The much
longer list of defects this code inherited from `main` is deliberately *not*
here: those predate the refactor and are not this branch's to fix. They are
recorded in `MAIN-BUGS.md`.

`direct_io.py` is byte-identical to `main`'s modulo comments, so nothing in it
is a new defect.

---

## 1. The "cannot import mlperf_common" advisory is unreachable

`client/fastcp:35-42` and `client/fastmd5:20-27` wrap the `copyplan` import in a
`try`/`except ImportError` that exits with a one-line explanation naming both
remedies. It never runs: the *unguarded* `import direct_io` at `fastcp:32` /
`fastmd5:18` executes first and fails first.

Reproduced — copy `client/fastcp` and `client/direct_io.py` somewhere with no
`mlperf_common` and run it:

```
Traceback (most recent call last):
  File "./fastcp", line 32, in <module>
    import direct_io
  ...
ImportError: No module named 'mlperf_common'. direct_io lives in the
mlperf_common package. ...
```

A chained three-frame traceback, in the exact scenario the friendly message was
written for. The shim's own text is reasonable, so this is cosmetic — but the
handler above it is dead code and reads as though it works.

Fix: move `import direct_io` inside the same guarded block, or drop the
now-redundant handler.

## 2. A stale `direct_io.py` beside the script silently shadows the package

`sys.path.insert(0, _HERE)` puts the script's own directory ahead of everything,
so a leftover standalone `direct_io.py` in an install's `bin/` outranks the
shim. Reproduced with `main`'s pre-refactor copy dropped next to `fastcp`:

```
direct_io resolved to : .../bin/direct_io.py
same module object    : False
same pread function   : False
```

The copy then runs the *stale* I/O primitives while `copyplan` comes from the
package. Harmless today only because the two are byte-identical — which means
the first edit to `mlperf_common/fileio/direct_io.py` silently stops reaching
any install carrying a leftover `bin/direct_io.py`, with no warning.

This is a live risk precisely because `setup.py` *does* install
`client/direct_io.py` into `bin/`, so upgrading from a pre-refactor install
leaves the old file there.

Related, not reproduced: `os.path.abspath` does not resolve symlinks, so
`ln -s <repo>/client/fastcp ~/bin/fastcp` makes `_HERE` `~/bin` and both inserts
point at the wrong tree. `os.path.realpath` would fix both.

## 3. `test_copyplan.py`'s refusal rows do not check *which* error was raised

`tests/test_copyplan.py:106-110` asserts only that some `CopyArgumentError`
escaped:

```python
try:
    copyplan.validate_copy_args(sources, destination, **kwargs)
    check(name, False, "no exception raised")
except copyplan.CopyArgumentError:
    check(name, True)
```

A misspelled fixture path, or a validator that grew over-eager and rejects the
wrong thing for the wrong reason, leaves every one of these rows green. Match
the message, or at least a distinct exception subclass per rule.
