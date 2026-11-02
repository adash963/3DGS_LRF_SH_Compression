# `src/sh_compression/__init__.py`

## Purpose

The package marker. Five lines, but one of them defines a constant that every saved
asset depends on.

## Line-by-line

```python
"""Scene-level low-rank + vector-quantised compression of 3DGS SH coefficients."""

__version__ = "0.1.0"

FORMAT_VERSION = 1
```

### Line 1 — Docstring

A one-line statement of what the package does, visible via `help(sh_compression)`.

### Line 3 — `__version__`

Standard Python package version. Currently unused programmatically, but present so
the package follows convention and can be released or pinned later.

### Line 5 — `FORMAT_VERSION`

**The significant line.** This integer is stamped into the metadata of every `.npz`
asset written by `serialization.py`, and checked on every load:

```python
if meta.get("format_version") != FORMAT_VERSION:
    raise ValueError(f"unsupported format_version ...")
```

Why it deserves to exist in a project of this size: if the storage layout ever
changed — say `U` began storing singular values separately rather than folded in —
old assets would still load structurally (the arrays are all still there, all the
right shapes) but reconstruct to *wrong values*. Silently. A version check converts
that silent corruption into a loud, immediate failure.

It also sits deliberately in `__init__.py` rather than inside `serialization.py`. As
a package-level constant it is a declaration about the package as a whole, and both
the writer and any future migration tool reference the same single definition.

`test_serialization.py::test_unsupported_version_rejected` verifies the mechanism by
editing a saved file's metadata to version 999 and confirming that loading raises.

## Why it matters

The deliberately empty rest of this file is itself a design choice. Nothing is
re-exported here — callers import from the specific module they need
(`from sh_compression.factorization import svd_factorize`). That keeps import costs
low and makes dependencies explicit at every call site, so reading any file tells you
exactly what it relies on.
