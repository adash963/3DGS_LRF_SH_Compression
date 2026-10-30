# `tests/conftest.py`

## Purpose

Pytest's shared fixture file — automatically discovered, no import needed. It builds
synthetic 3DGS models so the entire test suite runs **without the 265 MB checkpoint**,
which is what allows all 50 tests to execute in under a second on any machine.

## Line-by-line

### Lines 1–11 — Imports and path setup

```python
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sh_compression import sh_layout
from sh_compression.ply_io import GaussianPly, write_gaussian_ply
```

The same `sys.path` insertion the scripts use, so tests run without installing the
package. Because `conftest.py` is imported first by pytest, this path fix applies to
every test module in the suite.

### Lines 13–25 — Field definitions

```python
NON_SH_FIELDS = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)

# Field order matches scene/gaussian_model.py construct_list_of_attributes()
FIELD_ORDER = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + sh_layout.DC_FIELDS
    + sh_layout.REST_FIELDS
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)
```

**`FIELD_ORDER` deliberately mirrors the upstream ordering**, as the comment records:
position, normals, DC colour, rest colour, opacity, scale, rotation — 62 fields
total. The synthetic models are therefore structurally identical to real ones, not
merely similar.

That matters because the tests verify field preservation and ordering. A fixture with
an arbitrary field order could pass tests that a real model would fail.

Building the SH names from `sh_layout` rather than hardcoding them means the fixture
automatically follows any layout change.

### Lines 28–31 — `make_tiny_model`

```python
def make_tiny_model(n: int = 100, seed: int = 42) -> GaussianPly:
    rng = np.random.default_rng(seed)
    fields = {name: rng.normal(size=n).astype(np.float32) for name in FIELD_ORDER}
    return GaussianPly(field_names=list(FIELD_ORDER), fields=fields)
```

Generates a model with random float32 values in every column.

The values are physically meaningless — random normals are not valid quaternions or
plausible opacities — and that is fine, because none of the code under test
*interprets* these fields. `ply_io` treats non-SH fields as opaque data, so random
values exercise the code paths exactly as real values would, while making it obvious
that no test accidentally depends on plausible geometry.

`default_rng(seed)` fixes the values, so a test failure is reproducible.

### Lines 34–36, 39–44 — Fixtures

```python
@pytest.fixture
def tiny_model():
    return make_tiny_model()


@pytest.fixture
def tiny_ply(tmp_path, tiny_model):
    path = str(tmp_path / "tiny.ply")
    write_gaussian_ply(tiny_model, path)
    return path
```

`tiny_model` provides an in-memory model; `tiny_ply` writes one to disk and returns
the path, for tests that need real file I/O.

`tmp_path` is pytest's built-in per-test temporary directory fixture. Each test gets
a fresh one, automatically cleaned up — so tests cannot interfere with each other
through leftover files, and nothing is written into the repository.

`tiny_ply` depending on `tiny_model` is fixture composition: pytest resolves the
chain automatically, and both are rebuilt per test rather than shared, so a test that
mutates a model cannot affect another.

`make_tiny_model` is exposed as a plain function as well as a fixture because the
integration test imports it directly to build a variant with structured (rather than
random) SH.

## Why it matters

The test suite's speed and portability come from this file. All 50 tests run in
0.7 seconds with no external data, which means:

- they run on every commit without friction,
- they run in Colab as a smoke check before any GPU work (they did, in every session:
  "46 passed"),
- a reviewer can clone the repository and verify the core mathematics immediately,
  without downloading a checkpoint.

The deliberate structural fidelity — real field names, real ordering, real dtypes —
is what makes results on synthetic models transferable to real ones.
