# `tests/unit/test_ply_roundtrip.py`

## Purpose

Verifies PLY reading, writing, extraction and reinsertion — including the property
that gives the project its experimental control: **non-SH fields survive untouched.**

## Line-by-line

### Lines 1–7 — Imports

```python
import numpy as np
import pytest

from sh_compression import sh_layout as L
from sh_compression.extraction import extract_flat_sh, roundtrip_unchanged
from sh_compression.ply_io import extract_sh, insert_sh, read_gaussian_ply
```

### Lines 10–15 — `test_read_write_roundtrip`

```python
def test_read_write_roundtrip(tiny_model, tiny_ply):
    reread = read_gaussian_ply(tiny_ply)
    assert reread.field_names == tiny_model.field_names
    for f in tiny_model.field_names:
        assert np.array_equal(reread.fields[f], tiny_model.fields[f])
```

Writes a model, reads it back, compares everything.

**Line 12 checks field *order*, not just membership.** Column order affects the
file's byte layout, and preserving it is what keeps written files structurally
identical to their sources. Comparing lists (rather than sets) is what makes this an
ordering assertion.

Lines 13–14 compare every column exactly. Exact equality is achievable because the
fixture data is already float32, so writing float32 involves no conversion.

### Lines 18–23 — `test_extract_matches_ply_fields`

```python
def test_extract_matches_ply_fields(tiny_model):
    sh = extract_sh(tiny_model)
    for c in range(3):
        assert np.array_equal(sh[:, c, 0], tiny_model.fields[f"f_dc_{c}"])
        for j in range(15):
            assert np.array_equal(sh[:, c, 1 + j], tiny_model.fields[f"f_rest_{c*15+j}"])
```

Verifies the layout mapping again, but at the `ply_io` level rather than the pure
`sh_layout` level — this exercises the `np.stack` calls in `extract_sh`, which are a
separate opportunity to introduce a permutation.

All 48 positions are checked individually against their named source column.

### Lines 26–31 — `test_insert_preserves_non_sh`

```python
def test_insert_preserves_non_sh(tiny_model):
    sh = extract_sh(tiny_model)
    out = insert_sh(tiny_model, sh * 2.0)
    for f in tiny_model.non_sh_field_names():
        assert np.array_equal(out.fields[f], tiny_model.fields[f])
    assert np.allclose(extract_sh(out), sh * 2.0)
```

**The experimental-control test.** SH is doubled — an unmistakable change — and the
assertions confirm two complementary things:

1. Every non-SH field is *identical* (line 30). Geometry did not move.
2. The SH fields *did* change to the new values (line 31).

Both directions matter. A function that changed nothing would pass the first check
alone; this test proves the change was surgical.

This property is what licenses every comparison in the dissertation to be described
as "same geometry, different colour encoding".

### Lines 34–37 — `test_full_roundtrip_via_disk`

```python
def test_full_roundtrip_via_disk(tiny_ply, tmp_path):
    report = roundtrip_unchanged(tiny_ply, str(tmp_path / "rt.ply"))
    assert report["max_abs_coeff_diff"] == 0.0
    assert report["non_sh_fields_identical"]
```

Exercises the exact function used for the Stage 2 verification, on a synthetic model.

`== 0.0` is an **exact** assertion, not a tolerance. That is deliberate: the round
trip involves no arithmetic on the coefficients, only movement between
representations, so any non-zero difference indicates a real bug rather than
floating-point noise.

This test is why the real-model result of `0.0` was expected rather than surprising —
the property was already established on synthetic data.

### Lines 40–43 — `test_extract_flat_shape`

```python
def test_extract_flat_shape(tiny_ply):
    _, flat = extract_flat_sh(tiny_ply)
    assert flat.shape == (100, L.FLAT_DIM) and flat.dtype == np.float32
```

Confirms the pipeline entry point returns the expected shape and dtype — the contract
every downstream module relies on.

### Lines 46–56 — `test_wrong_degree_rejected`

```python
def test_wrong_degree_rejected(tmp_path, tiny_model):
    from sh_compression.ply_io import GaussianPly, write_gaussian_ply

    names = [f for f in tiny_model.field_names if f != "f_rest_44"]
    broken = GaussianPly(names, {k: tiny_model.fields[k] for k in names})
    p = str(tmp_path / "broken.ply")
    write_gaussian_ply(broken, p)
    with pytest.raises(ValueError, match="SH degree"):
        read_gaussian_ply(p)
```

Constructs a deliberately malformed model by **removing one SH column**, writes it,
and confirms the reader rejects it.

This simulates a lower-degree model. The `match="SH degree"` assertion checks the
error explains the actual cause, so a user encountering it learns something
actionable rather than seeing a bare count mismatch.

### Lines 59–63 — `test_nan_model_rejected`

```python
def test_nan_model_rejected(tiny_model):
    tiny_model.fields["f_dc_0"] = tiny_model.fields["f_dc_0"].copy()
    tiny_model.fields["f_dc_0"][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        extract_sh(tiny_model)
```

Injects a NaN into one coefficient of one Gaussian and confirms extraction refuses it.

**Line 60's `.copy()` is not incidental.** Without it the test would mutate the array
in place; although pytest rebuilds the fixture per test, copying makes the test's
independence explicit rather than dependent on fixture scoping.

Catching NaN here matters because a single NaN coefficient would propagate through
the SVD to contaminate *every* singular vector, turning a one-Gaussian problem into a
whole-scene failure with no obvious origin.

## Why it matters

These eight tests establish the two properties the dissertation's Stage 2 result
depends on:

- **Losslessness** — extraction and reinsertion introduce zero coefficient error,
  through memory *and* through disk.
- **Isolation** — SH changes never disturb geometry.

Because both are proven on synthetic data first, the identical result on the
1,071,462-Gaussian real model confirms behaviour that was already specified, rather
than being a fortunate observation.
