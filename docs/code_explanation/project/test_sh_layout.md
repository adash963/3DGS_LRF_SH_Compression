# `tests/unit/test_sh_layout.py`

## Purpose

Proves that **no channel or coefficient permutation occurs** anywhere in the layout
code. These six tests are the foundation of the dissertation's claim to be
manipulating the correct coefficients — a claim that cannot be verified by inspecting
renders, because a permutation bug produces plausible-looking but wrong output.

## Line-by-line

### Lines 1–5 — Imports

```python
import numpy as np
import pytest

from sh_compression import sh_layout as L
```

Aliased to `L` for brevity across many references.

### Lines 8–9 — `test_constants`

```python
def test_constants():
    assert L.N_COEFFS == 16 and L.FLAT_DIM == 48 and len(L.REST_FIELDS) == 45
```

Pins the layout arithmetic. Trivial, but it catches an accidental change to
`SH_DEGREE` — which would silently alter every derived constant and invalidate the
entire pipeline's assumptions.

### Lines 12–16 — `test_flatten_unflatten_lossless`

```python
def test_flatten_unflatten_lossless():
    rng = np.random.default_rng(0)
    sh = rng.normal(size=(50, 3, 16)).astype(np.float32)
    assert np.array_equal(L.unflatten(L.flatten(sh)), sh)
```

Round-trip through the flat form must return the input **exactly** —
`np.array_equal`, not `allclose`. Reshaping is a pure reinterpretation of memory with
no arithmetic, so exact equality is the correct assertion; accepting approximate
equality here would mask a genuine bug.

### Lines 19–24 — `test_flat_column_mapping`

```python
def test_flat_column_mapping():
    """Coefficient j of channel c must land at flat column c*16+j."""
    sh = np.zeros((1, 3, 16), dtype=np.float32)
    sh[0, 1, 5] = 7.0
    flat = L.flatten(sh)
    assert flat[0, 1 * 16 + 5] == 7.0 and np.count_nonzero(flat) == 1
```

A **single-value probe**. One element is set — channel 1, coefficient 5 — and the
test asserts two things:

1. the value appears at column `1*16+5 = 21`, and
2. **exactly one** element is non-zero.

The second assertion is what makes this rigorous. Checking only the first would pass
even if the value had been duplicated elsewhere; `count_nonzero == 1` proves the
mapping is a bijection at this position, with no leakage.

### Lines 27–37 — `test_ply_field_mapping_no_permutation`

```python
def test_ply_field_mapping_no_permutation():
    """f_dc_c -> [:,c,0]; f_rest_{c*15+j} -> [:,c,1+j]."""
    n = 4
    dc = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    rest = 100 + np.arange(n * 45, dtype=np.float32).reshape(n, 45)
    sh = L.from_ply_fields(dc, rest)
    assert np.array_equal(sh[:, :, 0], dc)
    for c in range(3):
        for j in range(15):
            assert np.array_equal(sh[:, c, 1 + j], rest[:, c * 15 + j])
    dc2, rest2 = L.to_ply_fields(sh)
    assert np.array_equal(dc2, dc) and np.array_equal(rest2, rest)
```

**The most important test in the file** — it verifies the channel-major claim that
`from_ply_fields`' central reshape depends on.

Using `np.arange` makes every value distinct, so any misplacement is detectable —
unlike random data, where a coincidental match is conceivable, or constant data,
where permutation is invisible. Offsetting `rest` by 100 keeps the two ranges
non-overlapping, so a DC/rest confusion would also be caught.

Lines 34–36 check **every one of the 45 rest positions individually** rather than
asserting on the array as a whole. A blanket comparison would pass if two columns
were swapped in a way that preserved the aggregate; per-position assertions cannot.

Lines 37–38 then verify the inverse: converting back reproduces the original inputs
exactly. Together these establish that `from_ply_fields` and `to_ply_fields` are
exact mutual inverses — one of the two links in the bit-exact round-trip chain.

### Lines 40–47 — `test_bad_shapes_rejected`

```python
def test_bad_shapes_rejected():
    with pytest.raises(ValueError):
        L.flatten(np.zeros((5, 3, 15)))
    with pytest.raises(ValueError):
        L.unflatten(np.zeros((5, 47)))
    with pytest.raises(ValueError):
        L.from_ply_fields(np.zeros((5, 3)), np.zeros((5, 44)))
```

Three near-miss shapes: 15 coefficients instead of 16, 47 columns instead of 48, 44
rest fields instead of 45. Each is off by exactly one — the kind of mistake a
degree-2 model or an off-by-one slice would produce, and precisely the case where
silent acceptance would be most damaging.

### Lines 50–55 — `test_nan_detection`

```python
def test_nan_detection():
    sh = np.zeros((2, 3, 16))
    sh[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        L.validate_finite(sh)
```

A single NaN among 96 values must be caught. The `match="non-finite"` argument
asserts on the error *message*, not just the exception type — so a different
`ValueError` raised for an unrelated reason would fail the test.

## Why it matters

A permutation bug in SH handling is the worst class of error this project could have:
everything runs, all shapes are valid, files save and load, renders are produced —
and every result is wrong in a way that no amount of staring at output would reveal.

These tests eliminate that possibility by construction, using distinct known values
at known positions. They are the reason the dissertation can assert "we are provably
compressing the correct coefficients" rather than merely assuming it.
