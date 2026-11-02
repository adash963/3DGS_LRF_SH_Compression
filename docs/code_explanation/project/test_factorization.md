# `tests/unit/test_factorization.py`

## Purpose

Proves the SVD mathematics on data whose true rank is **known in advance**. This is
the Stage 3 gate: the brief required the mathematics to be validated on synthetic
data before any conclusion was drawn from the real scene.

## Line-by-line

### Lines 1–12 — Setup

```python
import numpy as np
import pytest

from sh_compression.factorization import (
    analyse_ranks, effective_rank, make_synthetic, reconstruct, svd_factorize,
)

RANKS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]


def rel_err(x, u, v):
    return np.linalg.norm(x - reconstruct(u, v)) / np.linalg.norm(x)
```

`rel_err` is the shared measure: reconstruction error as a fraction of signal
magnitude. Scale-free, so thresholds mean the same thing regardless of the data's
absolute size.

### Lines 15–19 — `test_exact_rank_recovers`

```python
@pytest.mark.parametrize("true_rank", [1, 2, 4])
def test_exact_rank_recovers(true_rank):
    x = make_synthetic(200, rank=true_rank, seed=1)
    u, v = svd_factorize(x, true_rank)
    assert rel_err(x, u, v) < 1e-6
```

**The foundational test.** A matrix built as a product of `[200, r]` and `[r, 48]`
factors cannot have rank above `r`. Factorising at exactly `r` must therefore recover
it essentially perfectly.

The `1e-6` threshold accommodates float32 output from a float64 computation. This is
a genuine test of correctness rather than a check that code runs: if `svd_factorize`
mishandled the singular-value folding, or truncated the wrong components, the error
would be orders of magnitude larger.

`parametrize` runs it three times, so a bug affecting only rank-1 (a plausible
edge case, since rank-1 involves single-column slices) is caught.

### Lines 22–27 — `test_rank3_worse_than_rank4_on_rank4_data`

```python
def test_rank3_worse_than_rank4_on_rank4_data():
    x = make_synthetic(200, rank=4, seed=2)
    e3 = rel_err(x, *svd_factorize(x, 3))
    e4 = rel_err(x, *svd_factorize(x, 4))
    assert e4 < 1e-6 < e3
```

Directly implements a brief requirement: "rank 3 has higher error than rank 4 for
exact rank-4 data."

The chained comparison `e4 < 1e-6 < e3` asserts two things at once — rank 4 is
essentially exact, and rank 3 is *not*. The second half is what makes the test
meaningful: a degenerate implementation returning near-zero error for every rank
would pass the first condition but fail here.

### Lines 30–35 — `test_error_monotone_energy_monotone`

```python
def test_error_monotone_energy_monotone():
    x = make_synthetic(300, rank=4, noise=0.1, seed=3)
    res = analyse_ranks(x, RANKS)
    errs, energy = res.rel_frobenius_error, res.explained_energy
    assert all(errs[i + 1] <= errs[i] + 1e-12 for i in range(len(errs) - 1))
    assert all(energy[i + 1] >= energy[i] - 1e-12 for i in range(len(energy) - 1))
```

Two mathematical invariants of truncated SVD: adding a component can never increase
error, and can never decrease captured energy.

The `± 1e-12` tolerances permit floating-point ties without allowing genuine
violations. Noise is added (line 31) so the spectrum decays gradually rather than
truncating sharply — exercising the monotonicity check across many ranks rather than
just around a cliff.

Violation here would indicate a serious bug, such as components being taken out of
singular-value order.

### Lines 38–41 — `test_full_rank_reconstruction_exact`

```python
def test_full_rank_reconstruction_exact():
    x = make_synthetic(100, seed=4)  # full-rank random
    u, v = svd_factorize(x, 48)
    assert rel_err(x, u, v) < 1e-5
```

Full-rank factorisation of a genuinely full-rank random matrix must reproduce the
input. This is the synthetic counterpart of the real-model result where `lr_r48_fp32`
rendered at 101 dB.

The looser `1e-5` threshold reflects accumulated float32 error across all 48
components. Note this is *numerical* equivalence, not exactness — which is precisely
why the report describes the full-rank configuration as "numerically equivalent"
rather than "bit-exact", a distinction the static audit was right to insist on.

### Lines 44–48 — `test_constant_data_is_rank_one`

```python
def test_constant_data_is_rank_one():
    x = make_synthetic(150, constant=True, seed=5)
    u, v = svd_factorize(x, 1)
    assert rel_err(x, u, v) < 1e-6
    assert effective_rank(analyse_ranks(x, [1]).singular_values) == 1
```

A degenerate case: every row identical, so the matrix is exactly rank 1. Both the
factorisation and `effective_rank` must recognise this.

Line 48 is the only direct test of `effective_rank` — the function that produced the
dissertation's "12 of 48" result — on data with an unambiguous answer.

### Lines 51–54 — `test_ranks_above_max_skipped`

```python
def test_ranks_above_max_skipped():
    x = make_synthetic(10, seed=6)  # max rank 10
    res = analyse_ranks(x, RANKS)
    assert max(res.ranks) <= 10
```

With only 10 rows, `min(N, 48) = 10`, so requesting rank 48 is meaningless. The sweep
must silently drop out-of-range ranks rather than crash — which is what allows one
standard rank list to be reused across matrices of any size.

### Lines 57–67 — `test_invalid_inputs_rejected`

```python
def test_invalid_inputs_rejected():
    x = make_synthetic(50, seed=7)
    for bad in (0, -1, 49, 2.5):
        with pytest.raises(ValueError):
            svd_factorize(x, bad)
    bad_x = x.copy()
    bad_x[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        svd_factorize(bad_x, 4)
    with pytest.raises(ValueError):
        svd_factorize(x.ravel(), 4)
```

Four invalid ranks — zero, negative, above maximum, and a float — plus a non-finite
value and a 1-D input.

`2.5` is the subtle one: a float rank would slice arrays in ways that appear to work
in some NumPy versions, so the explicit `isinstance` check in `svd_factorize` is
tested here rather than assumed.

### Lines 70–76 — `test_input_not_mutated`

```python
def test_input_not_mutated():
    x = make_synthetic(50, seed=8)
    before = x.copy()
    svd_factorize(x, 4)
    analyse_ranks(x, [1, 4])
    assert np.array_equal(x, before)
```

Confirms neither function modifies its input — a brief requirement, and a real
concern because both convert to float64 internally, which could have been done in
place.

This matters practically: `run_render_eval.py` extracts the SH matrix **once** and
passes it to 12 configurations in sequence. If any of them mutated it, every
subsequent configuration would silently operate on corrupted data.

### Lines 79–83 — `test_chunked_reconstruction_agrees`

```python
def test_chunked_reconstruction_agrees():
    x = make_synthetic(1000, rank=6, seed=9)
    u, v = svd_factorize(x, 6)
    assert np.allclose(reconstruct(u, v, chunk_size=100), reconstruct(u, v), atol=1e-6)
```

Verifies the memory optimisation does not alter results. With `chunk_size=100` and
1000 rows, ten chunks are exercised against the single-pass default.

## Why it matters

This file is the Stage 3 gate, and it embodies the brief's core methodological
demand: validate the mathematics on data with known answers *before* trusting it on
data where the answer is what you are trying to discover.

Because these ten tests pass, the "effective rank 12" finding on the real scene is
produced by machinery already demonstrated to recover known ranks exactly, respect
monotonicity, reject invalid input, and leave its inputs alone. The finding is
therefore evidence about the *data*, not a possible artefact of the code.
