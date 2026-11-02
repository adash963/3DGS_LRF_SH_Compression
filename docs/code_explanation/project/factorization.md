# `src/sh_compression/factorization.py`

## Purpose

The mathematical heart of the project: truncated singular value decomposition of the
scene SH matrix, the rank-sweep analysis that produced the "effective rank 12"
result, and synthetic data generators used to prove the mathematics before trusting
it on real data.

This implements the dissertation's core formulation `X ≈ U·V`.

## Line-by-line

### Lines 1–7 — Imports

```python
"""Truncated-SVD low-rank factorisation of the scene SH matrix X [N, 48]."""

from dataclasses import dataclass
import numpy as np
from .sh_layout import FLAT_DIM
```

Pure NumPy — no GPU dependency. This is deliberate: the factorisation runs on the
`N × 48` matrix, and because 48 is tiny, a full SVD is cheap even at N > 1,000,000.
Keeping it CPU-only means rank analysis runs on a laptop with no CUDA.

### Lines 10–15 — `_check_input`

```python
def _check_input(x: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError(f"expected 2-D matrix, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("input matrix contains NaN/Inf")
```

Shared validation. The NaN check matters especially here: `np.linalg.svd` on a
matrix containing a single NaN returns *all*-NaN factors, so the error would surface
far downstream with no clue as to its origin.

### Lines 18–32 — `svd_factorize`

```python
def svd_factorize(x: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """X ~= U V with U [N, r] (singular values folded in) and V [r, D]."""
    _check_input(x)
    max_rank = min(x.shape)
    if not (isinstance(rank, (int, np.integer)) and 1 <= rank <= max_rank):
        raise ValueError(f"rank must be an int in 1..{max_rank}, got {rank!r}")
    u, s, vt = np.linalg.svd(np.asarray(x, dtype=np.float64), full_matrices=False)
    u_r = (u[:, :rank] * s[:rank]).astype(np.float32)
    v_r = vt[:rank].astype(np.float32)
    return u_r, v_r
```

**The central function.**

- Line 21: `max_rank = min(N, 48)` — for real scenes this is 48. A rank above this is
  mathematically meaningless (the matrix cannot have more independent directions
  than its narrower dimension).
- Line 22 validates. The `isinstance` check rejects floats: `rank=4.0` would
  otherwise slice arrays in a way that silently misbehaves in some NumPy versions.
  `{rank!r}` uses `repr`, so `4.0` and `'4'` are distinguishable in the message.
- **Line 24 computes the SVD in float64**, even though inputs and outputs are
  float32. Singular value computation involves accumulating sums of squares across
  a million rows; doing that in float32 loses precision in the small singular values,
  which are exactly the ones the rank analysis inspects. `full_matrices=False`
  requests the "economy" SVD — instead of an N×N matrix `u` (which for a million
  Gaussians would be 4 terabytes), it returns N×48.
- **Line 25 folds the singular values into U.** Standard SVD gives `X = U·Σ·Vᵀ` with
  three factors. Writing `u[:, :rank] * s[:rank]` scales each column of U by its
  singular value, absorbing Σ, so the result is two factors: `X ≈ U·V`. This is
  reported in the dissertation as "singular values folded in".

  Two reasons for it: the storage format only needs to hold two arrays, and the
  quantiser sees latent values already scaled by importance — meaning a fixed-range
  quantiser naturally allocates its precision to the components that matter.
- Line 26 keeps the first `rank` rows of `Vᵀ`, giving the shared basis `V [r, 48]`.
- Output is float32, halving storage relative to float64 with no measurable quality
  cost.

### Lines 35–43 — `reconstruct`

```python
def reconstruct(u: np.ndarray, v: np.ndarray, chunk_size: int = 1_000_000) -> np.ndarray:
    n = u.shape[0]
    out = np.empty((n, v.shape[1]), dtype=np.float32)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        out[start:stop] = u[start:stop].astype(np.float32) @ v.astype(np.float32)
    return out
```

Computes `U @ V` in row blocks. Mathematically a single matrix multiply would do;
chunking bounds peak memory, since a full-rank intermediate for a million Gaussians
is ~200 MB and Colab runtimes are memory-constrained.

The `.astype(np.float32)` calls handle the fp16 ablation: when factors are stored as
float16, they are promoted before multiplication (float16 matmul would both lose
precision and be slow on CPU).

`test_factorization.py::test_chunked_reconstruction_agrees` asserts chunked and
unchunked results match, so the optimisation cannot alter results.

### Lines 46–68 — `RankAnalysis`

```python
@dataclass
class RankAnalysis:
    singular_values: np.ndarray
    ranks: list[int]
    rel_frobenius_error: list[float]
    mse: list[float]
    max_abs_error: list[float]
    explained_energy: list[float]
    theoretical_bytes_fp32: list[int]

    def as_rows(self) -> list[dict]:
        ...
```

A results container holding parallel lists indexed by rank. `as_rows()` transposes
it into one dict per rank, which is the shape a CSV writer wants — this is how
`analyse_rank.py` produces `error_vs_rank.csv` without hand-formatting.

Note `theoretical_bytes_fp32` is explicitly named *theoretical*: it is the raw
arithmetic size, distinct from the measured `.npz` file size reported everywhere in
the results. Keeping the distinction in the field name prevents the two being
confused.

### Lines 71–96 — `analyse_ranks`

```python
def analyse_ranks(x: np.ndarray, ranks: list[int]) -> RankAnalysis:
    _check_input(x)
    x64 = np.asarray(x, dtype=np.float64)
    u, s, vt = np.linalg.svd(x64, full_matrices=False)
    max_rank = len(s)
    usable = [r for r in ranks if 1 <= r <= max_rank]
    total_energy = float(np.sum(s**2))
    fro_x = float(np.linalg.norm(x64))
```

**This function produced the dissertation's effective-rank result.**

Line 75 computes the SVD *once* and reuses it for every rank — since a truncated
reconstruction at rank r just takes the first r components, there is no need to
re-decompose. Line 77 silently drops out-of-range ranks so a standard sweep list can
be reused on small test matrices.

Line 78 defines total energy as `Σσ²`. Line 79 gets the Frobenius norm for
normalisation.

```python
    for r in usable:
        recon = (u[:, :r] * s[:r]) @ vt[:r]
        diff = x64 - recon
        res.rel_frobenius_error.append(
            float(np.linalg.norm(diff)) / fro_x if fro_x > 0 else 0.0
        )
        res.mse.append(float(np.mean(diff**2)))
        res.max_abs_error.append(float(np.max(np.abs(diff))) if diff.size else 0.0)
        res.explained_energy.append(
            float(np.sum(s[:r] ** 2)) / total_energy if total_energy > 0 else 1.0
        )
        res.theoretical_bytes_fp32.append(4 * (x.shape[0] * r + r * x.shape[1]))
```

For each rank: reconstruct (line 82), then compute four complementary error views —

- **relative Frobenius error** — overall error as a fraction of signal magnitude;
  scale-free and the headline number in the results tables
- **MSE** — mean squared error, comparable to image-domain MSE
- **max absolute error** — worst single coefficient, revealing outliers that averages
  hide
- **explained energy** — `Σᵣσ² / Σσ²`, the fraction of variance captured. **This is
  the quantity that yields "effective rank 12": the smallest r reaching 99%.**

The `if fro_x > 0` / `if total_energy > 0` guards handle an all-zero matrix without
dividing by zero.

Line 92: theoretical bytes = `4 × (N·r + r·48)` — four bytes per float32 across both
factors.

### Lines 99–103 — `effective_rank`

```python
def effective_rank(s: np.ndarray, energy_threshold: float = 0.99) -> int:
    energy = np.cumsum(s**2) / np.sum(s**2)
    return int(np.searchsorted(energy, energy_threshold) + 1)
```

Line 101 builds the cumulative energy curve; line 102 binary-searches for the
threshold crossing. The `+1` converts a 0-based index to a 1-based rank. **On the
real scene this returns 12** — the single most-cited number in the dissertation's
motivation.

### Lines 106–111 — `make_synthetic`

```python
def make_synthetic(n, dim=FLAT_DIM, rank=None, noise=0.0, seed=0, constant=False):
    rng = np.random.default_rng(seed)
    if constant:
        return np.tile(rng.normal(size=(1, dim)), (n, 1)).astype(np.float32)
    if rank is None:
        x = rng.normal(size=(n, dim))
    else:
        x = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, dim))
    if noise > 0:
        x = x + rng.normal(scale=noise, size=x.shape)
    return x.astype(np.float32)
```

Generates test matrices with **known** rank structure — the basis of Stage 3's
verification.

The construction on the `rank=r` path is the important one: multiplying `[n, r]` by
`[r, dim]` yields a product that cannot have rank above `r`, by the rank inequality
of matrix products. So the true answer is known in advance, and the SVD's output can
be checked against ground truth rather than merely inspected for plausibility.

`constant=True` tiles one row N times, producing an exactly rank-1 matrix — a
degenerate case worth testing separately. `noise` perturbs an exact low-rank matrix
to simulate realistic data whose spectrum decays rather than truncating sharply.
`default_rng(seed)` makes every generated matrix reproducible.

## Why it matters

The scientific credibility of Stage 3 rests on this file being validated *before*
being trusted. `test_factorization.py` uses `make_synthetic` to assert that exact
rank-k data reconstructs at rank k to within 1e-6, that rank k−1 is strictly worse,
that error decreases monotonically with rank, and that inputs are never mutated.

Only after those pass is the same code pointed at the real 1,071,462-Gaussian
matrix — so the "effective rank 12" finding comes from machinery already
demonstrated correct on data with known answers.
