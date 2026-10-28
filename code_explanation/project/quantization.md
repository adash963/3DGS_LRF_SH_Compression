# `src/sh_compression/quantization.py`

## Purpose

The second compression stage. After factorisation reduces each Gaussian to `r`
latent numbers, this file reduces the cost of *storing* those numbers, by two
independent methods:

- **Scalar quantisation** — map each float to an 8- or 16-bit integer
- **K-means vector quantisation** — replace each Gaussian's whole latent vector with
  an index into a shared codebook

Both operate on `U` (the `[N, r]` per-Gaussian factor), never on `V`. That choice is
deliberate: `U` holds `N × r` values (millions), while `V` holds only `r × 48` (a
few thousand). Compressing `V` would save nothing measurable while adding error to
the basis every Gaussian depends on.

## Line-by-line

### Lines 1–8 — Imports

```python
"""Quantisation of the per-Gaussian U factor: scalar (uniform) and K-means VQ."""

from dataclasses import dataclass
import numpy as np
from scipy.cluster.vq import kmeans2
```

`scipy`'s `kmeans2` is used rather than scikit-learn's — it is a lighter dependency
and accepts an explicit `seed`, which is required for the determinism the project
claims.

## Scalar quantisation

### Lines 13–21 — `ScalarQuantized`

```python
@dataclass
class ScalarQuantized:
    codes: np.ndarray          # uint8/uint16 [N, r]
    scale: np.ndarray          # float32 [r]
    offset: np.ndarray         # float32 [r]
    bits: int

    def decode(self) -> np.ndarray:
        return (self.codes.astype(np.float32) * self.scale + self.offset).astype(np.float32)
```

Holds the quantised representation. Note the shapes: `codes` is `[N, r]` (one
integer per Gaussian per latent dimension) but `scale` and `offset` are `[r]` — one
pair per *column*, not per element. That is what makes this cheap: the metadata is
`2r` floats regardless of how many Gaussians there are.

`decode` reverses the mapping via broadcasting — `codes` is `[N, r]`, `scale` is
`[r]`, and NumPy applies each column's scale down the whole column.

(The dissertation's methodology originally mis-stated this cost as `2·48·r·4` bytes;
the correct figure is `2·r·4`. The static audit caught it; measured file sizes were
never affected.)

### Lines 24–40 — `scalar_quantize`

```python
def scalar_quantize(x: np.ndarray, bits: int = 8) -> ScalarQuantized:
    if bits not in (8, 16):
        raise ValueError(f"bits must be 8 or 16, got {bits}")
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("input must be a finite 2-D matrix")
    levels = (1 << bits) - 1
    lo = x.min(axis=0).astype(np.float64)
    hi = x.max(axis=0).astype(np.float64)
    span = np.where(hi > lo, hi - lo, 1.0)
    codes = np.round((x - lo) / span * levels)
    dtype = np.uint8 if bits == 8 else np.uint16
    return ScalarQuantized(
        codes=codes.astype(dtype),
        scale=(span / levels).astype(np.float32),
        offset=lo.astype(np.float32),
        bits=bits,
    )
```

- Line 29: `levels = 2^bits − 1` → 255 for 8-bit. Using 255 rather than 256 means
  both endpoints are exactly representable: code 0 maps to `lo`, code 255 to `hi`.
- **Lines 30–31 compute per-column min and max** (`axis=0` reduces over Gaussians).
  This is what makes the scheme adaptive: because singular values were folded into
  `U`, its first column has a much larger range than its last, and per-column ranges
  give each the full integer range rather than wasting precision on a shared scale.
- Line 32 guards against a constant column, where `hi == lo` would divide by zero.
  Substituting `1.0` maps every value to code 0, and decode returns `offset` — which
  is exactly the constant. Correct behaviour, no special case needed downstream.
- Line 33 normalises to `[0, 1]`, scales to `[0, levels]`, and rounds. Maximum error
  is half a step, i.e. `span / (2·255)`.
- Line 34 picks the smallest integer type that fits.

Quality result: at rank 16 this is indistinguishable from unquantised float32
(34.63 dB vs 34.68 dB) at roughly a quarter of the storage.

## Vector quantisation

### Lines 45–60 — `VQQuantized`

```python
@dataclass
class VQQuantized:
    codebook: np.ndarray       # float32 [K, r]
    indices: np.ndarray        # uint8/uint16/uint32 [N]
    codebook_size: int
    seed: int

    @property
    def index_bits(self) -> int:
        return int(np.ceil(np.log2(max(self.codebook_size, 2))))

    def decode(self) -> np.ndarray:
        if self.indices.max(initial=0) >= self.codebook.shape[0]:
            raise ValueError("index out of codebook bounds")
        return self.codebook[self.indices].astype(np.float32)
```

The compression here is dramatic and worth stating plainly: instead of `r` floats per
Gaussian, each Gaussian stores **one integer**. At K=256 that is one byte, versus 64
bytes for 16 float32s.

- `index_bits` returns `⌈log₂K⌉` — the *logical* information content. Note the
  distinction the audit flagged: `_index_dtype` stores uint8 for all K ≤ 256, so at
  K=64 the logical 6 bits occupy 8 physical bits. The property describes entropy, not
  the current file layout.
- `decode` is pure fancy indexing: `codebook[indices]` takes a `[N]` integer array
  and returns `[N, r]` by looking up each row. Line 58's bounds check guards against
  a corrupted file producing an out-of-range read.

### Lines 63–68 — `_index_dtype`

```python
def _index_dtype(k: int):
    if k <= 256:
        return np.uint8
    if k <= 65536:
        return np.uint16
    return np.uint32
```

Smallest integer type that can address the codebook.

### Lines 71–93 — `kmeans_vq`

```python
def kmeans_vq(x, codebook_size, seed=0, iters=50) -> VQQuantized:
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("input must be a finite 2-D matrix")
    if codebook_size < 1:
        raise ValueError(f"codebook_size must be >= 1, got {codebook_size}")
    x64 = np.asarray(x, dtype=np.float64)
    unique_rows = np.unique(x64, axis=0)
    if len(unique_rows) <= codebook_size:
        codebook = unique_rows
        idx = _assign(x64, codebook)
    else:
        codebook, idx = kmeans2(x64, codebook_size, iter=iters, minit="++", seed=seed)
        codebook, idx = _fix_empty_clusters(x64, codebook, idx)
```

- **Lines 82–85 handle the exact-representation case.** If the data contains no more
  distinct rows than the codebook has slots, every row can be stored losslessly.
  This is not just an optimisation — it is a *testable correctness property*
  (`test_vq_unique_vectors_exact`), giving a case where VQ must be exactly lossless.
- Line 87 runs k-means. `minit="++"` selects k-means++ initialisation, which spreads
  initial centroids rather than picking randomly, yielding better and more stable
  results. `seed=seed` makes it reproducible — the project's determinism claim
  depends on this argument.
- Line 88 repairs empty clusters (below).

```python
    k = codebook.shape[0]
    return VQQuantized(
        codebook=codebook.astype(np.float32),
        indices=idx.astype(_index_dtype(k)),
        codebook_size=k,
        seed=seed,
    )
```

`k` is read back from the actual codebook rather than the requested size, because the
unique-rows path may produce fewer entries.

### Lines 96–105 — `_assign`

```python
def _assign(x: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    out = np.empty(len(x), dtype=np.int64)
    step = max(1, 2_000_000 // max(1, codebook.shape[0]))
    for s in range(0, len(x), step):
        e = min(s + step, len(x))
        d = ((x[s:e, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
        out[s:e] = d.argmin(axis=1)
    return out
```

Nearest-centroid assignment. Line 101 is a broadcast trick: `x[s:e, None, :]` has
shape `[chunk, 1, r]` and `codebook[None, :, :]` has `[1, K, r]`, so subtracting
gives `[chunk, K, r]` — every point against every centroid. Summing over the last
axis gives squared distances `[chunk, K]`, and `argmin` picks the closest.

That intermediate is why chunking is mandatory: for a million points and K=256 at
r=16, a single unchunked array would be ~32 GB. Line 98 sizes the chunk so the
intermediate stays around 2 million elements per row-block regardless of K.

### Lines 108–120 — `_fix_empty_clusters`

```python
def _fix_empty_clusters(x, codebook, idx):
    counts = np.bincount(idx, minlength=codebook.shape[0])
    empty = np.flatnonzero(counts == 0)
    if len(empty) == 0:
        return codebook, idx
    d = ((x - codebook[idx]) ** 2).sum(axis=1)
    worst = np.argsort(d)[::-1]
    codebook = codebook.copy()
    for i, c in enumerate(empty):
        codebook[c] = x[worst[i % len(worst)]]
    idx = _assign(x, codebook)
    return codebook, idx
```

Handles a real k-means failure mode: a centroid can end up owning no points, wasting
a codebook slot and the bits spent addressing it.

Lines 109–110 count assignments and find empty clusters. Line 113 computes each
point's squared distance to its assigned centroid — its *representation error*. Line
114 sorts descending, so `worst[0]` is the worst-represented point. Lines 116–117
move each empty centroid onto one of those worst points, which both fills the slot
and targets the largest current error. Line 118 reassigns everything against the
repaired codebook.

`copy()` on line 115 avoids mutating scipy's returned array — an aliasing bug that
would be hard to diagnose.

`test_quantization.py::test_vq_no_empty_clusters` exercises this with a pathological
distribution (a tight cluster plus three far outliers) designed to strand centroids.

## Why it matters

This file produced the dissertation's most interesting analytical finding. VQ
achieves 201–264× compression, but its quality is nearly *rank-independent*: r16/K256
beats r8/K256 by only 0.37 dB.

The explanation is visible right here in the data structure. Each Gaussian's entire
identity is one index — `⌈log₂K⌉` bits — no matter how expressive the latent space
is. Increasing `r` enriches the codebook's vectors but not each Gaussian's ability to
select among them. **The information bottleneck moves from the basis to the index
entropy**, which is why further gains would come from entropy-coding the indices
rather than from richer codebooks.
