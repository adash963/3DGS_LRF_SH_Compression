# `tests/unit/test_quantization.py`

## Purpose

Verifies both quantisers: error bounds, determinism, edge cases, and the empty-cluster
repair. Determinism is the load-bearing property here — the dissertation claims VQ
results are reproducible, and that claim is only as good as these tests.

## Line-by-line

### Lines 8–14 — `test_scalar_roundtrip_error_bounded`

```python
def test_scalar_roundtrip_error_bounded():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 8)).astype(np.float32)
    q = scalar_quantize(x, bits=8)
    span = x.max(axis=0) - x.min(axis=0)
    assert np.all(np.abs(q.decode() - x) <= span / 255 * 0.51 + 1e-6)
```

Asserts the **theoretical error bound** rather than an empirical threshold.

Uniform quantisation to 255 levels has a step of `span/255`, and rounding to nearest
gives a maximum error of half a step. The test uses `0.51` rather than `0.5` to allow
for float32 representation error in the scale factor, plus a small absolute epsilon.

This is a much stronger test than "error is small": it verifies the quantiser achieves
the *mathematically optimal* bound for its design, so a subtly wrong scale or offset
would fail even though errors would still look small.

### Lines 17–22 — `test_scalar_16bit_better_than_8bit`

```python
def test_scalar_16bit_better_than_8bit():
    ...
    e8 = np.abs(scalar_quantize(x, 8).decode() - x).max()
    e16 = np.abs(scalar_quantize(x, 16).decode() - x).max()
    assert e16 < e8
```

More bits must mean less error. Simple, but it catches a bit-width parameter that is
accepted and then ignored — a failure mode that no error-bound test would detect,
since 8-bit error also satisfies a 16-bit-shaped bound loosely.

### Lines 25–30 — `test_scalar_invalid`

```python
def test_scalar_invalid():
    with pytest.raises(ValueError):
        scalar_quantize(np.zeros((5, 2)), bits=7)
    bad = np.zeros((5, 2)); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        scalar_quantize(bad)
```

7 bits is rejected because no NumPy integer type holds it exactly; permitting it
would mean storing 7-bit values in 8-bit slots while reporting the smaller figure.
NaN input is rejected for the usual reason.

### Lines 33–43 — `test_vq_roundtrip_and_bounds`

```python
def test_vq_roundtrip_and_bounds():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1000, 8)).astype(np.float32)
    q = kmeans_vq(x, 64, seed=0)
    dec = q.decode()
    assert dec.shape == x.shape
    assert q.indices.max() < q.codebook.shape[0]
    assert q.index_bits == 6
    single = kmeans_vq(x, 1, seed=0)
    assert np.mean((dec - x) ** 2) < np.mean((single.decode() - x) ** 2)
```

Four checks: output shape, index bounds, correct bit calculation (`log₂64 = 6`), and
— most usefully — that 64 centroids beat **one** centroid.

That last comparison is a meaningful baseline. A single centroid is the mean of the
data, so any working clustering must do better; a broken implementation that assigned
everything to one cluster would fail here.

### Lines 46–51 — `test_vq_deterministic`

```python
def test_vq_deterministic():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 4)).astype(np.float32)
    q1, q2 = kmeans_vq(x, 32, seed=7), kmeans_vq(x, 32, seed=7)
    assert np.array_equal(q1.indices, q2.indices)
    assert np.array_equal(q1.codebook, q2.codebook)
```

**The test the reproducibility claim depends on.** K-means is randomly initialised;
without a fixed seed, two runs produce different codebooks and different — though
similarly good — results.

Exact equality is asserted on both codebook and indices. Because the seed is also
stored in the saved asset, this establishes that a VQ result can be regenerated
rather than merely reused.

### Lines 54–61 — `test_vq_unique_vectors_exact`

```python
def test_vq_unique_vectors_exact():
    """A codebook holding every unique vector reproduces the input exactly."""
    rng = np.random.default_rng(4)
    base = rng.normal(size=(10, 4)).astype(np.float32)
    x = base[rng.integers(0, 10, size=300)]
    q = kmeans_vq(x, 16, seed=0)
    assert np.allclose(q.decode(), x, atol=1e-6)
```

Constructs 300 rows drawn from only 10 distinct vectors, then quantises with a
codebook of 16. Since 10 ≤ 16, every distinct vector gets its own entry and
**VQ becomes lossless**.

This exercises the unique-rows fast path in `kmeans_vq` and provides a rare
quantisation case with an exact expected answer — a much stronger assertion than any
error threshold.

### Lines 64–71 — `test_vq_no_empty_clusters`

```python
def test_vq_no_empty_clusters():
    rng = np.random.default_rng(5)
    # pathological: tight cluster + few outliers encourages empty clusters
    x = np.vstack([rng.normal(scale=1e-4, size=(500, 4)),
                   rng.normal(loc=100, size=(3, 4))]).astype(np.float32)
    q = kmeans_vq(x, 32, seed=0)
    counts = np.bincount(q.indices, minlength=q.codebook.shape[0])
    assert (counts > 0).all()
```

A **deliberately hostile** distribution: 500 points in a very tight cluster (scale
1e-4) plus 3 far outliers at distance 100. Standard k-means with 32 centroids
typically strands several with no assigned points.

The test asserts every codebook entry is used, exercising `_fix_empty_clusters`.
Empty entries would waste both codebook storage and index space — the bits spent
addressing them buy nothing.

### Lines 74–80 — `test_vq_larger_codebook_not_worse`

```python
def test_vq_larger_codebook_not_worse():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(2000, 8)).astype(np.float32)
    e = {k: float(np.mean((kmeans_vq(x, k, seed=0).decode() - x) ** 2))
         for k in (16, 64, 256)}
    assert e[256] <= e[64] * 1.05 and e[64] <= e[16] * 1.05
```

More centroids should not produce meaningfully worse error. The 5% tolerance is
deliberate and worth explaining: k-means finds a *local* optimum, so a larger
codebook can occasionally land slightly worse by chance. Demanding strict improvement
would produce a flaky test; forbidding meaningful regression catches real bugs while
tolerating the algorithm's nature.

This is a case where a looser threshold is the *correct* choice rather than a
weakening — the brief forbids weakening thresholds to obtain a pass, and this
tolerance was chosen from the algorithm's properties, not from an observed failure.

### Lines 83–88 — `test_vq_invalid`

```python
def test_vq_invalid():
    with pytest.raises(ValueError):
        kmeans_vq(np.zeros((10, 2)), 0)
    bad = np.zeros((10, 2)); bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        kmeans_vq(bad, 4)
```

Zero codebook size and non-finite input both rejected.

## Why it matters

This file underwrites two dissertation claims.

**Determinism.** `test_vq_deterministic` plus the stored seed means every VQ result
in the results tables is regenerable, not merely repeatable-in-principle.

**Correct byte accounting.** The empty-cluster and index-bounds tests ensure the
codebook is fully utilised and indices are valid, so the reported storage — codebook
plus indices — describes a structure that is genuinely all in use.
