# `tests/unit/test_metrics.py`

## Purpose

Validates the image-quality metrics and the storage-ratio arithmetic. The brief
required metric code to be checked in both directions — identical images should score
perfectly, degraded images should score worse — before any measurement was trusted.

## Line-by-line

### Lines 1–6 — Imports

```python
import numpy as np
import pytest

from sh_compression.metrics import psnr, ssim
from sh_compression.storage import baseline_sh_bytes, ratios
```

### Lines 9–14 — `test_identical_images_perfect`

```python
def test_identical_images_perfect():
    rng = np.random.default_rng(0)
    img = rng.random((32, 32, 3)).astype(np.float32)
    assert psnr(img, img) == float("inf")
    assert ssim(img, img) == pytest.approx(1.0)
```

The upper-bound anchor. Identical images must give infinite PSNR (MSE is exactly
zero) and SSIM of 1.0.

**This test is what makes the Stage 2 result interpretable.** When the round-trip
comparison returned `inf` on all 38 real views, that was recognisable as "pixel
identical" precisely because the metric's behaviour on identical input was already
established. Without it, `inf` could plausibly have been a bug.

`pytest.approx` is used for SSIM because its windowed computation accumulates tiny
floating-point error even on identical input; PSNR's `inf` is exact because it comes
from an explicit `mse == 0` branch.

### Lines 17–24 — `test_perturbed_images_worse`

```python
def test_perturbed_images_worse():
    rng = np.random.default_rng(1)
    img = rng.random((64, 64, 3)).astype(np.float32)
    small = np.clip(img + rng.normal(scale=0.01, size=img.shape), 0, 1).astype(np.float32)
    big = np.clip(img + rng.normal(scale=0.2, size=img.shape), 0, 1).astype(np.float32)
    assert psnr(img, small) > psnr(img, big)
    assert ssim(img, small) > ssim(img, big)
```

The direction test: a lightly perturbed image (noise σ=0.01) must score better than a
heavily perturbed one (σ=0.2), on **both** metrics.

A metric that returned a constant, or had its sign inverted, or ignored one of its
arguments, would pass the identical-images test but fail here. The two tests together
establish that the metrics respond correctly in both magnitude and direction — which
is the brief's requirement, and the minimum needed before quoting metric values as
evidence.

`np.clip(..., 0, 1)` keeps the perturbed images in valid range, so the metrics are
computed on data matching their `data_range=1.0` assumption.

### Lines 27–29 — `test_shape_mismatch_rejected`

```python
def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        psnr(np.zeros((4, 4, 3)), np.zeros((5, 5, 3)))
```

Mismatched shapes must raise rather than broadcast. Without the explicit check, NumPy
might broadcast some shape combinations into a meaningless number — silently
comparing images of different resolutions, which would be a serious methodological
error if two configurations somehow rendered at different sizes.

### Lines 32–39 — `test_storage_ratios`

```python
def test_storage_ratios():
    n = 1000
    assert baseline_sh_bytes(n) == n * 48 * 4
    r = ratios(n, compressed_sh_file_bytes=n * 48, baseline_ply_bytes=n * 62 * 4)
    assert r["sh_compression_ratio"] == pytest.approx(4.0)
    assert r["whole_model_bytes_compressed"] == n * 14 * 4 + n * 48
    assert 1.0 < r["whole_model_ratio"] < 4.0
```

Verifies the ratio arithmetic on hand-checkable numbers.

The setup is chosen so the expected answers are obvious: baseline SH is `n×48×4`
bytes, compressed is `n×48` bytes (a quarter), so the SH ratio is exactly **4.0**.

Line 36 checks the whole-model composite. The model has 62 columns of which 48 are
SH, so 14 are geometry: `n×14×4` bytes of geometry plus `n×48` bytes of compressed
colour.

**Line 37 asserts the whole-model ratio sits strictly between 1 and 4** — an
inequality rather than a value, and it encodes a conceptual truth: whole-model
compression must be *worse* than SH-only compression, because uncompressed geometry
dilutes the gain. If this assertion ever failed, the ratio definitions would be
wrong.

That relationship is exactly why the report quotes both figures (15× SH-only, 3.6×
whole-model) rather than only the flattering one.

## Why it matters

These five tests are small but they license every quality number in the dissertation.

The pair of anchor tests — perfect on identical, monotonic on degraded — means the
metrics are known to behave correctly at the extreme the project cares most about
(near-lossless) and across the range it measures. The storage tests mean the ratios
are arithmetic that has been checked against worked examples rather than trusted.
