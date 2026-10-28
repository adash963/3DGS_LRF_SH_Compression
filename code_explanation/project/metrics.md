# `src/sh_compression/metrics.py`

## Purpose

Image-quality measurement: PSNR, SSIM and LPIPS computed over directories of
rendered images. This is where compression results become *quality* results — the
numbers that appear in every rate-distortion figure.

## Line-by-line

### Lines 1–6 — Imports

```python
"""Image-quality metrics (PSNR, SSIM, optional LPIPS) over render directories."""

import json, os
import numpy as np
```

Deliberately light at module level. Heavy dependencies (`imageio`, `skimage`,
`torch`, `lpips`) are imported *inside* the functions that need them, so importing
this module on a machine without them still works — which is what allows the metric
sanity tests to run in a minimal environment.

### Lines 9–22 — `load_image`

```python
def load_image(path: str) -> np.ndarray:
    """Load an image as float32 [H, W, 3] in [0, 1]."""
    import imageio.v3 as iio

    img = iio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0
    return img
```

Normalises everything to one representation: float32, three channels, range `[0, 1]`.

- Lines 14–15 expand greyscale to three channels so downstream code never needs a
  special case.
- Lines 16–17 drop an alpha channel if present. Renders have no meaningful alpha, and
  including it would distort metrics.
- **Lines 19–20 are a heuristic**: if the maximum exceeds 1.5, the image is assumed
  to be 0–255 integer data and divided by 255. The threshold sits between the two
  plausible ranges. The edge case — a genuinely float image whose brightest pixel is
  below 1.5 — is not a concern here because renders and ground-truth photos are both
  8-bit PNGs.

Consistent normalisation is a correctness requirement, not a convenience: PSNR
depends on the data range, so comparing a `[0,1]` image against a `[0,255]` one would
produce a meaningless number rather than an error.

### Lines 25–31 — `psnr`

```python
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))
```

Peak Signal-to-Noise Ratio: `10·log₁₀(MAX²/MSE)`, with MAX = 1.0 because images are
normalised, so the numerator is simply 1.

- Line 26 rejects mismatched shapes rather than letting broadcasting produce a wrong
  answer.
- Line 28 accumulates in **float64**. Summing millions of squared differences in
  float32 loses precision exactly when the differences are small — which is the
  high-PSNR regime this project cares most about.
- **Lines 29–30 return infinity for identical images.** This is not a corner case
  here: it is the Stage 2 result. All 38 round-trip views returned `inf`, which is
  how "pixel-identical" was established. Returning `inf` rather than capping keeps
  that distinguishable from merely very good.

The `101.14 dB` full-rank figure comes from this function — finite, and therefore
correctly described as *numerically equivalent* rather than bit-identical.

### Lines 34–37 — `ssim`

```python
def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))
```

Structural Similarity, delegated to scikit-image's reference implementation rather
than reimplemented — a deliberate choice, since a hand-rolled SSIM with a subtly
wrong window or constant would produce plausible-looking but incomparable numbers.

`channel_axis=-1` tells it colour is the last axis (so it computes across channels
correctly rather than treating them as spatial); `data_range=1.0` matches the
normalisation. Both arguments are required for correctness, not defaults.

SSIM measures local structural agreement (luminance, contrast, structure) rather than
per-pixel difference, so it complements PSNR: it responds to *structural* damage like
blur or blocking, which PSNR under-weights.

### Lines 40–58 — `lpips_distance`

```python
_LPIPS_MODEL = None
_LPIPS_DEVICE = None


def lpips_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    """LPIPS (AlexNet), on CUDA when available. Returns None when unavailable."""
    global _LPIPS_MODEL, _LPIPS_DEVICE
    try:
        import lpips
        import torch
    except ImportError:
        return None
    if _LPIPS_MODEL is None:
        _LPIPS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False).to(_LPIPS_DEVICE)
    def to_t(x):
        t = torch.from_numpy(x.transpose(2, 0, 1)[None] * 2.0 - 1.0).float()
        return t.to(_LPIPS_DEVICE)
    with torch.no_grad():
        return float(_LPIPS_MODEL(to_t(a), to_t(b)).item())
```

Learned Perceptual Image Patch Similarity — a *perceptual* metric that compares deep
network features rather than pixels, and correlates better with human judgement than
PSNR or SSIM. Lower is better (it is a distance).

- **Lines 47–50 make it optional.** If `lpips` or `torch` is missing, it returns
  `None` instead of raising, and `compare_dirs` simply omits the metric. This is what
  allowed the pipeline to run before LPIPS was installed, and it implements the
  brief's instruction that LPIPS is preferred but may be documented as omitted.
- **Lines 51–54 cache the model in a module-level global.** Loading AlexNet weights
  takes seconds; without caching that cost would be paid 38 times per configuration,
  or 456 times across the evaluation.
- Line 53 selects CUDA when available — added specifically so the LPIPS pass over 12
  configurations × 38 views × 2 references (912 comparisons) ran in minutes on the T4
  rather than tens of minutes on Colab's weak CPU.
- Line 56 does two conversions: `transpose(2,0,1)` reorders `[H,W,C]` to PyTorch's
  `[C,H,W]`, `[None]` adds a batch dimension, and `* 2.0 - 1.0` rescales `[0,1]` to
  `[-1,1]`, which is the range the LPIPS network expects. Getting that scaling wrong
  would produce numbers that look reasonable but are systematically off.
- Line 58 wraps in `torch.no_grad()` — no gradients are needed, and this avoids
  building an autograd graph over every comparison.

### Lines 61–91 — `compare_dirs`

```python
def compare_dirs(dir_a: str, dir_b: str, use_lpips: bool = True) -> dict:
    names_a = sorted(f for f in os.listdir(dir_a) if f.lower().endswith((".png", ".jpg")))
    names_b = sorted(f for f in os.listdir(dir_b) if f.lower().endswith((".png", ".jpg")))
    common = sorted(set(names_a) & set(names_b))
    if not common:
        raise ValueError(f"no common images between {dir_a} and {dir_b}")
```

The workhorse that produced every metric in the results.

- Lines 66–67 list image files, **sorted**. `os.listdir` returns arbitrary order;
  sorting makes results deterministic and pairs images by name. This directly
  satisfies the brief's "file ordering is deterministic" requirement.
- Line 68 intersects the two sets, so images are matched **by filename** rather than
  by position — the guard against comparing view 5 of one set against view 12 of
  another if a render were missing.
- Lines 69–70 fail loudly on no overlap, catching a wrong-path mistake immediately.

```python
    per_view, failed = [], []
    for name in common:
        try:
            a = load_image(os.path.join(dir_a, name))
            b = load_image(os.path.join(dir_b, name))
            row = {"view": name, "psnr": psnr(a, b), "ssim": ssim(a, b)}
            if use_lpips:
                d = lpips_distance(a, b)
                if d is not None:
                    row["lpips"] = d
            per_view.append(row)
        except Exception as e:  # record, never silently drop
            failed.append({"view": name, "error": str(e)})
```

Lines 72–84 compute metrics per view. Two design points:

- **Per-view results are retained**, not just averages. This allows checking whether
  a mean is driven by one catastrophic view, and satisfies the brief's requirement to
  save per-view metrics.
- **Failures are recorded, not swallowed.** The `except` appends to `failed` rather
  than skipping silently — the difference between "38 views, 0 failed" and "36 views
  averaged, 2 quietly dropped", which would bias the mean upward. The inline comment
  states this intent.

```python
    summary = {}
    for key in ("psnr", "ssim", "lpips"):
        vals = [r[key] for r in per_view if key in r and np.isfinite(r[key])]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
```

Lines 85–89 average each metric. **The `np.isfinite` filter matters**: if any view
returned `inf` PSNR, including it would make the mean `inf` and destroy the summary.
Excluding non-finite values yields a mean over the views that have a finite value —
a defensible choice, though worth stating in the report where it applies.

```python
    return {
        "n_views": len(common), "per_view": per_view,
        "failed_views": failed, "summary": summary,
        "dir_a": dir_a, "dir_b": dir_b,
    }
```

The returned dict records the inputs alongside the outputs, so a saved metrics file
is self-describing about which directories it compared.

### Lines 94–98 — `save_metrics`

```python
def save_metrics(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
```

Writes results as indented JSON — human-readable in a diff and machine-readable for
figure generation. `exist_ok=True` makes directory creation idempotent.

## Why it matters

Two dissertation claims rest here.

**"Metric code is validated."** `test_metrics.py` checks both directions: identical
images give `inf` PSNR and SSIM 1.0; a heavily perturbed image scores worse than a
lightly perturbed one. Together these establish the metrics respond correctly in
magnitude and direction — the brief's explicit requirement.

**"No cherry-picking."** Sorted deterministic ordering, filename-based pairing, all
common views included, per-view results preserved, and failures recorded rather than
dropped. Every one of those is a structural guarantee rather than a promise about
process.
