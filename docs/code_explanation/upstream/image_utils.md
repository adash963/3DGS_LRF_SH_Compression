# `utils/image_utils.py`

## Purpose

MSE and PSNR in PyTorch, used by upstream's training loop and `metrics.py`. Only
8 lines of code, but worth documenting because it is **a second, independent PSNR
implementation** alongside our own — and comparing them is a useful cross-check.

## Line-by-line

### Lines 14–15 — `mse`

```python
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
```

Mean squared error, computed **per batch element**. `.view(img1.shape[0], -1)`
flattens everything except the first dimension, then `.mean(1)` averages within each
element, returning one value per image rather than a single scalar.

`keepdim=True` preserves the dimension so the result broadcasts cleanly in later
arithmetic.

### Lines 17–19 — `psnr`

```python
def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
```

PSNR with MAX = 1.0, assuming images in `[0, 1]`.

The form differs from ours but is algebraically identical:

- Upstream: `20 · log₁₀(1/√MSE)`
- Ours: `10 · log₁₀(1/MSE)`

Since `20·log₁₀(1/√x) = 20·log₁₀(x^-1/2) = -10·log₁₀(x) = 10·log₁₀(1/x)`, the two
agree exactly.

## Differences from our implementation

Our `sh_compression/metrics.py` differs in three deliberate ways:

| | Upstream | Ours |
|---|---|---|
| Backend | PyTorch (GPU) | NumPy (CPU) |
| Accumulation | float32 | **float64** |
| MSE = 0 | `inf` via division | explicit `inf` branch |

**The float64 accumulation matters** for this project. Averaging millions of squared
differences in float32 loses precision precisely in the high-PSNR regime — which is
where our most important measurements sit (34 dB for the headline configuration,
101 dB for the full-rank anchor).

**The explicit zero branch matters too.** Upstream's expression computes
`1/√0 = inf` and `log₁₀(inf) = inf`, which happens to give the right answer but relies
on IEEE division-by-zero semantics. Ours tests `mse == 0` and returns `float("inf")`
directly, which is unambiguous — and this is the path that produced the Stage 2
pixel-identical result.

## Why it matters

Two points for the report.

**Metric choice was deliberate, not accidental.** We did not simply reuse upstream's
PSNR; we implemented it in NumPy with higher-precision accumulation because our
measurements happen where float32 accumulation is weakest.

**The implementations agree.** Since the formulas are algebraically equivalent and the
only differences are precision-related, our PSNR values are directly comparable to any
number produced by upstream's `metrics.py` — useful if a reader wants to cross-check
against the 3DGS paper's own evaluation tooling.
