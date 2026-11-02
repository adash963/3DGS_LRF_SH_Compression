# `utils/loss_utils.py`

## Purpose

Training losses: L1, L2, and SSIM (both a pure-PyTorch implementation and a fused CUDA
one). Not used by this project — we never train — but the SSIM implementation is worth
documenting because it is **a second SSIM alongside the scikit-image one our metrics
use**, and the report should be clear about which produced its numbers.

## Line-by-line

### Lines 16–22 — Fused SSIM import and constants

```python
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2
```

`C1` and `C2` are the standard SSIM stabilisation constants from Wang et al. (2004),
defined as `(K₁L)²` and `(K₂L)²` with `K₁=0.01`, `K₂=0.03`, and data range `L=1`.

**These are the same constants scikit-image uses by default**, which is why our SSIM
values are directly comparable to upstream's despite different implementations.

### Lines 24–38 — `FusedSSIMMap` (training only)

```python
class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ...
    @staticmethod
    def backward(ctx, opt_grad):
        ...
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None
```

A custom autograd function wrapping the CUDA fused-SSIM kernel, with a hand-written
backward pass. This is the training-speed optimisation added in the October 2024
update.

Note the `backward` returns gradients only for `img1` (the render), not `img2` (the
ground truth) — correct, since ground truth is constant.

### Lines 40–44 — L1 and L2

```python
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()
```

3DGS trains with a weighted combination of L1 and SSIM
(`lambda_dssim = 0.2` in `OptimizationParams`), so 80% L1 and 20% SSIM-based.

Worth noting for the report's discussion: the model we compress was optimised against
*these* objectives. It was never optimised for LPIPS, which is part of why LPIPS
values sit where they do even for the baseline (0.142 against ground truth).

### Lines 46–54 — Gaussian window construction

```python
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
```

Builds the Gaussian weighting window SSIM uses for local statistics.

Line 48 normalises the 1-D window to sum to 1. Line 52 forms the 2-D window as the
outer product of two 1-D windows — valid because a 2-D Gaussian is separable.

`sigma=1.5` with an 11×11 window are the parameters from the original SSIM paper, and
**the same defaults scikit-image uses** — another reason the two implementations agree.

### Lines 56–86 — `ssim` and `_ssim`

```python
def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
```

```python
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, ...) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, ...) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, ...) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
```

The textbook SSIM computation, expressed as convolutions:

- Lines 67–68: local means, via Gaussian-weighted convolution
- Lines 74–76: local variances and covariance, using the identity
  `Var(X) = E[X²] − E[X]²`
- Line 81: the SSIM formula, combining luminance, contrast and structure terms
- Line 84: mean over the whole map

`groups=channel` makes the convolution depthwise, so channels are processed
independently rather than mixed.

### Lines 89–91 — `fast_ssim`

```python
def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
```

The CUDA-accelerated entry point used in training.

## Which SSIM produced our numbers

**Neither of these.** Our `sh_compression/metrics.py` calls
`skimage.metrics.structural_similarity` with `channel_axis=-1, data_range=1.0`.

That was a deliberate choice: scikit-image's is a widely-used reference
implementation, so our numbers are comparable to the broader literature rather than to
this codebase alone. Reimplementing SSIM — or reusing a training-oriented one — risks
subtle parameter differences that would make cross-paper comparison meaningless.

The implementations should agree closely, since they share the standard constants
(`0.01²`, `0.03²`) and window parameters (11×11, σ=1.5). Any residual difference comes
from edge handling and averaging details, not from the definition.

For precision, the report should state which implementation produced its SSIM figures
— scikit-image — rather than leaving it ambiguous.

## Why it matters

Two points:

1. **Provenance of our SSIM numbers.** Documenting that they come from scikit-image,
   not from this file, removes an ambiguity a careful marker could raise.
2. **Context for the LPIPS results.** The model was trained on L1 + SSIM only. It was
   never optimised for perceptual similarity, which is useful background when
   discussing why LPIPS behaves as it does across the compression sweep.
