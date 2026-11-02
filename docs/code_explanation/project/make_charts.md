# `scripts/figures/make_charts.py`

## Purpose

Draws the two comparison charts of Chapter 5 into `results/figures/thesis/`:

- **Figure 5.2** — the same-checkpoint comparison: SH size against SH-colour PSNR for the
  thesis configurations and the peer-style SH treatments;
- **Figure 5.3** — where the thesis sits among published 3DGS compression methods on Tanks
  and Temples (whole-model ratio against ΔPSNR).

## Line-by-line

### Lines 1–23 — Header, imports, paths

The docstring names the inputs: `results/experiments/image_metrics/`,
`results/experiments/sh_baselines/` (written by `scripts/compare_sh_baselines.py`), and
the published values typed into the script. `REPO`, `RES` and `OUT` are resolved from the
file's location, and the output folder is created if missing.

### Lines 25–33 — Style

The shared palette and rcParams also used by `make_analysis_figures.py`.

### Lines 35–49 — Published values

```python
TT_3DGS = dict(psnr=23.14, ssim=0.841, lpips=0.183, mb=411.0)
TT = [("HAC (low)", 24.04, 0.846, 0.187, 8.5), ...]
BASE_GT_PSNR = 24.9298
```

`TT_3DGS` is the reference 3DGS on Tanks and Temples (average of *train* and *truck*,
held-out views) and `TT` holds 16 published methods as (name, PSNR, SSIM, LPIPS, MB).
These are author-reported values as collected by the compression survey of Bagdasarian
et al. (2025); each method's ratio is `411 / MB` and its ΔPSNR is `PSNR − 23.14`, which
reproduces thesis Table 5.10. `BASE_GT_PSNR` is the thesis baseline against GT, taken from
the rank-48 control, which reproduces the uncompressed model.

### Lines 52–57 — `save`

PNG and SVG at 300 dpi with a white background and tight bounding box.

### Lines 60–108 — Figure 5.2 (`same_checkpoint`)

Reads `sh_baselines.json` and plots:

- the thesis 8-bit scalar configurations r = 4 … 48 as a blue line, filled where the
  configuration was also rendered on the GPU (r = 8, 16) and hollow where only the proxy
  exists;
- thesis VQ at r = 8, K = 64 and 256 as blue diamonds;
- the peer-style treatments as orange squares.

Every point is labelled from a fixed offset table (lines 79–91). The footnote is built
from the file itself, including the Pearson coefficient of the proxy validation, so the
figure cannot disagree with the data it shows.

### Lines 111–156 — Figure 5.3 (`published`)

Plots the 16 published methods (orange squares) and four thesis configurations (blue
circles: scalar r = 16, low-rank r = 8 float16, VQ r = 8 K = 256, low-rank r = 16
float32). A thesis point is its measured `whole_model_ratio` against `gt_psnr − 24.9298`.
The dashed vertical line is the SH-only ceiling of 4.43×. The footnote states the two
protocol differences that limit the comparison: published Δ values are on held-out views
against each paper's own baseline, while the thesis Δ is on views seen in training.

### Lines 159–162 — Entry point

Figure 5.3 always; Figure 5.2 only when `sh_baselines.json` exists.

## Running it

    python scripts/figures/make_charts.py

## Why it matters

Figure 5.2 is the visual form of the thesis's strongest comparative claim, and Figure 5.3
places the thesis honestly among methods that retrain and compress geometry. Both figures
are drawn from committed files plus a published-value table that is visible in the
source, so a reader can check every point.
