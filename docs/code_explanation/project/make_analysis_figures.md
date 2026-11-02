# `scripts/figures/make_analysis_figures.py`

## Purpose

Draws thesis Figures 4.1, 4.2, 4.3 and 5.1 into `results/figures/thesis/` as 300-dpi PNG
and SVG. It reads only committed files, so it runs on any machine without the checkpoint:

| Figure | Content | Input |
|---|---|---|
| 4.1 | Singular values of X and cumulative explained energy E(r) | `results/experiments/rank_analysis/singular_values.csv` |
| 4.2 | Predicted SH-payload ratio against rank | storage model (Equation 3.10) |
| 4.3 | Predicted whole-model ratio and the 4.43× bound | storage model (Equation 3.11, Table 4.6) |
| 5.1 | PSNR, SSIM and LPIPS against GT versus compressed size | `results/experiments/image_metrics/metrics_<name>.json` |

## Line-by-line

### Lines 1–32 — Header, imports, paths

`REPO` is three directories up from the file, so the script works from any current
directory. It imports `effective_rank` and `FLAT_DIM` from the thesis package rather than
re-deriving them, so Figure 4.1 marks the effective rank with exactly the rule that
produced the "12" in the thesis (smallest r with E(r) ≥ 0.99).

### Lines 34–42 — Style

The same rcParams and palette as `make_charts.py` (Arial, light grid, no top/right
spines; blue, orange, aqua, violet), so all thesis figures read as one set.

### Lines 44–48 — Storage model constants

```python
BYTES_PER_GAUSSIAN = 62 * 4                     # 248
SH_BYTES = FLAT_DIM * 4                         # 192 (f_dc + f_rest)
NON_SH_BYTES = BYTES_PER_GAUSSIAN - SH_BYTES    # 56 (incl. the three zero-filled normals)
OPERATING_RANK = 16
```

The per-Gaussian byte counts of thesis Table 1.1. The 56 non-SH bytes include the three
normals that the reference exporter writes as zeros, which is why the bound is 4.43× and
not 5.6×. `OPERATING_RANK` is the smallest rank that meets the 1 dB tolerance (Section
5.3); Figures 4.1 and 5.1 highlight it.

### Lines 51–60 — `save` and `panel_title`

`save` writes PNG and SVG with a white background and a tight bounding box, then closes
the figure. `panel_title` gives every panel the same bold, left-aligned title.

### Lines 66–98 — Figure 4.1

Reads the 48 singular values, computes E(r) in percent as `cumsum(s²) / sum(s²)`, and
draws two panels:

- **(a)** the spectrum on a log axis, with a dashed line and leader at the effective rank;
- **(b)** E(r) with the 99 % threshold dashed and r = 1, 12 and 16 marked and labelled
  (90.1 %, 99.1 %, 99.6 %).

The label positions are given in data coordinates, so they stay put for this spectrum
but would need moving for a different scene.

### Lines 104–128 — Figure 4.2

Plots the asymptotic SH-payload ratio for r = 1 … 48 (N ≫ 48, fixed-size arrays and zlib
ignored):

- low-rank float32: `192 / (4r)` = 48 / r;
- low-rank float16, or 16-bit scalar codes: `192 / (2r)` = 96 / r;
- 8-bit scalar codes: `192 / r`;
- K-means VQ: 192× with a 1-byte index (K ≤ 256), 96× with a 2-byte index, both flat in r.

The legend sits in the empty band between the scalar curve and the VQ lines so it covers
neither.

### Lines 133–162 — Figure 4.3

`WHOLE_MODEL_CONFIGS` lists the eight bars in Table 4.6 order with their SH bytes per
Gaussian, and each bar is `248 / (56 + B_SH)`: 1.00, 2.07, 2.82, 3.44, 3.88, 4.13, 4.28
and 4.35×. The dashed line is `248 / 56` = 4.43×, the ratio if SH were removed entirely.
Bar values are printed inside the bars so that none of them crosses the bound line.

### Lines 167–221 — Figure 5.1

Loads the 12 rendered metric files and groups them into families by `mode` and
`factor_dtype`:

- low-rank float32 (filled blue circles, joined by a line);
- low-rank float16 (hollow blue circle);
- 8-bit scalar (orange squares);
- K-means VQ (aqua diamonds).

The x axis is the measured `.npz` size from the render run. The dashed reference is the
rank-48 float32 control (`lr_r48_fp32`), which reproduces the uncompressed model (thesis
Table 5.1: 24.93 dB / 0.8639 / 0.1418 against GT). The rank-16, 8-bit operating point is
labelled on the PSNR panel, and the legend runs along the bottom.

### Lines 224–228 — Entry point

Draws the four figures in thesis order.

## Running it

    python scripts/figures/make_analysis_figures.py

## Why it matters

These four figures had no generating script; the thesis versions were drawn outside the
repository. With this script every figure in Chapters 3–5 can be rebuilt from committed
files. The data and values match the thesis exactly; the layout differs in two places:
in Figure 5.1 the "uncompressed" and "rank 16, 8-bit" labels no longer overlap, and
Figures 4.2 and 4.3 use the same style as the other figures instead of the older serif
style.
