# `scripts/compare_sh_baselines.py`

## Purpose

Applies the SH treatments that published 3DGS compression methods rely on to the **same**
checkpoint as the thesis, stores each one in the same zlib `.npz` container, and scores
every result on the CPU. It writes `results/experiments/sh_baselines/`, which is the source
of thesis Tables 5.4, 5.7 and 5.8, the proxy column of Table 5.6, Figure 5.2 (drawn by
`scripts/figures/make_charts.py`) and the proxy validation quoted in Section 3.5
(Pearson 0.993, Spearman 0.991 over 11 rendered configurations).

It is step 11 of thesis Table 3.7 and the command in Appendix B.7.

## Line-by-line

### Lines 1–22 — Docstring

States the two quality measures (coefficient error, split into the DC term and the
higher-order bands, and the view-dependent colour proxy) and lists the baselines. The
closing remark matters for interpretation: the published methods fine-tune after applying
these treatments, so the rows produced here are **lower bounds** on those methods, not
re-implementations of them.

### Lines 23–39 — Imports

```python
from sh_compression.extraction import compress, extract_flat_sh
from sh_compression.quantization import kmeans_vq, scalar_quantize
from sh_compression.serialization import CompressedSH
from sh_compression.storage import ratios
```

The baselines reuse the thesis package: the same scalar quantiser, the same K-means, the
same byte accounting. Any difference between a thesis row and a baseline row therefore
comes from the treatment (factorise first or not, drop bands or not), not from a different
implementation of quantisation.

### Lines 41–51 — SH constants and band columns

`C0`–`C3` are copied from `utils/sh_utils.py`, the constants the rasteriser uses.
`DC_COLS = [0, 16, 32]` follows the flat layout of `sh_layout.py`
(`X[:, 16c + j]`, coefficient 0 of each channel), and `REST_COLS` holds the other 45
columns. These two column sets give the DC and higher-order errors of Table 5.7.

### Lines 54–65 — `sh_basis`

Evaluates the 16 degree-3 basis functions for a batch of unit view directions, with the
rasteriser's sign convention (`-C1 * y`, `C1 * z`, `-C1 * x`, …). Returns `[n, 16]`.

### Lines 68–72 — `test_views`

```python
cams = sorted(json.load(f), key=lambda c: c["img_name"])
return [c for i, c in enumerate(cams) if i % llffhold == 0]
```

The same rule as `scene/dataset_readers.py`: sort by image name and keep every eighth
camera. On the 301 cameras of T&T *train* this gives the 38 test views used everywhere
else in the thesis.

### Lines 75–107 — `ColourProxy`

**`__init__` (lines 78–92)** does the expensive work once:

- opacity weights are the sigmoid of the stored logits;
- `(xyz - pos) @ rot` moves the Gaussian centres into the camera frame (`cameras.json`
  stores the camera-to-world rotation, so multiplying on the right applies its
  transpose);
- a Gaussian counts as visible if it is in front of the 0.2 near plane used by the
  rasteriser and inside the field of view (`|x/z| < W / 2fx`, `|y/z| < H / 2fy`);
- for the visible Gaussians it stores their indices, the SH basis of their view direction
  (Gaussian minus camera position, normalised) and their weights.

The 37 rows then reuse these per-view arrays instead of recomputing them.

**`_rgb` (lines 94–98)** turns coefficients into colour exactly as the renderer's SH
evaluation does: basis · coefficients per channel, plus 0.5.

**`psnr` (lines 100–107)** computes the opacity-weighted mean squared colour error over
all views and channels and returns it as PSNR (infinite when nothing changed).

Two simplifications make this a *proxy*. It ignores alpha blending, occlusion and screen
footprint, so a Gaussian hidden behind others counts as much as a visible one of the same
opacity. And it clips every colour to [0, 1], whereas the renderer clamps each Gaussian's
colour only at 0 (`gaussian_renderer/__init__.py`, line 80) and clips the final image
(line 119). Both are why the script validates the proxy against the rendered PSNR before
anyone relies on it (lines 214–226).

### Lines 110–120 — `npz_bytes` and `band_mask`

`npz_bytes` writes the given arrays with `np.savez_compressed` into a temporary folder and
returns the file size, so a baseline's rate is measured in the same container as the
thesis assets. (Baseline files carry no `meta_json` entry; the difference is a few hundred
bytes.) `band_mask` keeps the columns whose coefficient index is below `(deg + 1)²`.

### Lines 123–151 — `baseline_methods`

Yields one tuple per peer-style treatment:

| Name | Treatment | Stands for |
|---|---|---|
| `fp16_raw` | all 48 coefficients as float16 | EAGLES-style storage |
| `q8_raw` | per-coefficient 8-bit uniform quantisation | 8-bit post-processing |
| `deg2`, `deg1`, `deg0` | keep SH degree ≤ L (float32), drop the rest | LightGaussian / Reduced 3DGS band reduction |
| `deg1_q8`, `deg0_q8` | band drop, then 8-bit | the same, quantised |
| `rawvq_k64`, `rawvq_k256` | K-means of the raw 48-D vectors, same `kmeans_vq` and seed | CompGS / Compressed 3DGS |

For the band drops only the kept columns are stored; the reconstruction puts zeros in the
dropped ones. The K-means timing printed here is the 397 s / 28 s quoted in Section 5.4.5
(see `sh_baselines_run.log`).

### Lines 154–165 — Arguments

`--ply` and `--cameras` are required. `--extra-scalar` adds thesis scalar configurations
that were never rendered (`4:8 12:8 24:8 32:8 48:8 16:16`, the rows of Table 5.4).
`--vq-sizes` and `--seed` control the raw-SH K-means.

### Lines 167–178 — Loading once

Reads the model and its SH matrix, the PLY size (for whole-model ratios), the Gaussian
centres and opacities for the proxy, and every committed rendered metric file, keyed by
configuration name.

### Lines 180–196 — `row`

Builds one CSV row: file size and bytes per Gaussian, the SH and whole-model ratios
(`storage.ratios`), the relative coefficient error overall and for the DC and
higher-order columns separately, the proxy PSNR, and the rendered metrics when a rendered
file exists for that name.

### Lines 198–212 — The 37 rows

1. The 22 thesis assets in `--ablation-dir`. These are the `asset.npz` files written by
   `scripts/run_ablation.py`, which are not committed, so run the ablation first.
2. The 6 extra scalar configurations, compressed in memory and sized through a temporary
   save.
3. The 9 baselines.

### Lines 214–226 — Proxy validation

Pairs the proxy with the rendered PSNR against the uncompressed render for every
configuration that has one, excluding the full-rank control (101.14 dB, kept out by the
`< 99` test). That leaves 11 pairs. Pearson is `np.corrcoef`; Spearman is the same on the
ranks. The pairs are saved too: the thesis's "estimated rendered PSNR" columns are a
linear fit to them (rendered ≈ 0.965 × proxy + 2.31), computed outside this script.

### Lines 228–236 — Output

`sh_baselines.csv` (one row per configuration) and `sh_baselines.json` (the same rows plus
`n_points`, the PLY size, the number of test views and the validation block).

## Running it

    python scripts/run_ablation.py --config configs/quantization_ablation.yaml --ply $PLY
    python scripts/compare_sh_baselines.py --ply $PLY --cameras $MODEL/cameras.json

`$PLY` and `$MODEL` come from the pretrained checkpoint (see `data/README.md`). On the
Apple-silicon Mac of the thesis the comparison took about ten minutes, most of it in the
raw-SH K-means. The notebook's `STAGE = 'baselines'` runs both commands on Colab.

## Why it matters

This is the thesis's strongest comparative result: between about 1 MB and 30 MB of SH,
no training-free peer treatment is both smaller and better than a thesis configuration
(Section 5.4.5). The comparison is fair in the way that matters here (same checkpoint,
same container, same quantisers), and honest about its limits: the baselines are not
fine-tuned, and the proxy is validated rather than assumed.
