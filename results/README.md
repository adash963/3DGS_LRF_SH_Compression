# results/

Everything the thesis quotes comes from the files in this folder. This page lists what is
committed, and what the scripts write but is deliberately left out of git (large binaries
and renders), so the file names in the thesis can be matched to the repository.

## Committed

| Path | Written by | Used in the thesis for |
|---|---|---|
| `experiments/rank_analysis/singular_values.csv`, `energy_vs_rank.csv`, `error_vs_rank.csv` | `scripts/analyse_rank.py` | SH spectrum and effective rank (Chapter 4, RQ1) |
| `experiments/ablation/<name>/config.yaml`, `metrics.json`, `git_commit.txt` | `scripts/run_ablation.py` | Storage, ratios and coefficient error for all 22 configurations |
| `experiments/ablation/all_results.csv`, `all_results.json` | `scripts/run_ablation.py` | Table of the 22 configurations |
| `experiments/image_metrics/metrics_<name>.json` (12 files) | `scripts/run_render_eval.py` | PSNR, SSIM, LPIPS vs GT and vs the uncompressed render; `timing_s` (RQ4) |
| `experiments/sh_baselines/sh_baselines.csv`, `sh_baselines.json`, `sh_baselines_run.log` | `scripts/compare_sh_baselines.py` | Same-checkpoint comparison with peer-style SH treatments, the SH-colour proxy and its validation against the rendered PSNR, DC vs higher-order error (Section 3.5, Tables 5.4, 5.6, 5.7, 5.8) |
| `experiments/trained_heldout/summary.json` | Colab notebook, cell 7 | Held-out check on a newly trained checkpoint (Section 3.6.2, Table 3.7 step 13) |
| `figures/thesis/fig*.png`, `fig*.svg` | `scripts/figures/make_diagrams.py`, `make_analysis_figures.py`, `make_charts.py` | The thesis figures (3.1–3.3, 4.1–4.3, 5.1–5.3), named by figure number |
| `figures/results_table.md` | `scripts/generate_report_assets.py` | Results table of the 22 configurations (Table 3.7 step 10) |
| `figures/visual_comparisons.png` | Crops exported from the Colab renders by `scripts/export_crops.py` | Figure B.5 |
| `stages/stage_1_baseline.md`, `stage_1_environment.json`, `stage_2_layout.md`, `stage_2_roundtrip_metrics.json` | Colab sessions, August 2026 | Baseline model and render (Tables 3.1, 4.2, 5.13), GPU environment (Table 3.8), exact SH round trip and pixel-identical renders (Section 3.2.2) |
| `final_reproduction/reproduction_report.md` | Colab notebook | Clean-runtime reproduction (August 2026) |

## Thesis tables and figures

| Thesis | File | Written by |
|---|---|---|
| Tables 3.1 and 4.2 (input model) | `stages/stage_1_baseline.md` | `scripts/inspect_model.py` |
| Section 3.2.2 (SH round trip; 38/38 renders pixel-identical) | `stages/stage_2_roundtrip_metrics.json`, `stages/stage_2_layout.md` | `extraction.roundtrip_unchanged` (notebook cell 6) |
| Table 3.4 (experiment grid) | `../configs/quantization_ablation.yaml`, `../configs/rank_ablation.yaml` | – |
| Table 3.8 (GPU environment) | `stages/stage_1_environment.json` | notebook cell 1 |
| Section 3.6.2 (held-out check) | `experiments/trained_heldout/summary.json` | notebook cell 7 |
| Appendix B.2 (archive sizes and SHA-256) | `../data/README.md` | checked by notebook cell 4 |
| Table 4.5, Figure 4.1 | `experiments/rank_analysis/` | `scripts/analyse_rank.py`; figure by `scripts/figures/make_analysis_figures.py` |
| Table 4.6, Figures 4.2 and 4.3 | analytical, no data file | `scripts/figures/make_analysis_figures.py` |
| Tables 5.1–5.3 and 5.5, rendered columns of 5.6, Tables 5.9 and 5.13, Figure B.3 | `experiments/image_metrics/`, `experiments/ablation/` | `scripts/run_render_eval.py`, `scripts/run_ablation.py` |
| Figure 5.1 | `experiments/image_metrics/` | `scripts/figures/make_analysis_figures.py` |
| Tables 5.4, 5.7 and 5.8, proxy column of 5.6, Figure 5.2 | `experiments/sh_baselines/` | `scripts/compare_sh_baselines.py`; figure by `scripts/figures/make_charts.py` |
| Tables 5.10–5.12, Figure 5.3 | published values in `scripts/figures/make_charts.py`; thesis rows from `experiments/image_metrics/` | `scripts/figures/make_charts.py` |
| Figures 3.1–3.3 | – | `scripts/figures/make_diagrams.py` |
| Figure B.1 | printed only; the values are in `stages/stage_1_baseline.md` and `experiments/rank_analysis/` | `scripts/inspect_model.py`, `scripts/analyse_rank.py` |
| Figure B.2 | printed only; the same configuration is `experiments/ablation/sq_r16_b8/metrics.json` | `scripts/compress_model.py` |
| Figure B.5 | `figures/visual_comparisons.png` | `scripts/export_crops.py` |
| Baseline render time in Table 5.13 | `stages/stage_1_baseline.md` | Colab session |

Some numbers are derived rather than written by a script:

- The per-view spread of the operating point in Section 5.4.1 (mean loss 0.62 dB,
  standard deviation 0.56 dB, 95 % confidence interval ±0.19 dB, range 0.10–2.99 dB over
  the 38 views) is the difference between `per_view_vs_gt` in
  `experiments/image_metrics/metrics_lr_r48_fp32.json` (the control, equal to the
  uncompressed model) and in `metrics_sq_r16_b8.json`; the interval uses the t
  distribution with 37 degrees of freedom.

- The "estimated rendered PSNR" in Tables 5.4 and 5.8 is the linear fit of the 11 pairs in
  `sh_baselines.json` → `proxy_validation.pairs_proxy_vs_rendered`
  (rendered ≈ 0.965 × proxy + 2.31, residual RMS 0.52 dB).
- The K-means times of the thesis VQ configurations on the Mac in Section 5.4.5 (121 s and
  9 s) are not recorded in any file; the raw-SH times (397 s and 28 s) are in
  `sh_baselines_run.log`.

## Written when you run the scripts, not committed

| Path in the thesis | Why it is not in git |
|---|---|
| `experiments/ablation/<name>/asset.npz` | The compressed containers are up to 194 MB each (`*.npz` is in `.gitignore`). Re-create them with `scripts/run_ablation.py`; their sizes are recorded in each `metrics.json` (`asset_file_bytes`). `scripts/compare_sh_baselines.py` reads them, so run the ablation first. |
| `results/assets/<name>.npz`, `results/variant.ply`, `results/staging/`, `results/renders/<name>/`, `results/experiments/renders/` | Outputs of the single-configuration commands in Appendix B: compressed files, decoded PLYs and rendered images. Too large to commit, and rebuilt in seconds to minutes. |
| `figures/*.png` other than `visual_comparisons.png` | Working plots written by `scripts/analyse_rank.py` and `scripts/generate_report_assets.py` (Table 3.7 step 10, Appendix B.3). They duplicate the thesis figures in `figures/thesis/`, so git ignores them. |
| `GaussianViewTest.zip`, `tandt_db.zip`, `<model_dir>/…` | External downloads (pretrained checkpoint and Tanks and Temples images); sizes and SHA-256 in `../data/README.md`, credits in the README. |
