# Stage 6 verification — automated rendering evaluation

Date: 2026-08-02. Driver: scripts/run_render_eval.py (single command per batch).

## Protocol conformance

- Identical cameras (test split, eval=True, every 8th of 301), identical resolution,
  identical unmodified renderer, identical metric code for every configuration. ✓
- Both comparisons recorded per view: vs dataset ground truth AND vs baseline
  renders (isolates compression effect from model fit). ✓
- Per-view + mean metrics saved: results/experiments/image_metrics/metrics_<name>.json
  (12 configs × 38 views × 2 references). ✓
- Failed views recorded: none occurred (failed_views: [] in all files). ✓
- No cherry-picking: all views of the split, deterministic sorted order. ✓
- Raw renders preserved in Colab session at /content/work/eval/renders/<name>/
  (persistence to Drive pending user auth; noted as risk). ✓/risk
- Metric sanity: unit tests (identical→perfect, perturbed→worse); empirical anchor:
  round-trip model scored PSNR=inf/SSIM=1.0, full-rank scored 101 dB. ✓
- Timing recorded separately per phase (compress / reconstruct / render_total). ✓
- LPIPS: deferred, metrics code is CUDA-ready; to be run for final configs or
  omission documented. (pending)

## Regenerability

Figures and tables are produced only by scripts/generate_report_assets.py from
all_results.json + image_metrics/*.json:
rate_distortion_psnr/ssim/vsbase_psnr.png, psnr_vs_rank.png, coeff_error_vs_rank.png,
storage_breakdown.png, results_table.md. STAGE 6 PASSED (LPIPS item open).
