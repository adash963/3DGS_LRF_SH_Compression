# Experiment plan

## Pipeline

```
pretrained scene PLY
  → extract SH  X [N,48]                     (stage 2)
  → rank analysis (SVD)                      (stage 3)
  → truncated SVD  X ≈ U V, rank r           (stage 4)
  → optional quantisation of U (scalar / K-means VQ)   (stage 5)
  → serialise compressed asset (measured bytes)        (stage 4/5)
  → reconstruct SH → reinsert into model → temp PLY    (stage 4/5)
  → render with unmodified baseline renderer (Colab)   (stage 4/5/6)
  → PSNR / SSIM / (LPIPS) vs ground truth + vs baseline renders  (stage 6)
  → ablation matrix + rate-distortion figures          (stage 7)
```

## Experiment matrix (stage 7)

Low-rank only (fp32 factors): r ∈ {1, 2, 4, 6, 8, 12, 16, 48(full)}
Factor dtype ablation: r ∈ {4, 8, 16} × {fp32, fp16}
VQ (U quantised, V fp32): r ∈ {4, 8, 16} × K ∈ {64, 128, 256}
Scalar quantisation baseline: r ∈ {8, 16} × {8-bit}

Uninformative configurations may be pruned using stage-3/4 evidence; pruning decisions
are recorded in DEVELOPMENT_LOG.md.

## Metrics

- Compressed asset bytes on disk (component breakdown: U/V/codebook/indices/metadata)
- SH-only compression ratio = (N*48*4 bytes) / compressed SH bytes
- Whole-model ratio = baseline PLY bytes / (baseline non-SH bytes + compressed SH bytes)
- PSNR, SSIM per view + mean (identical cameras, resolution, renderer for all configs)
- LPIPS if reliably installable in Colab (else documented omission)
- Timing (reported separately): decode, reconstruct, render

## Evaluation views

The scene's test/eval split as produced by `render.py` (all views, no cherry-picking).
Ground truth = dataset images; additionally baseline-vs-compressed render comparisons.

## Reproducibility

Every experiment directory contains: config.yaml, metrics.json, environment.json,
git_commit.txt, stdout/stderr logs, the compressed asset, sample renders. Figures are
generated only from these machine-readable files by `scripts/generate_report_assets.py`.
