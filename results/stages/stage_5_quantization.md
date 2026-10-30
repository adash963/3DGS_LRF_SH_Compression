# Stage 5 verification — quantisation

Date: 2026-08-02. Eval on Colab T4 session 1.

## Gates

- Every rendered codebook configuration decodes and renders 38/38 views. ✓
- No NaNs / invalid colours / field-ordering errors (renders finite; SSIM > 0.9 vs
  baseline for all quantised configs). ✓
- Byte accounting complete: codebook + indices + V + metadata per asset
  (component_bytes in metrics; file size is authoritative). ✓
- Determinism: fixed seed kmeans verified by unit test. ✓

## Key measured results

| config | bytes | SH ratio | vs-base PSNR | vs-base SSIM | GT PSNR |
|---|---|---|---|---|---|
| sq_r8_b8 | 7,099,397 | 29.0× | 28.00 | 0.9658 | 22.75 |
| sq_r16_b8 | 13,670,820 | 15.0× | **34.63** | **0.9919** | 24.31 |
| vq_r8_k64 | 778,508 | 264× | 23.26 | 0.9124 | 20.66 |
| vq_r8_k256 | 1,025,253 | 201× | 24.56 | 0.9266 | 21.33 |
| vq_r16_k256 | 1,031,711 | 199× | 24.92 | 0.9287 | 21.50 |

## Findings (honest)

- **Scalar 8-bit quantisation is essentially free**: sq_r16_b8 matches lr_r16_fp32
  quality (34.63 vs 34.68 dB) at 4.7× smaller storage — the best mid-range operating
  point (15× SH-only; whole model 265.7 MB → ~73.7 MB).
- **K-means VQ of U trades hard**: 200-264× SH compression but degradation is
  codebook-limited, not rank-limited (log2 K bits/Gaussian caps information; r16/k256
  ≈ r8/k256). Renders remain coherent (SSIM ≥ 0.91 vs baseline) with visible softening
  of view-dependent colour. STAGE 5 PASSED.
