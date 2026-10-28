# Stage 4 verification — low-rank compressor

Date: 2026-08-02. Commits: dad9438 (code), eval on Colab T4 session 1.

## Gates

- Full-rank compressed round trip reproduces the baseline: lr_r48_fp32 renders at
  **101.14 dB PSNR / SSIM 1.0000 vs baseline renders** (float noise only). ✓
- Lower ranks render successfully: r ∈ {1,2,4,8,16} all 38/38 views, no NaNs. ✓
- Saved assets load in a clean process (run_render_eval loads via CompressedSH.load
  in a fresh subprocess-equivalent path; serialization unit tests cover version and
  corruption handling). ✓
- Reported size == actual file bytes (asserted in unit tests; measured values in
  results/experiments/ablation/*/metrics.json). ✓

## Key measured results (SH-only ratio; PSNR vs baseline renders / vs GT)

| config | bytes | ratio | vs-base PSNR | GT PSNR |
|---|---|---|---|---|
| lr_r1_fp32 | 3,949,292 | 52.1× | 19.86 | 18.38 |
| lr_r4_fp32 | 16,056,762 | 12.8× | 24.72 | 21.40 |
| lr_r8_fp32 | 32,148,149 | 6.4× | 28.01 | 22.76 |
| lr_r16_fp32 | 64,315,591 | 3.2× | 34.68 | 24.32 |
| lr_r48_fp32 | 194,284,344 | 1.06× | 101.14 | 24.93 (≈ baseline fit) |
| lr_r8_fp16 | 16,144,088 | 12.7× | 28.01 | 22.76 |

fp16 factors: identical quality to fp32 at half the size (PSNR differs < 0.001 dB).
Full per-view data: results/experiments/image_metrics/. STAGE 4 PASSED.
