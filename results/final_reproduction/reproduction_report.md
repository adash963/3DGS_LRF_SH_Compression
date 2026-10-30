# Final clean-runtime reproduction — PASSED

Date: 2026-08-02, second fresh Colab session (runtime booted 10:37 UTC).
Repository state under test: commit 95697f7 (branch dissertation), cloned fresh by
the notebook itself. Procedure: Runtime → Disconnect and delete runtime → reload
notebook from GitHub → Run all. No source files or notebook cells edited.

## Steps executed and outcomes

1. Environment: Tesla T4 (14.6 GB), CUDA 12.8, torch 2.11.0+cu128, Python 3.12.13 ✓
2. Clone + submodules at branch tip (95697f7) ✓
3. Pinned installs + CUDA rasteriser compile + import check ✓
4. Data download + cfg_args fix (scripts/fix_cfg_args.py, automatic) ✓
5. 46/46 CPU unit tests pass ✓
6. Baseline render: 38 test views ✓
7. Bit-exact SH round trip asserted (max_abs_coeff_diff == 0.0) ✓
8. Headline configuration sq_r16_b8 reproduced end-to-end
   (compress → serialise → load → reconstruct → render → evaluate):

       gt_psnr      24.312173518667425
       gt_ssim       0.8584615572502738
       vsbase_psnr  34.63220971980244
       vsbase_ssim   0.9919354727393702

   Assertions passed: vsbase_psnr > 34.0, gt_psnr > 24.0, vsbase_ssim > 0.99,
   asset_file_bytes < 15,000,000.
9. Notebook printed: STAGE all PASS

## Determinism note

Reproduced metrics match the original session's values (27a3b14 evidence) to every
printed decimal — SVD and per-column scalar quantisation are fully deterministic;
no seed variance is involved in the headline configuration.

## Gate closures

- Stage 1's deferred fresh-runtime Run-all requirement: CLOSED.
- Stage 7's one-experiment clean reproduction rider: CLOSED.
- Stage 8 main-result reproduction requirement: CLOSED.
