# Stage 2 verification — SH extraction and exact reinsertion

Date: 2026-08-02. Commit: a9feb6b.

## Layout determination

See REQUIREMENTS.md and `src/sh_compression/sh_layout.py`; exact index mapping is
unit-tested (`test_sh_layout.py::test_ply_field_mapping_no_permutation`,
`test_ply_roundtrip.py::test_extract_matches_ply_fields`).

## Coefficient-level round trip (real model, 1,071,462 points)

Run both locally (macOS) and in Colab; identical results:

    max_abs_coeff_diff: 0.0
    non_sh_fields_identical: true

(results/stages/stage_2_roundtrip_metrics.json)

## Render-level round trip (Colab, T4)

original PLY → extract SH → reinsert unchanged → save → render 38 test views via
scripts/render_model.py (staged model dir, unmodified render.py):

    38 of 38 views pixel-identical (PSNR = inf on every view)
    min SSIM: 1.0
    failed views: 0

Gate target was PSNR > 50 dB; achieved bit-exact output. STAGE 2 PASSED.
