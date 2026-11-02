# Stage 1 verification — reproducible environment and baseline render

Date: 2026-08-02. Commit under test: c705152 (fixes included after two failures,
see DEVELOPMENT_LOG.md).

## Environment (Colab, captured from notebook cell 1)

| item | value |
|---|---|
| Python | 3.12.13 |
| OS | Linux-6.6.122+-x86_64-with-glibc2.35 |
| GPU | Tesla T4, 14.6 GB VRAM |
| Driver | 580.82.07 |
| CUDA (torch) | 12.8 |
| PyTorch | 2.11.0+cu128 |

## Checks

- CUDA detected: yes (assert in notebook).
- `diff_gaussian_rasterization` + `simple_knn` compiled and imported.
- 46/46 CPU unit tests pass in Colab.
- Model loads: 1,071,462 Gaussians, 265,724,108-byte PLY, SH = 77.4% of file bytes.
- Baseline render: 38/38 test views (eval=True split of 301 cameras), ~2.14 it/s,
  outputs finite, correct dimensions; ground-truth copies written alongside.

## Failures encountered and fixed (evidence in DEVELOPMENT_LOG.md)

1. Windows-authored cfg_args crashed `eval()` → scripts/fix_cfg_args.py.
2. Missing `depths`/`train_test_exp` fields (sentinel parser) → complete field set.

## Reproducibility note

Cells 1-6 of `notebooks/colab_runner.ipynb` executed on a fresh runtime; cell 4 was
amended mid-session to call fix_cfg_args.py. A full untouched Run-all verification on
a brand-new runtime is scheduled as part of the Stage 8 final reproduction (single
verification covers both gates); until then this is recorded as a known risk in
STAGE_STATUS.yaml, not claimed as passed.
