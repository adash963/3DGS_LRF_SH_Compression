# Stage 0 verification report — requirements and baseline selection

Date: 2026-08-02

## Decisions

| Item | Decision |
|---|---|
| Baseline repo | this repository (`adash963/3DGS`, fork of official `graphdeco-inria/gaussian-splatting`), branch `dissertation` |
| Base commit | 54c035f (upstream main) |
| Licence | Gaussian-Splatting License (research use) — LICENSE.md |
| Dev scene | `train` (Tanks&Temples), pretrained iteration-30000 checkpoint from camenduru's `GaussianViewTest.zip` HF mirror |
| Dataset | `tandt_db.zip` (official 650 MB mirror on HF) |
| Validation scene | `truck` (optional, after full pipeline works) |
| GPU | Colab T4 (16 GB) sufficient; rasteriser needs CC 7.0+ |
| Disk | ~2 GB (model subset + dataset scene) |

## SH layout determination (blocking requirement — PASSED)

Verified by direct code inspection of `scene/gaussian_model.py`:
- `save_ply` (L239-253): fields `f_dc_0..2` then `f_rest_0..44`, written as
  `transpose(1,2).flatten` → channel-major.
- `load_ply` (L281-309): `f_dc_{c}` → channel c DC; `f_rest` reshaped `(N, 3, 15)` →
  `f_rest_{c*15+j}` = channel c, coefficient j+1. Degree-3 assert (45 fields).
- Canonical form adopted: `[N, 3, 16]`, flat `[N, 48]`, column `c*16+j`.
- Encoded in `src/sh_compression/sh_layout.py`; unit-tested for exact index mapping
  (test_sh_layout.py::test_ply_field_mapping_no_permutation).

## Consistency checks

- Planned factorisation `X[N,48] ≈ U[N,r] V[r,48]` matches the verified layout. ✓
- No custom CUDA kernel required: reconstruction happens offline, rendering uses the
  unmodified `diff-gaussian-rasterization`. ✓
- Install/render commands documented in `notebooks/colab_runner.ipynb` (Run-all). ✓

## Evidence

- 46/46 unit tests pass locally (`pytest tests/unit -q`): layout, PLY round trip,
  synthetic SVD (Stage-3 math pre-verified), quantisation, serialisation, metric sanity.
- Full script pipeline smoke-tested on a 5000-point synthetic PLY:
  inspect → analyse_rank → compress (VQ r8/K64: 8,220 B asset, reported size ==
  file size) → reconstruct → 22-config ablation matrix, 22/22 ok.
