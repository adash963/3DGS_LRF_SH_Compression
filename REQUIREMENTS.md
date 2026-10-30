# Requirements

## Research question

Can scene-level low-rank factorisation, followed by vector quantisation, reduce the
storage required by 3D Gaussian Splatting (3DGS) spherical-harmonic (SH) coefficients
while retaining acceptable rendered visual quality?

## Mathematical formulation (frozen)

For a trained 3DGS scene with N Gaussians and degree-3 SH:

- Each Gaussian stores 3 colour channels x 16 SH coefficients = 48 features.
- Scene-level SH matrix: `X in R^(N x 48)`, canonical tensor form `[N, 3, 16]`
  (channel-major: coefficient j of channel c sits at flat index `c*16 + j`).
- Factorisation: `X ~= U V` with `U in R^(N x r)` (per-Gaussian latents, singular
  values folded into U) and `V in R^(r x 48)` (shared basis), r = selected rank.
- Quantisation: K-means vector quantisation of the rows of U (codebook `C in R^(K x r)`,
  per-Gaussian index in `{0..K-1}`); optional scalar quantisation baseline.

This is a deliberate simplification of the broader proposal (no residual VQ, no
sensitivity-aware codebooks, no end-to-end training, no custom CUDA kernels). Those are
stretch goals and are not claimed as implemented unless verified.

## Baseline selection (Stage 0 decision)

- Repository: this repository — fork of the official implementation
  `graphdeco-inria/gaussian-splatting` (fork: `adash963/3DGS`), Colab-proven
  (camenduru template uses the same source).
- Licence: Gaussian-Splatting License (research use; see LICENSE.md).
- Renderer: unmodified `diff-gaussian-rasterization` CUDA rasteriser via `render.py`.
- SH storage layout (verified by code inspection, `scene/gaussian_model.py`):
  - in-memory: `_features_dc [N,1,3]`, `_features_rest [N,15,3]` (coeff-major, channel last)
  - on disk (PLY): `f_dc_0..2` then `f_rest_0..44`, channel-major
    (`f_rest_{c*15+j}` = channel c, higher-order coefficient j)
  - non-SH fields: x y z, nx ny nz, opacity, scale_0..2, rot_0..3
- Degree-3 SH confirmed: `max_sh_degree=3`, 45 rest fields (= 3*(16-1)).

## Scene selection

- Development scene: **train** (Tanks&Temples), pretrained checkpoint from the official
  release subset hosted at
  `https://huggingface.co/camenduru/gaussian-splatting/resolve/main/GaussianViewTest.zip`
  (contains a trained `train` model, iteration 30000) — small download, known to work in
  Colab with this exact codebase.
- Dataset (cameras + ground-truth images):
  `https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip`
  (T&T + DB COLMAP, 650 MB official mirror).
- Validation scene (optional, only after full pipeline works): **truck** (T&T).

## Environments

- Local (macOS, no CUDA): repository editing, unit tests, SH extraction, SVD,
  quantisation, serialisation, analysis, plotting. Python 3.12 venv at `../.venv`.
- Colab (NVIDIA GPU): rasteriser compilation, rendering, image metrics against renders.
  Pinned deps in `requirements-colab.txt`; notebook `notebooks/colab_runner.ipynb` is a
  thin execution shell — all logic lives in `src/` and `scripts/`.

## Mandatory deliverables

See SUCCESS_CRITERIA.md and EXPERIMENT_PLAN.md. Storage numbers are measured from real
files on disk; temporary reconstructed PLYs are never reported as the compressed size.
