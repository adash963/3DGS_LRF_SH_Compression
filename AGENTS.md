# Agent operating notes

This branch (`dissertation`) hosts an UpGrad academic project: compression of 3DGS
spherical-harmonic coefficients via scene-level low-rank factorisation + vector
quantisation. The upstream 3DGS code is the *baseline renderer* and must stay
unmodified except where a stage explicitly requires integration.

## Ground rules

- All implementation logic lives in `src/sh_compression/` and `scripts/`;
  `notebooks/colab_runner.ipynb` is a thin GPU execution shell only.
- Every stage: implement → test → fix → retest → save evidence under
  `results/stages/` → update `STAGE_STATUS.yaml` → commit.
- Never weaken thresholds to pass; never hide negative results; log failures in
  `DEVELOPMENT_LOG.md`.
- Storage numbers = measured bytes on disk of the `.npz` asset. Reconstructed PLYs
  are temporary render artefacts, never reported as compressed size.
- Local (macOS, venv at `../.venv`, no CUDA): tests, SVD, VQ, serialisation, analysis,
  figures. Colab (T4): rasteriser compile, rendering, image metrics.

## Key commands

    ../.venv/bin/python -m pytest tests/unit -q
    ../.venv/bin/python scripts/inspect_model.py --ply <ply>
    ../.venv/bin/python scripts/analyse_rank.py --ply <ply>
    ../.venv/bin/python scripts/compress_model.py --ply <ply> --rank 8 --mode vq --codebook-size 256 --out asset.npz
    ../.venv/bin/python scripts/reconstruct_model.py --asset asset.npz --baseline-ply <ply> --out-ply out.ply
    ../.venv/bin/python scripts/evaluate_model.py --dir-a <gt> --dir-b <renders> --out metrics.json
    ../.venv/bin/python scripts/run_ablation.py --config configs/quantization_ablation.yaml --ply <ply> --out-dir results/experiments/ablation
    ../.venv/bin/python scripts/generate_report_assets.py

## SH layout (do not change without updating sh_layout.py + tests)

PLY: `f_dc_{c}` (c=0..2), `f_rest_{c*15+j}` (channel-major). Canonical `[N,3,16]`,
flat `[N,48]`, column `c*16+j`. Verified against `scene/gaussian_model.py`.
