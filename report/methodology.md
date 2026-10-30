# Methodology (working notes → report section 8-9)

## Pipeline

1. Load pretrained 3DGS checkpoint (PLY, iteration 30000, T&T `train`).
2. Extract SH: PLY fields → canonical `[N,3,16]` → flat `X [N,48]` float32
   (layout verified by unit test against `scene/gaussian_model.py` semantics).
3. Truncated SVD (float64 accumulation): `X ≈ U V`, singular values folded into U.
4. Optional quantisation of U rows:
   - scalar: per-column uniform, 8/16-bit, stored codes + scale/offset;
   - K-means VQ: scipy kmeans2 (k-means++ init, fixed seed), empty-cluster repair,
     codebook fp32 `[K,r]` + per-Gaussian index (uint8/16).
5. Serialise versioned `.npz` (zip-compressed); measured file bytes are the
   compressed size. Component accounting: U/V/codebook/indices/metadata.
6. Reconstruct `X̂ = Û V`, reinsert into an otherwise untouched copy of the model,
   write temporary PLY, render with the **unmodified** CUDA rasteriser.
7. Evaluate renders: PSNR/SSIM (+LPIPS if available) against ground-truth eval
   images and against baseline renders; identical cameras/resolution/metric code.

## Design rationale

- Scene-level (global) basis V: 48-dim feature space bounds max rank at 48, making
  full SVD trivially cheap even for millions of Gaussians (N×48 matrix).
- Quantising U (not V): U carries N×r values and dominates storage; V is r×48 ≈
  negligible, so it stays fp32/fp16.
- Determinism: fixed seeds for k-means; SVD is deterministic up to sign.

## Deviations from the proposal

- No residual VQ, sensitivity-aware clustering, QAT, or CUDA-side decoding
  (documented stretch goals; see REQUIREMENTS.md).
