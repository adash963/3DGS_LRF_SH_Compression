# Compressing 3D Gaussian Splatting Spherical-Harmonic Coefficients via Scene-Level Low-Rank Factorisation and Vector Quantisation

> Draft v1. Every experimental number in this document traces to a committed file
> under `results/` (referenced inline). Published results from other papers are
> explicitly labelled "reported by the authors" and were NOT reproduced here.

## 1. Abstract

3D Gaussian Splatting (3DGS) achieves real-time photorealistic novel-view synthesis
but produces large scene files, of which the majority is view-dependent colour
stored as degree-3 spherical-harmonic (SH) coefficients — 77.4% of the 265.7 MB
model studied here. We investigate whether scene-level low-rank factorisation of
the SH coefficient matrix, optionally followed by quantisation of the per-Gaussian
factor, can reduce this storage while retaining acceptable rendered quality. On a
pretrained Tanks & Temples *train* scene (1,071,462 Gaussians), the N×48 SH matrix
has an effective rank of 12 at the 99% energy criterion. Truncated SVD with 8-bit
scalar quantisation of the per-Gaussian factor compresses SH storage 15.0× (205.7 MB
→ 13.7 MB) while rendering at 34.6 dB PSNR / 0.992 SSIM / 0.018 LPIPS against the
uncompressed baseline's renders — a 0.62 dB reduction in ground-truth PSNR. K-means
vector quantisation reaches 201–264× SH compression with graceful degradation
(23.3–24.9 dB vs baseline), and analysis shows its quality is bounded by codebook
index entropy rather than factorisation rank. All results derive from measured
file sizes and automated per-view metrics, and the pipeline is reproducible from a
single Colab notebook on a free T4 GPU.

## 2. Background and motivation

3DGS [Kerbl et al. 2023] represents a scene as millions of anisotropic 3D Gaussians
optimised from posed photographs; each Gaussian stores position, rotation, scale,
opacity, and view-dependent colour as degree-3 SH coefficients (16 per colour
channel, 48 float32 values). Serialised scenes commonly reach hundreds of megabytes,
hindering distribution, streaming, and edge deployment. In the scene studied here
the SH block alone is 205,720,704 bytes — 77.4% of the file
(results/stages/stage_1_baseline.md). SH coefficients are therefore the single
highest-leverage compression target that leaves geometry untouched.

## 3. Literature review (summary; numbers are authors' reported results)

- **LightGaussian** [Fan et al. 2023] reports ~15× total model compression via
  pruning, SH distillation to lower degree, and vector quantisation.
- **Compact 3D Gaussians** [Lee et al. 2024] reports ~10–25× via learnable masking,
  hash-grid colour, and codebook quantisation of geometry.
- **HAC** [Chen et al. 2024] reports 75×+ via hash-grid-assisted context modelling
  with entropy coding, building on anchor-based **Scaffold-GS** [Lu et al. 2024].
- Common threads: (i) SH/colour is the dominant storage cost; (ii) quantisation with
  fine-tuning recovers most quality; (iii) entropy coding provides a further
  constant factor. These systems couple compression to retraining; none isolates
  the contribution of a purely post-hoc, scene-level linear factorisation of SH.

## 4. Research gap

Prior systems entangle many mechanisms (pruning, retraining, entropy coding),
making it hard to attribute gains. The simple question — *how compressible are a
trained scene's SH coefficients under a global linear basis, without any
retraining?* — lacks a controlled, measured answer.

## 5. Research question

Can scene-level low-rank factorisation, followed by vector quantisation, reduce the
storage required by 3DGS spherical-harmonic coefficients while retaining acceptable
rendered visual quality?

## 6. Aim and objectives

Aim: a technically correct, reproducible, honestly-evaluated post-hoc SH compression
pipeline with a controlled ablation. Objectives O1–O8 map to stages 1–8
(SUCCESS_CRITERIA.md): baseline reproduction, exact SH round trip, rank analysis,
serialisable low-rank compressor, quantisation, automated evaluation, ablation,
report + reproduction package.

## 7. Scope

In scope: one development scene (T&T *train*, pretrained 30k-iteration checkpoint),
offline compression/decompression, unmodified CUDA rasteriser, PSNR/SSIM/LPIPS
evaluation. Out of scope (documented stretch goals, not implemented): end-to-end or
quantisation-aware training, residual VQ, custom rasteriser decoding, entropy
coding, mobile deployment (REQUIREMENTS.md).

## 8. Methodology

Pipeline (EXPERIMENT_PLAN.md): extract SH from the checkpoint PLY into a canonical
matrix → truncated SVD → optional quantisation of the per-Gaussian factor →
versioned serialisation (measured bytes) → reconstruction → reinsertion into an
otherwise untouched model → rendering with the unmodified renderer → automated
per-view metrics against both ground-truth photos and baseline renders. Every
configuration uses identical cameras (the eval=True test split: every 8th of 301
cameras, 38 views), identical resolution, renderer, and metric code.

## 9. Mathematical formulation

X ∈ R^(N×48) (channel-major, DC first; layout verified against the reference
implementation and unit-tested). Truncated SVD X = UΣVᵀ ≈ (U_r Σ_r)(V_rᵀ) = U·V with
U ∈ R^(N×r) (singular values folded in), V ∈ R^(r×48). Quantisation of U:
(a) per-column uniform 8/16-bit scalar with stored scale/offset;
(b) K-means VQ (k-means++ init, fixed seed, empty-cluster repair): codebook
C ∈ R^(K×r), per-Gaussian index ∈ {0..K−1} at ⌈log2 K⌉ bits. Storage:
lowrank = N·r·s + 48·r·s; scalar = N·r + 2·48·r·4 + 48·r·s; VQ = N·⌈log2K⌉/8 +
K·r·4 + 48·r·4 (+ metadata; measured .npz bytes are authoritative).

## 10. System architecture

`src/sh_compression/` (layout, PLY I/O, factorisation, quantisation, versioned
serialisation, metrics, storage accounting) + CLI scripts + a thin Colab notebook.
50 automated tests (46 unit, 4 integration) pass on macOS/arm64 and Colab/x86_64.

## 11. Dataset and experimental setup

Scene: Tanks & Temples *train*; pretrained iteration-30000 checkpoint (camenduru
HF mirror of the official-code output); COLMAP dataset from the official T&T+DB
bundle. Environment: Colab T4 (16 GB), CUDA 12.8, PyTorch 2.11.0, Python 3.12.13
(results/stages/stage_1_environment.json). Caveat: the checkpoint was trained with
eval=False (all cameras), so our test views were seen in training — metrics measure
compression degradation relative to the baseline, not novel-view generalisation
(DEVELOPMENT_LOG.md).

## 12. Baseline

1,071,462 Gaussians; 265,724,108-byte PLY; 38/38 test views render at 2.14 it/s;
baseline fit to ground truth: 24.93 dB PSNR / 0.864 SSIM / 0.142 LPIPS (mean over
38 views; results/experiments/image_metrics/metrics_lr_r48_fp32.json — the r=48
configuration is numerically the baseline, rendering at 101.14 dB vs the baseline's
own renders).

## 13. SH extraction and verification

Extraction → unchanged reinsertion → save → render is **bit-exact**: max coefficient
difference 0.0; all non-SH fields byte-identical; all 38 round-trip renders
pixel-identical (PSNR = ∞, SSIM = 1.0)
(results/stages/stage_2_layout.md, stage_2_roundtrip_metrics.json).

## 14. Effective-rank analysis

Singular spectrum of the real X (results/experiments/rank_analysis/): rank 1 holds
90.1% of energy; effective rank at 99% energy is **12** of 48; relative Frobenius
error falls from 0.315 (r=1) to 0.065 (r=16). Synthetic-data validation of the SVD
machinery (exact-rank recovery, monotonicity, mutation-freedom) is unit-tested
(results/stages/stage_3_synthetic_tests.md).

## 15. Low-rank compression results

(results/stages/stage_4_low_rank.md; per-view JSONs in image_metrics/)

| r (fp32) | SH bytes | ratio | PSNR vs base | GT PSNR |
|---|---|---|---|---|
| 1 | 3,949,292 | 52.1× | 19.86 | 18.38 |
| 4 | 16,056,762 | 12.8× | 24.72 | 21.40 |
| 8 | 32,148,149 | 6.4× | 28.01 | 22.76 |
| 16 | 64,315,591 | 3.2× | 34.68 | 24.32 |
| 48 | 194,284,344 | 1.06× | 101.14 | 24.93 |

fp16 factors halve storage with metrics identical to three decimals (lr_r8_fp16 ≡
lr_r8_fp32 at 28.01 dB) — factor precision is not the bottleneck.

## 16. Quantisation results

(results/stages/stage_5_quantization.md)

| config | SH bytes | ratio | PSNR vs base | SSIM vs base | LPIPS vs base | GT PSNR |
|---|---|---|---|---|---|---|
| sq r8 8-bit | 7,099,397 | 29.0× | 28.00 | 0.966 | 0.036 | 22.75 |
| **sq r16 8-bit** | **13,670,820** | **15.0×** | **34.63** | **0.992** | **0.018** | **24.31** |
| vq r8 K=64 | 778,508 | 264× | 23.26 | 0.912 | 0.096 | 20.66 |
| vq r8 K=256 | 1,025,253 | 201× | 24.56 | 0.927 | 0.070 | 21.33 |
| vq r16 K=256 | 1,031,711 | 199× | 24.92 | 0.929 | 0.070 | 21.50 |

8-bit scalar quantisation of U is essentially free (sq_r16 ≡ lr_r16 within 0.05 dB
at 4.7× less storage). K-means VQ trades hard: 200×+ with visible but structured
degradation.

## 17. Rate-distortion analysis

Figures (auto-generated from result files by scripts/generate_report_assets.py):
rate_distortion_psnr/ssim/lpips.png, rate_distortion_vsbase_psnr.png,
psnr_vs_rank.png, storage_breakdown.png, results_table.md. The knee of the curve
sits at the scalar-quantised configurations; VQ forms a separate low-rate cluster.
Key analytical finding: **VQ quality is codebook-limited, not rank-limited** —
r16/K256 gains only 0.37 dB over r8/K256 despite doubling latent rank, because each
Gaussian retains only ⌈log2 K⌉ = 8 bits of identity. Under VQ, index bits (1.07 MB
at K=256) dominate storage; codebook and V are ~10 KB combined — future gains lie
in index entropy coding, not richer codebooks.

## 18. Visual comparison

results/figures/visual_comparisons.png (crops of test view 00012): sq_r16_b8 is
indistinguishable from baseline; vq configurations progressively desaturate
view-dependent colour (green hillside → grey haze) while geometry stays sharp —
the expected failure mode when compressing colour coefficients only. Included
failure case: lr_r1 shows global colour flattening.

## 19. Discussion

Coefficient-space error and image error diverge systematically: rank-1 keeps 90% of
matrix energy yet loses 5 dB of image PSNR, because perceptually salient specular
and shading variation lives in the low-energy tail. Conversely 8-bit scalar noise
on U (coefficient rel-err 0.133 → 0.1333 at r8) is invisible in renders. The
correct evaluation currency for SH compression is rendered-image metrics, and both
references matter: vs-baseline isolates compression loss; vs-GT bounds it by model
quality. Compared with published systems (authors' numbers, not reproduced), our
post-hoc 15× SH-only result is of the same order as LightGaussian's reported
model-level ratio, achieved with a far simpler, retraining-free mechanism — while
the VQ cluster shows what that simplicity costs at extreme rates: without
fine-tuning, quantisation error cannot be redistributed to where it is invisible.

## 20. Limitations

See report/limitations.md: single scene; post-hoc only (no QAT); reconstruction to
full SH before rendering (storage/transmission benefit, unchanged runtime memory);
test views seen in training (documented); temporary reconstructed PLYs are not the
compressed representation (the .npz asset is); Colab hardware variability affects
timing only.

## 21. Threats to validity

Internal: metric bugs — mitigated by sanity tests (identical→∞/1.0, perturbed→worse)
and the bit-exact round-trip anchor. Construct: PSNR/SSIM may miss perceptual
effects — mitigated with LPIPS and visual crops. External: one scene — findings are
existence results for this scene class, not universal claims. Conclusion: seeds
fixed for VQ; SVD deterministic; per-view distributions saved.

## 22. Future work

Index entropy coding (arithmetic/rANS over VQ indices); quantisation-aware
fine-tuning; residual VQ; per-cluster bases; second-scene validation (truck);
on-the-fly decode in the rasteriser.

## 23. Conclusion

Scene-level truncated SVD plus cheap quantisation is a genuinely effective post-hoc
compressor for 3DGS SH: 15× SH-only (3.6× whole-model) at near-transparent quality,
201–264× at usable quality, with every claim traceable to measured artifacts. The
answer to the research question is yes — with the honest qualification that at
extreme rates the binding constraint becomes index entropy, which post-hoc methods
alone cannot relax.

## 24. Reproducibility instructions

One notebook: `notebooks/colab_runner.ipynb` on a fresh Colab T4 runtime, Run all
(STAGE='all'). It pins the branch, installs pinned deps, compiles the rasteriser,
downloads model+data, runs all unit tests, renders the baseline, verifies the
bit-exact round trip, reproduces the headline configuration (sq_r16_b8) end-to-end,
and asserts its metrics (>34 dB vs base, >24 dB GT, SSIM >0.99, asset <15 MB).
Local CPU work: `pip install -e .[metrics,dev] && pytest tests -q`; figures:
`python scripts/generate_report_assets.py`.

## 25. References

- B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. "3D Gaussian Splatting for
  Real-Time Radiance Field Rendering." ACM TOG 42(4), 2023.
- Z. Fan et al. "LightGaussian: Unbounded 3D Gaussian Compression with 15x
  Reduction and 200+ FPS." arXiv:2311.17245, 2023.
- J. C. Lee et al. "Compact 3D Gaussian Representation for Radiance Field."
  CVPR 2024.
- Y. Chen et al. "HAC: Hash-grid Assisted Context for 3D Gaussian Splatting
  Compression." ECCV 2024.
- T. Lu et al. "Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering."
  CVPR 2024.
- Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli. "Image quality assessment: from
  error visibility to structural similarity." IEEE TIP 13(4), 2004. (SSIM)
- R. Zhang, P. Isola, A. Efros, E. Shechtman, O. Wang. "The Unreasonable
  Effectiveness of Deep Features as a Perceptual Metric." CVPR 2018. (LPIPS)
