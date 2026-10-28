# Discussion (developed in report_draft.md §19; evidence-backed notes below)

All points below are now supported by committed results:

- Why coefficient-space error and rendered-image error differ: rendering weights
  coefficients by view-dependent SH basis functions and opacity/overlap; low-energy
  coefficients (high orders) can carry visually salient specular detail.
- Storage anatomy: for N ≈ 10^6 and r=8, U ≈ 32 MB fp32 vs indices ≈ 1 MB at K=256 —
  VQ of U is where the ratio comes from; V and codebook are noise.
- Expected trend: PSNR monotone in r and K; knee of the rate-distortion curve is the
  headline result whether or not it beats published systems.
- Negative-result framing if quality collapses at useful ratios: quantify the gap,
  attribute it (rank floor vs quantisation noise), connect to why published methods
  add per-Gaussian fine-tuning.
