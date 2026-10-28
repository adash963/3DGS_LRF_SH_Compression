# Limitations (running list — kept honest as evidence accumulates)

- Single development scene (T&T `train`); one optional validation scene at most.
  Conclusions cannot generalise across scene types without broader evaluation.
- Post-hoc compression of a frozen checkpoint; no end-to-end or quantisation-aware
  training. Reported quality is a lower bound on what a trained variant might reach.
- Reconstruction to full SH before rendering: runtime memory footprint is unchanged;
  the contribution is storage/transmission compression only. No custom rasteriser
  integration was implemented.
- Coefficient-space error is not a perceptual metric; image metrics (PSNR/SSIM/LPIPS)
  on rendered views are the primary quality evidence, and they in turn depend on the
  chosen evaluation views.
- Colab hardware variability (driver/GPU allocation) can affect timing measurements;
  storage and quality metrics are hardware-independent.
- Published comparison numbers (LightGaussian, HAC, …) are quoted from papers, not
  reproduced here.
