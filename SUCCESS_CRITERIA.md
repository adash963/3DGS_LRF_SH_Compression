# Success criteria

## Stage gates

- S0: SH layout determined from source; requirements internally consistent; no custom
  CUDA kernel required by the plan.
- S1: fresh Colab runtime → clone pinned commit → install → compile rasteriser → load
  pretrained scene → render evaluation set; all finite, correct dimensions.
- S2: extract → unchanged reinsert → save → render: max SH coefficient diff at numerical
  precision (< 1e-6 fp32); round-trip renders visually identical (PSNR > 50 dB vs
  baseline renders); non-SH fields byte-identical.
- S3: synthetic tests — exact rank-k data reconstructs at rank k (rel. Frobenius error
  < 1e-6); error monotone non-increasing in rank; energy monotone non-decreasing;
  full-rank matches input; malformed inputs rejected.
- S4: full-rank compressed round trip reproduces baseline; lower ranks render; saved
  assets load in a clean process; reported size == actual file size.
- S5: every codebook config decodes; ≥1 quantised config renders all eval views; no
  NaN/invalid colours; complete byte accounting.
- S6: all metrics regenerable from saved files without manual value copying; metric
  sanity checks pass (identical → ~perfect, perturbed → worse).
- S7: experiment matrix complete with per-run artifacts; figures auto-generated; failed
  runs documented.
- S8: clean-checkout + fresh-Colab reproduction of baseline, one compressed config,
  main metrics and figures; every report claim traceable to an artifact or citation.

## Honesty constraints

- Negative or weak rate-distortion results are reported and analysed, not hidden.
- No threshold is weakened to force a pass; disabled tests are documented with risk.
- Published numbers from literature are labelled as quoted, never as reproduced.
