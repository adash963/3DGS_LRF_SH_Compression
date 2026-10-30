# Static audit discrepancies

This file lists only discrepancies, evidentiary gaps, and recommended corrections. See `STATIC_AUDIT.md` for the complete claim-by-claim audit.

## Material discrepancies

### 1. Scalar-storage formula is incorrect

`report/report_draft.md:97` states:

`scalar = N·r + 2·48·r·4 + 48·r·s`

The implementation stores `codes [N,r]`, `scale [r]`, `offset [r]`, and `V [r,48]` (`src/sh_compression/quantization.py:11-19`; `src/sh_compression/serialization.py:69-76`). For 8-bit scalar codes and fp32 V, the raw formula is:

`N·r·1 + 2·r·4 + r·48·4 + metadata/container overhead`.

The published formula overcounts scale/offset by 48×. This does not invalidate the headline measured `.npz` size, because the code uses actual file size as authoritative, but the mathematical description must be corrected.

### 2. “Byte-identical” is not what the verifier checks

The real-model verification converts each non-SH field to float32 and calls `np.array_equal` (`src/sh_compression/extraction.py:26-31`). It does not compare raw byte buffers, field dtypes, PLY headers, or complete-file regions. The stored Boolean in `results/stages/stage_2_roundtrip_metrics.json:4` therefore supports value equality after float32 conversion, not byte identity.

Recommended wording: “All non-SH float32 field values compared equal after write/read.” Add buffer/file hashes before claiming byte identity.

### 3. Pixel-identical round-trip has no machine-readable evidence

`results/stages/stage_2_layout.md:20-27` says all 38 views are pixel-identical, but `stage_2_roundtrip_metrics.json:1-5` contains no render results, and no round-trip renders or hashes are committed.

Recommended correction: label the statement as a recorded run outcome pending archival, or commit per-view filenames plus image hashes and metrics.

### 4. Universal JSON/CSV traceability claim is false

The repository cannot support “every reported result is traceable to committed JSON or CSV.” At least these rely on Markdown or missing raw artifacts:

- point count and baseline PLY file size: `results/stages/stage_1_baseline.md:21-24`;
- 38-view round-trip pixel identity: `results/stages/stage_2_layout.md:20-27`;
- visual-crop selection/provenance: `results/stages/stage_7_ablation.md:17-23`;
- raw PLY, renders, assets, and logs: absent or ignored by `.gitignore:13-16`.

Recommended wording: “All numeric SVD, storage, and rendered-metric tables are traceable to committed CSV/JSON; baseline and round-trip execution claims are recorded in stage evidence, with raw inputs/outputs not archived.”

### 5. Figure provenance is mixed

`scripts/generate_report_assets.py:14-95` does load committed JSON for numeric rate-distortion plots and `results_table.md`. It does not establish the stronger all-figures claim:

- storage breakdown loads ignored `.npz` assets (`generate_report_assets.py:97-120`);
- rank figures are generated while re-analysing the PLY rather than by reading committed rank CSVs (`scripts/analyse_rank.py:23-68`);
- `visual_comparisons.png` depends on uncommitted raw render crops (`scripts/export_crops.py:14-39`);
- committed PNGs contain no checked source-file hashes.

Recommended correction: publish a figure manifest containing figure path, generator, source paths, source hashes, generator commit, and output hash. Make the rank-figure generator consume committed CSVs.

### 6. Experiment artifact package is incomplete

The 22 ablation directories do contain `config.yaml`, `metrics.json`, and `git_commit.txt`. However, `EXPERIMENT_PLAN.md:42-46` also promised `environment.json`, stdout/stderr logs, the compressed asset, and sample renders. Those items are absent from Git. The 12 render records contain embedded config but no Git commit field.

Recommended correction: either archive the promised items or narrow the reproducibility claim. At minimum add `git_commit`, environment, input hashes, and command line to each image-metric JSON.

### 7. `aa39dcb` is not the final described state

The Git object contains the stage-1–7 evidence and visual comparison, but at that object:

- `report/report_draft.md` still contains stage-8 TODO placeholders;
- `STAGE_STATUS.yaml` says stage 6 is in progress and lists only stages 0–3 as passed;
- the full report/reproduction documentation was added later in `ae2faa2` and `778afd6`.

Recommended correction: call `aa39dcb` the “stage-7 evidence boundary,” or update the cited commit to the commit intended to represent the final report state.

## Numerical and unit discrepancies

### 8. MB and MiB are not distinguished

| Item | Bytes | MB (10^6) | MiB (2^20) |
|---|---:|---:|---:|
| baseline PLY | 265,724,108 | 265.724108 | 253.414257 |
| baseline SH | 205,720,704 | 205.720704 | 196.190552 |
| sq-r16 render-run asset | 13,670,820 | 13.670820 | 13.037510 |
| calculated sq-r16 whole representation | 73,674,224 | 73.674224 | 70.261215 |

The manuscript values are correct as decimal MB. Label them explicitly; do not use MB and MiB interchangeably.

### 9. Duplicate run records have slightly different asset sizes

The storage sweep and render sweep record separate `.npz` totals for the same names:

| Configuration | Storage sweep | Render sweep | Render − storage |
|---|---:|---:|---:|
| lr_r1_fp32 | 3,949,315 | 3,949,292 | −23 |
| lr_r2_fp32 | 7,967,493 | 7,967,470 | −23 |
| lr_r4_fp32 | 16,056,786 | 16,056,762 | −24 |
| lr_r8_fp32 | 32,148,190 | 32,148,149 | −41 |
| lr_r16_fp32 | 64,315,628 | 64,315,591 | −37 |
| lr_r48_fp32 | 194,284,343 | 194,284,344 | +1 |
| lr_r8_fp16 | 16,144,113 | 16,144,088 | −25 |
| sq_r8_b8 | 7,099,421 | 7,099,397 | −24 |
| sq_r16_b8 | 13,670,845 | 13,670,820 | −25 |
| vq_r8_k64 | 778,534 | 778,508 | −26 |
| vq_r8_k256 | 1,025,279 | 1,025,253 | −26 |
| vq_r16_k256 | 1,031,737 | 1,031,711 | −26 |

These differences are immaterial to rounded ratios, but the report uses render-run sizes while `results/figures/results_table.md` uses storage-run sizes. Identify the source run in every table and do not imply one universal byte count.

### 10. Component accounting is raw, while reported total is ZIP-compressed

`CompressedSH.component_bytes()` reports raw NumPy array payload sizes and adds a hard-coded 512-byte metadata estimate (`serialization.py:114-127`). `save()` uses `np.savez_compressed` and returns actual archive bytes (`serialization.py:49-79`). Therefore component numbers do not add to the reported compressed file size and should not be presented as a compressed-byte partition.

For sq-r16-b8, source-defined raw accounting is:

- uint8 codes: 17,143,392 bytes;
- fp32 V: 3,072 bytes;
- fp32 scale/offset: 128 bytes;
- metadata estimate: 512 bytes;
- total raw estimate: 17,147,104 bytes;
- measured compressed asset: 13,670,820 or 13,670,845 bytes depending on run.

### 11. K=64/K=128 indices are not bit-packed

The report describes `ceil(log2 K)` logical bits per Gaussian, but `_index_dtype` stores uint8 for all K≤256 (`src/sh_compression/quantization.py:61-66`). K=64 and K=128 therefore occupy 8 physical bits per index, not 6 or 7. This does not affect the K=256 calculation but affects general VQ storage descriptions.

Recommended wording: “Logical index entropy is ceil(log2 K) bits; the current `.npz` implementation uses uint8 storage for K≤256.”

### 12. “Codebook and V are ~10 KB” is rank-dependent

At K=256:

- r8: codebook 8,192 B + V 1,536 B = 9,728 B;
- r16: codebook 16,384 B + V 3,072 B = 19,456 B.

The statement in `report/report_draft.md:176-180` is accurate for r8 but not r16. The broader conclusion that indices dominate remains supported: about 99.05% of raw accounted bytes at r8 and 98.17% at r16.

### 13. Whole-model size is calculated, not emitted

The 73.7 MB value follows exactly from the code's composite model:

`265,724,108 − 205,720,704 + 13,670,820 = 73,674,224 bytes`.

There is no committed deployed package of that size. The temporary reconstructed PLY is explicitly not compressed storage (`src/sh_compression/extraction.py:60-65`; `scripts/reconstruct_model.py:18-22`). Say “calculated deployable composite size” rather than implying that an emitted PLY shrank to 73.7 MB.

## Scope and interpretation discrepancies

### 14. FP16 render equivalence is only demonstrated at rank 8

The repository has render JSONs for `lr_r8_fp16` and `lr_r8_fp32`, and they are effectively identical. It has coefficient/storage records for r4 and r16 fp16 but no corresponding fp16 render JSONs. Limit the render-quality statement to rank 8, or render the remaining pairs.

### 15. “34 experiment runs” counts phases, not unique configurations

There are 22 unique configured names. Twelve of those names also have a render-phase JSON. Thus 34 means 22 storage records plus 12 render-phase records, not 34 unique configurations. No committed logs establish whether these were 34 separate process invocations.

### 16. No recorded failures is narrower than no failures occurred

All 22 status rows are `ok`; all 12 render records have empty `failed_views` and complete 38-view arrays. This supports “no committed experiment or view is recorded as failed.” Missing logs and raw session output prevent the stronger historical claim that no unrecorded attempts failed. `DEVELOPMENT_LOG.md:68-71` in fact documents one failed transfer attempt, although not a failed experiment.

### 17. Full-rank reconstruction is numerical, not exact

The rank-analysis float64 path records relative error `1.3290e-13`; the serialized float32 storage sweep records about `1.706e-7`; the render JSON rounds coefficient error to `0.0`; and renders average 101.1419 dB, finite rather than infinite. Use “numerically equivalent” or “near-exact,” not “bit-exact,” for the full-rank compressed configuration.

### 18. Render metrics are committed only to four decimals

`DEVELOPMENT_LOG.md:63-71` says per-view values were rounded to four decimals during transfer and raw originals remained in Colab. Re-averaging all 38 stored rows reproduces every stored summary after rounding, which supports the published precision but not additional digits.

### 19. Notebook is not pinned by default

`notebooks/colab_runner.ipynb:13-18` defines `COMMIT=''`, meaning branch tip, even though the comments call the step a pinned ref. Reproduction can drift as the branch advances. Set an exact commit in the distributed notebook or require the user to fill it before Run all.

### 20. Fresh-runtime reproduction remains pending

At inspected HEAD, `STAGE_STATUS.yaml:1-18` marks stage 8 in progress and fresh-runtime Run-all pending; `DEVELOPMENT_LOG.md:77-84` says it was launched but does not record completion. The notebook also omits LPIPS in its headline reproduction command (`--no-lpips`, `notebooks/colab_runner.ipynb:96`). Do not describe the full reproduction package as completed until a committed outcome record exists.
