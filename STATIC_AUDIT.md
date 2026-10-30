# Static audit of the SH-compression findings

Audit date: 2026-08-02  
Repository: `/Users/kaushikthiyagarajan/projects/dash_ai/3DGS`  
Branch: `dissertation`  
Inspected HEAD: `778afd63d7bd9f7e30f6078b50f5727a87898e2d`  
Reported target object: `aa39dcbd8e7b84fb5efef750fb9a65f967a3e6d7`

## 1. Executive conclusion

The central numerical story is supported by committed source and result records, but the repository does not support every claim at the same evidentiary level.

- **Supported (15):** claims 1–6, 10, 12–13, 15–18, 20, and 24.
- **Partially supported (7):** claims 8, 11, 14, 19, 22, 23, and 25.
- **Contradicted (1):** claim 21.
- **Not statically verifiable (1):** claim 7.
- **Missing evidence (1):** claim 9.

The strongest supported result is the rank-16, 8-bit scalar operating point. The committed render record reports 13,670,820 bytes, 15.0482× SH compression, 34.6322 dB PSNR, 0.9919 SSIM, and 0.0184 LPIPS versus baseline across 38 stored per-view rows (`results/experiments/image_metrics/metrics_sq_r16_b8.json:8-26`). From the committed baseline figures, the calculated whole representation is 73,674,224 bytes, or 73.674 MB (decimal), and 3.6067× smaller than the original PLY.

The main qualifications are:

1. The baseline PLY, raw baseline/round-trip/compressed renders, and deployed compressed assets are not committed. The PLY count and size therefore remain committed-result assertions rather than direct static observations.
2. The round-trip JSON records coefficient equality and a Boolean for non-SH fields, but no committed per-view round-trip metrics or images support the “38 pixel-identical views” claim.
3. “Byte-identical” is stronger than the implementation: it compares arrays after conversion to float32, not raw bytes (`src/sh_compression/extraction.py:26-31`).
4. The report's scalar-storage formula is wrong: it writes `2·48·r·4` for scale/offset, while the implementation stores two vectors of length `r`, so the term must be `2·r·4` (`report/report_draft.md:91-98`; `src/sh_compression/quantization.py:11-19`; `src/sh_compression/serialization.py:69-76,123-126`).
5. The committed storage-sweep and render-sweep `.npz` byte values differ by 1–41 bytes for overlapping configurations. The difference is negligible numerically, but the report must name which run supplies the byte count.
6. `aa39dcb` contains the stage-1–7 code/results/figures, but not the final findings draft or current reproduction documentation; at that object, `STAGE_STATUS.yaml` still says stage 6 is in progress.

## 2. Repository and commits inspected

The worktree was clean before this audit. HEAD was `778afd6` (“README reproduction + troubleshooting; dev log stage 8 entry”). The requested object `aa39dcb` exists and is an ancestor of HEAD. Seven tracked files changed after it: `DEVELOPMENT_LOG.md`, `README.md`, `STAGE_STATUS.yaml`, `notebooks/colab_runner.ipynb`, `report/discussion.md`, `report/report_draft.md`, and `report/results.md`.

Git history places the principal evidence at:

- `91c9873`: initial package, configurations, and claimed 46 unit tests.
- `c705152`: completed `cfg_args` compatibility fix.
- `5f00812` / `3ca40fc`: stage evidence and rank-analysis outputs.
- `82b0fa5`: storage sweep, 12 render records, and rate-distortion outputs.
- `27a3b14`: LPIPS records and figure.
- `aa39dcb`: visual-comparison figure and stage-7 summary.
- `ae2faa2` and `778afd6`: final report/reproduction documentation after `aa39dcb`.

All 22 `git_commit.txt` files contain `91c98737dd48da5a1302d59e062ef4bc498527cd`. This hash is valid and is an ancestor of `aa39dcb`, but it identifies the source checkout recorded by the storage sweep, not the later commit that first committed the result records.

## 3. Scope and limitations

This was a non-execution audit. No Python, tests, notebooks, renderer, CUDA build, package installation, experiment, or project code was run. Inspection was limited to source text, JSON/CSV/YAML/Markdown, figures, static ZIP metadata, and Git objects. Simple arithmetic was recalculated from stored numbers with generic text-processing tools.

A committed result record is treated as evidence that the repository records a result, not independent proof that the experiment ran. The following are absent from Git:

- the 265,724,108-byte baseline PLY;
- raw baseline, round-trip, or compressed renders;
- compressed `.npz` assets (`.gitignore:13-16`);
- stdout/stderr logs for each experiment;
- per-run environment files except the one stage-1 environment record.

Ignored `.npz` files are present in this working tree and their static ZIP member headers agree with the storage implementation, but they are explicitly excluded from Git and are not reproducible evidence from a clean checkout. This audit does not use them to upgrade any claim's classification.

## 4. Claim-by-claim findings

| # | Classification | Finding and exact evidence |
|---:|---|---|
| 1 | **Supported** | `results/stages/stage_1_baseline.md:21-24` records 1,071,462 points. The same `n_points` is recorded in `results/stages/stage_2_roundtrip_metrics.json:2` and is implied exactly by `baseline_sh_bytes=205720704` in `results/experiments/ablation/all_results.csv:2-23`, because `205720704/(48·4)=1,071,462`. The PLY itself is absent, so this is artifact support rather than direct inspection. |
| 2 | **Supported** | `results/stages/stage_1_baseline.md:22` and every row of `results/experiments/ablation/all_results.csv:2-23` record 265,724,108 bytes. That is 265.724108 MB or 253.414257 MiB. “265.7 MB” is correct only with decimal MB. |
| 3 | **Supported** | `src/sh_compression/storage.py:12-14` defines SH bytes as `N·48·4`; `results/experiments/ablation/all_results.csv:2-23` records 205,720,704. Recalculation gives 205.720704 MB, 196.190552 MiB, and `205720704/265724108=77.4189%`. All 48 fields, including three DC terms, are included by `src/sh_compression/sh_layout.py:13-20,36-57`. |
| 4 | **Supported** | The canonical `[N,3,16]` and flat `[N,48]` mappings are explicit in `src/sh_compression/sh_layout.py:1-20,23-57`; reference PLY load semantics are in `scene/gaussian_model.py:281-294,307-314`; tests encode the same mapping at `tests/unit/test_sh_layout.py:7-36`. |
| 5 | **Supported** | Stored SVD values are in `results/experiments/rank_analysis/energy_vs_rank.csv:2-11` and `error_vs_rank.csv:2-11`. The requested rows are: r1 energy 0.9009290279/error 0.3147554163; r4 0.9691884051/0.1755323186; r8 0.9823779192/0.1327481858; r12 0.9912406646/0.0935913211; r16 0.9957619009/0.0651006847; r48 1.0/1.3290e-13. `src/sh_compression/factorization.py:66-87` reconstructs `(U_r·s_r)@Vt_r`, calculates error, and uses squared singular values for energy. The input matrix is absent, so the error values cannot be independently recomputed. |
| 6 | **Supported** | `src/sh_compression/factorization.py:91-94` defines the smallest rank whose cumulative `s²` share reaches 0.99. Recalculation from all singular values in `singular_values.csv:2-49` gives r11 below 0.99 and r12 = 0.9912406646, hence effective rank 12. |
| 7 | **Not statically verifiable** | The implementation is lossless in layout (`src/sh_compression/sh_layout.py:23-33`) and the committed record says `max_abs_coeff_diff: 0.0` (`results/stages/stage_2_roundtrip_metrics.json:2-4`). However, the source and output PLYs/hashes are absent. A result JSON is not independent proof of the real-model run. |
| 8 | **Partially supported** | The result JSON records `non_sh_fields_identical: true` (`results/stages/stage_2_roundtrip_metrics.json:4`), and insertion leaves non-SH array objects unchanged (`src/sh_compression/ply_io.py:73-84`). But the verification converts both arrays to float32 and calls `array_equal` (`src/sh_compression/extraction.py:26-31`); it does not compare bytes, dtypes, PLY headers, or file regions. “Value-identical after float32 conversion” is supported; “byte-identical” is not. |
| 9 | **Missing evidence** | Only the narrative `results/stages/stage_2_layout.md:20-27` says 38/38 views were pixel-identical. The machine-readable round-trip file has only three fields and no render rows (`stage_2_roundtrip_metrics.json:1-5`). No round-trip images or hashes are committed. |
| 10 | **Supported** | `results/experiments/image_metrics/metrics_lr_r48_fp32.json:1-25` records 38 views, zero failed views, 101.1419 dB, and SSIM 1.0 versus baseline. Re-averaging its 38 stored per-view rows yields the same values at the stored four-decimal precision. This is artifact support; raw images are absent. |
| 11 | **Partially supported** | `results/figures/results_table.md:3-24` has 22 rows whose storage/coefficient columns match `all_results.csv:2-23` and whose populated render columns match the 12 image-metric JSONs at displayed precision. The generator loads those files (`scripts/generate_report_assets.py:14-35,64-95`). However, the report tables use render-run asset bytes while the generated table uses storage-run bytes. The overlaps differ by 1–41 bytes (for example sq-r16: 13,670,845 in `all_results.csv:14` versus 13,670,820 in `metrics_sq_r16_b8.json:8`). Source selection is therefore ambiguous. |
| 12 | **Supported** | `metrics_sq_r16_b8.json:8-26` records 13,670,820 bytes, 15.0482×, 34.6322 dB, 0.9919 SSIM, 0.0184 LPIPS, 38 views, and no failures. The metrics are explicitly versus baseline in the `vsbase_*` keys; GT values are separate at lines 18-19 and 22. The storage-run record is 25 bytes larger (`results/experiments/ablation/sq_r16_b8/metrics.json:11-17`) but rounds identically. |
| 13 | **Supported** | `src/sh_compression/storage.py:25-29` defines the whole representation as baseline non-SH bytes plus compressed SH asset bytes. `265,724,108−205,720,704+13,670,820=73,674,224` bytes = 73.674224 MB = 70.261 MiB, ratio 3.6067×. It is a calculated composite size, not a committed deployed model file. |
| 14 | **Partially supported** | r8 fp16 and fp32 have effectively identical stored render metrics: 28.0127 dB in both, GT PSNR differs by 0.000047 dB, SSIM by at most 0.0001, and LPIPS by about 0.00001 (`metrics_lr_r8_fp16.json` and `metrics_lr_r8_fp32.json`, `summary` keys). Coefficient errors are close for r4/r8/r16 in `all_results.csv:4,6,8,10-12`, but r4 and r16 fp16 were not among the 12 rendered records. The broad plural claim is only render-supported at r8. |
| 15 | **Supported** | `metrics_lr_r16_fp32.json:16-25` versus `metrics_sq_r16_b8.json:17-26` gives changes of −0.0484 dB vs baseline PSNR, −0.0003 SSIM, +0.0002 LPIPS, and −0.0046 dB GT PSNR. “Almost no loss” is a fair qualitative description. |
| 16 | **Supported** | At K=256, r8→r16 changes vs-baseline PSNR 24.5572→24.9249 (+0.3677 dB), SSIM 0.9266→0.9287, and LPIPS 0.0701→0.0702 (`metrics_vq_r8_k256.json:18-27`; `metrics_vq_r16_k256.json:18-27`). The coefficient error only changes 0.1982→0.1958. “Plateau” is qualitative but consistent with the committed records. |
| 17 | **Supported** | K≤256 uses uint8 indices (`src/sh_compression/quantization.py:45-49,61-66,93-98`). For K=256/r8, raw accounting is 1,071,462 B indices, 8,192 B codebook, 1,536 B V, plus the code's 512 B metadata estimate: indices are 99.05%. At r16 they are 98.17%. Note that this is raw-array accounting; the authoritative `.npz` is ZIP-compressed. |
| 18 | **Supported** | Static pytest collection yields 46 unit cases: factorization 11 (one test parameterized three ways), metrics 4, PLY 7, quantization 9, serialization 9 (one test parameterized four ways), layout 6. Integration yields 4 cases from a three-way parameterized pipeline plus one point-count case (`tests/integration/test_full_pipeline.py:36-67`). No tests were executed in this audit. |
| 19 | **Partially supported** | The committed tree has 22 ablation directories/rows (`configs/quantization_ablation.yaml:2-28`; `all_results.csv:2-23`) and 12 image-metric JSONs, matching `results/stages/stage_7_ablation.md:5-15`. This is 34 run records if storage and render phases are counted separately, but the 12 rendered configurations overlap the 22 names, and there are no per-run logs proving 34 separate invocations. |
| 20 | **Supported** | All 22 `all_results.json` rows have `status="ok"`; all 12 image JSONs have `failed_views: []`, `n_views: 38`, 38 unique GT rows, and 38 unique baseline rows. `run_ablation.py:46-79` records exceptions as failed status; `metrics.py:61-93` records per-view exceptions. The claim is supported as “no committed record says failed,” not as proof that no unrecorded attempt failed. |
| 21 | **Contradicted** | Several report claims trace only to Markdown or absent raw files: baseline point/file statistics (`stage_1_baseline.md:21-24`), round-trip pixel identity (`stage_2_layout.md:20-27`), and visual-comparison provenance (`stage_7_ablation.md:17-23`). The PLY, renders, logs, and assets are not committed. Therefore not every reported result is traceable to committed JSON/CSV. |
| 22 | **Partially supported** | Numeric rate-distortion plots and the table are sourced from committed JSONs by `generate_report_assets.py:14-95`. But storage breakdown loads ignored `.npz` assets (`generate_report_assets.py:97-120` and `.gitignore:15-16`), SVD figures are generated during a new PLY analysis rather than from the committed CSV (`scripts/analyse_rank.py:23-68`), and `visual_comparisons.png` is made from uncommitted raw renders/crops via `scripts/export_crops.py:14-39`. No static provenance metadata inside the PNGs links them to exact source hashes. |
| 23 | **Partially supported** | Each of the 22 ablation directories contains `config.yaml`, `metrics.json`, and `git_commit.txt`, and names agree. The Git hash is valid. However, the 12 rendered records are flat JSON files with embedded config but no Git commit; and the broader promised directory contents—environment, logs, asset, sample renders—are missing from Git despite `EXPERIMENT_PLAN.md:42-46`. |
| 24 | **Supported** | The Windows path/eval fix and the later `depths`/`train_test_exp` completion are implemented in `scripts/fix_cfg_args.py:1-11,23-38`, invoked by the notebook at `notebooks/colab_runner.ipynb:74-89`, and documented in `DEVELOPMENT_LOG.md:20-45`. |
| 25 | **Partially supported** | `aa39dcb` contains the stage-1–7 result files and figures, including the stage-7 summary and visual comparison. But its `report/report_draft.md` still contains stage TODOs, and its `STAGE_STATUS.yaml:1-16` says stage 6 is in progress with only stages 0–3 passed. The complete findings draft and reproduction path were committed later (`ae2faa2`, `778afd6`), so `aa39dcb` is not the full documented state now described. |

## 5. Storage calculations and units

### Baseline

| Quantity | Bytes | Decimal MB | Binary MiB |
|---|---:|---:|---:|
| PLY | 265,724,108 | 265.724108 | 253.414257 |
| all 48 SH float32 fields | 205,720,704 | 205.720704 | 196.190552 |
| non-SH remainder | 60,003,404 | 60.003404 | 57.223705 |

SH share is 77.4189%. The report's “MB” values consistently match decimal MB, while code/figure component labels use KiB (`scripts/generate_report_assets.py:111-115`). The report should explicitly state decimal MB and avoid presenting the same numbers as MiB.

The SH count includes DC: 3 `f_dc` plus 45 `f_rest` = 48 float32 fields. There is no evidence that only the higher-order 45 were used in the 205.7 MB figure.

### Rank-16 scalar, 8 bit

Using source-defined dtypes:

- codes: `1,071,462·16·1 = 17,143,392` raw bytes;
- V: `16·48·4 = 3,072` raw bytes;
- scale + offset: `2·16·4 = 128` raw bytes;
- metadata: the implementation reports a fixed 512-byte estimate;
- raw component estimate: 17,147,104 bytes;
- measured ZIP-compressed asset record: 13,670,845 bytes in the storage sweep or 13,670,820 bytes in the render sweep.

The measured `.npz` can be smaller than the raw code array because `np.savez_compressed` is used (`serialization.py:74-79`). `component_bytes()` is raw-array accounting plus an estimated metadata value, not a partition of the compressed file (`serialization.py:114-127`). The committed metrics do not include `component_bytes`, and the assets are ignored, so complete archive-member accounting is not reproducible from Git.

The formula in `report/report_draft.md:97` must be corrected from:

`N·r + 2·48·r·4 + 48·r·s`

to, for 8-bit codes and fp32 V:

`N·r·1 + 2·r·4 + 48·r·4 + container/metadata overhead`.

### K=256 VQ

Implementation dtypes give:

| Mode | Indices | Codebook | V | Metadata estimate | Raw total | Index share |
|---|---:|---:|---:|---:|---:|---:|
| r8/K256 | 1,071,462 | 8,192 | 1,536 | 512 | 1,081,702 | 99.05% |
| r16/K256 | 1,071,462 | 16,384 | 3,072 | 512 | 1,091,430 | 98.17% |

The K=256 index-dominance claim is supported. The report's statement that “codebook and V are ~10 KB combined” (`report/report_draft.md:176-180`) applies approximately to r8 (9,728 B) but not r16 (19,456 B). Also, the mathematical description `ceil(log2 K)` bits is not the physical representation for K=64/128: the implementation stores one uint8 byte per index for every K≤256 (`quantization.py:61-66`).

Temporary reconstructed PLY size is not used in the storage formula. `decompress_to_ply` explicitly calls it a temporary artifact (`extraction.py:60-65`), and `reconstruct_model.py:18-22` warns that it is not compressed size.

## 6. SVD-result artifacts

`factorization.py:26-29` computes float64 SVD, folds singular values into U, and returns float32 factors. Reconstruction is `U@V` in float32 chunks (`factorization.py:32-39`). This matches the documented `X≈UV` formulation and not a different `USV` convention.

Energy is correctly defined as squared singular-value mass (`factorization.py:73,84-86,91-94`). Recalculation from the 48 committed singular values reproduces the stored energies at ranks 1, 4, 8, 12, 16, and 48 and yields effective rank 12. Error CSV values cannot be independently reproduced without X, which is not committed.

There are two legitimate full-rank error values with different numerical paths:

- rank-analysis float64 reconstruction: relative error `1.3290e-13` (`error_vs_rank.csv:11`);
- serialized float32 factors: about `1.706e-7` in the storage sweep (`all_results.csv:9`).

The image-metric file rounds `coeff_rel_frobenius_error` to `0.0` (`metrics_lr_r48_fp32.json:10`). “Numerically exact” is fair; “bit-exact” would not be.

## 7. Rendering-metric artifacts

Each of the 12 image JSONs contains:

- `n_views: 38`;
- 38 unique `per_view_vs_gt` rows;
- 38 unique `per_view_vs_baseline` rows;
- an empty `failed_views` array;
- separate `gt_*` and `vsbase_*` summary keys.

Re-averaging every stored per-view PSNR, SSIM, and LPIPS column matches each summary after four-decimal rounding. Thus the committed averages cover all stored views, not a selected subset. The code uses sorted common filenames (`metrics.py:61-83`) and averages finite successful rows (`metrics.py:84-93`). It does not report filenames present in only one directory, but the committed files have 38 unique rows in both comparisons.

The development log states that metrics were rounded to four decimals during transfer and the raw originals remained only in the Colab session (`DEVELOPMENT_LOG.md:63-71`). Claims beyond four-decimal precision are therefore unavailable from the repository.

The report correctly distinguishes “versus baseline” from “versus ground truth” in the headline tables. For sq-r16, 34.6322 dB/0.9919/0.0184 are versus baseline; 24.3122 dB/0.8585/0.1560 are versus GT. The claimed −0.62 dB GT change is `24.3122−24.9298 = −0.6176 dB`.

## 8. Quantisation accounting

Scalar quantisation is per-column min/max uniform quantisation with uint8 or uint16 codes and float32 scale/offset (`quantization.py:11-39`). V remains factor dtype, normally float32 (`extraction.py:43-54`). The serialized member set is V, codes, scale, offset, and JSON metadata (`serialization.py:61-76`).

VQ stores float32 codebook, uint8/16/32 indices selected from K, float32 V, and JSON metadata (`quantization.py:45-66,69-98`; `serialization.py:61-76`). At K=256, the index array dominates raw storage. At K=64 and K=128, it still occupies one byte per Gaussian rather than the ideal 6 or 7 bits; the report should distinguish logical entropy from physical storage.

Measured file bytes remain authoritative in code (`storage.py:17-29`), but the 512-byte metadata component is explicitly an estimate, and ZIP headers/member overhead are not included in `component_bytes()`. “Complete byte accounting” should mean all logical members are named plus total `.npz` bytes measured, not that compressed bytes are exactly allocated to components.

## 9. Test and experiment counts

### Tests

Static pytest collection count:

| Suite/file group | Defined test functions | Parameter expansion | Collected cases |
|---|---:|---:|---:|
| factorization | 9 | +2 | 11 |
| metrics | 4 | 0 | 4 |
| PLY round trip | 7 | 0 | 7 |
| quantization | 9 | 0 | 9 |
| serialization | 6 | +3 | 9 |
| SH layout | 6 | 0 | 6 |
| **unit total** | **41** | **+5** | **46** |
| integration | 2 | +2 | 4 |
| **total** | **43** | **+7** | **50** |

This verifies the collection count from source, not that the tests pass at HEAD.

### Experiments

- 22 configured storage/coefficient variants and 22 `ok` rows.
- 12 committed render-metric files for the documented subset.
- 0 failure records; each render record contains 38+38 per-view comparisons.

Calling this “34 runs” is acceptable only if a storage run and a later render run of the same named configuration count separately. There are 22 unique configuration names, not 34.

## 10. Traceability and reproducibility

Traceability is good for the numeric ablation tables and weaker for baseline/round-trip/visual claims.

- **Directly verified from source logic:** layout, inclusion of DC, `X≈UV`, squared-energy criterion, scalar/VQ dtypes, logical member accounting, storage formulas, test collection count, failure-recording paths, and Colab fixes.
- **Supported by committed experiment artifacts:** SVD values, 22 storage rows, 12×38 render rows, headline quality metrics, no recorded failures, and run-specific file sizes.
- **Asserted only in documentation or small summary JSON:** real PLY count/size, actual execution environment, real-model coefficient round trip, and 38 round-trip pixel identities.
- **Requires execution or missing raw artifacts:** genuine experiment execution, byte identity of PLY regions/files, pixel identity, exact regenerated figures, and clean-runtime reproducibility.

`EXPERIMENT_PLAN.md:42-46` promised per-experiment environment, logs, assets, and sample renders. The repository commits only config/metrics/commit for storage runs and rounded image-metric JSONs for render runs. The notebook's default `COMMIT=''` follows branch tip (`notebooks/colab_runner.ipynb:13-18`), so it is not pinned by default. Stage 8 fresh-runtime Run-all remains recorded as pending/in progress (`STAGE_STATUS.yaml:1-18`; `DEVELOPMENT_LOG.md:77-84`).

## 11. Numerical and terminology discrepancies

1. **MB/MiB:** reported MB values are decimal; state that explicitly. 265.7 MB is 253.4 MiB, and 205.7 MB is 196.2 MiB.
2. **Duplicate asset sizes:** overlapping storage/render records differ by 1–41 bytes. Use a run-qualified field, not a single allegedly universal asset size.
3. **Scalar formula:** scale/offset term is wrong by 48× in `report/report_draft.md:97`.
4. **Raw vs compressed components:** component bytes are raw `.npy` payload sizes plus a 512-byte metadata estimate; total `.npz` bytes are compressed file sizes.
5. **Logical vs physical VQ bits:** K=64/128 logical indices need 6/7 bits but are stored as uint8.
6. **“~10 KB codebook+V”:** true near r8/K256, not r16/K256.
7. **“Byte-identical”:** the real-model check is float32 value equality, not byte comparison.
8. **“Pixel-identical”:** only a Markdown assertion is committed; no per-view machine record or images.
9. **Full-rank exactness:** 101.14 dB is extremely close but finite; serialized coefficient error is nonzero before rounding.
10. **FP16 generality:** rendered equivalence is recorded only at rank 8.
11. **34 runs:** 34 phase records, but only 22 unique configurations.
12. **Commit pin:** `aa39dcb` predates the final report and reproduction changes.

## 12. Claims requiring a fresh-runtime run

The following cannot be closed by static inspection:

- load the original checkpoint and independently confirm N, PLY bytes, field dtypes, and SH degree;
- regenerate the extract→reinsert PLY, compare SH and non-SH raw bytes or typed field buffers, and save hashes;
- render baseline and round-trip outputs, then hash all 38 image pairs;
- recompute SVD/error values from the original X;
- regenerate all 22 assets and compare total/member sizes and dtypes;
- rerun all 12 rendered configurations and recompute PSNR/SSIM/LPIPS from retained images;
- run pytest and record actual pass/skip/fail counts;
- regenerate every figure from a clean checkout at a pinned commit and compare hashes;
- complete the pending fresh-Colab `STAGE='all'` path with `COMMIT` explicitly set.

## 13. Recommended report corrections

1. Replace “every claim traceable to committed JSON/CSV” with “numeric SVD, storage, and rendered-metric tables are traceable to committed CSV/JSON; baseline and round-trip execution claims rely on stage records, and raw model/renders are not archived.”
2. Replace “non-SH fields byte-identical” with “non-SH float32 field values compared equal after write/read,” unless a true byte/hash comparison is added.
3. Mark “38 pixel-identical views” as an asserted run outcome pending committed per-view hashes/JSON.
4. State all displayed file sizes in decimal MB and optionally add MiB in parentheses.
5. Correct the scalar storage formula to `N·r·1 + 2·r·4 + 48·r·4 + overhead` for rank-16 8-bit/fp32-V.
6. Say “one uint8 index per Gaussian for K≤256” when discussing physical storage; reserve `ceil(log2K)` for logical information.
7. Qualify the 73.7 MB value as a calculated composite deployment size, not an emitted PLY or archived package.
8. Qualify FP16 render equivalence as demonstrated at rank 8; describe r4/r16 evidence as coefficient-space only.
9. Identify whether each table's byte column comes from `all_results` storage runs or `image_metrics` render runs, and explain the 1–41 byte differences.
10. Replace “all figures from committed result files” with a per-figure provenance statement; identify storage breakdown's asset dependency and visual comparison's uncommitted render inputs.
11. Describe “34 runs” as “22 storage records plus 12 render-phase records covering a subset of the same configuration names.”
12. Replace `aa39dcb` with the actual final report/reproduction commit if the summary intends to describe the current state, or state that `aa39dcb` is the stage-7 evidence boundary.

