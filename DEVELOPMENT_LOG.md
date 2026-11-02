# Development log

Oldest entries first. I kept this log in my working repository,
[adash963/3DGS](https://github.com/adash963/3DGS), until I moved the code here on
2 November 2026, so the commit hashes below refer to that repository.

## 2026-08-02 — Stage 0 started

- Created `dissertation` branch from `main` (54c035f, upstream graphdeco-inria code).
- Decision: use this repository itself as the 3DGS baseline (official implementation,
  Colab-proven via camenduru template that clones the same source). Avoids a second
  external dependency.
- Verified SH layout by inspecting `scene/gaussian_model.py` (save_ply/load_ply):
  PLY `f_dc_{c}`, `f_rest_{c*15+j}` channel-major; in-memory `[N,1,3]` + `[N,15,3]`.
- Decision: pretrained `train` scene via camenduru's `GaussianViewTest.zip` mirror
  (small, iteration-30000 checkpoint) instead of the 14 GB official model zip.
- Local env: macOS arm64, Python 3.12.5 venv at `../.venv` (no CUDA locally; all
  GPU work happens in Colab).
- Known risk: Colab free-tier session limits; mitigation: notebook is Run-all
  reproducible and stages checkpoint results into the repo/Drive.

## 2026-08-02 — Stage 1 failure #1: render.py cfg_args crash (FIXED)

- Symptom: `render.py` fails in `get_combined_args` → `eval(cfgfile_string)` →
  `SyntaxError: (unicode error) 'unicodeescape' codec ...`.
- Diagnosis: GaussianViewTest checkpoint's `cfg_args` was authored on Windows and
  contains `source_path='C:\Users\PC\Desktop\cloud\train'`; `\U` starts a unicode
  escape under eval(). Additionally `eval=False` means no test split, so
  `--skip_train` would render zero views.
- Fix: `scripts/fix_cfg_args.py` rewrites cfg_args with POSIX paths and `eval=True`
  (standard every-8th-camera test split); original kept as `cfg_args.orig`.
  Notebook cell 4 now calls it after unzip. Regression risk: none to core package
  (data preparation only).
- Caveat recorded for report: checkpoint was trained with eval=False (all cameras),
  so test views were seen during training → metrics measure compression degradation
  vs baseline, not novel-view generalisation.

## 2026-08-02 — Stage 1 failure #2: missing depths/train_test_exp (FIXED)

- Symptom: `AttributeError: 'GroupParams' object has no attribute 'depths'` in
  scene/__init__.py:44 after the cfg_args rewrite.
- Diagnosis: render.py constructs ModelParams with sentinel=True → every CLI default
  is None → merged args contain only cfg_args fields; the Oct-2024 codebase added
  `depths` and `train_test_exp`, absent from the 2023-era checkpoint cfg_args and
  from our first rewrite.
- Fix: fix_cfg_args.py now writes the complete ModelParams field set
  (depths='', train_test_exp=False).

## 2026-08-02 — Stages 1-3 passed; render-eval subset decision

- Stage 1: T4 runtime, rasteriser compiled, 38/38 baseline test views rendered
  (evidence: results/stages/stage_1_*). Fresh-runtime Run-all re-verification
  deferred to Stage 8 final reproduction (recorded risk).
- Stage 2: coefficient round trip exact (0.0 diff) AND render round trip bit-exact
  (38/38 views PSNR=inf, SSIM=1.0) — results/stages/stage_2_layout.md.
- Stage 3: all synthetic SVD tests pass (mac + Colab); real-scene effective rank 12
  (99% energy) — strong low-rank structure confirms the research premise.
- Pruning decision for GPU render evaluation (rule 18 / stage 7 pruning clause):
  VQ coefficient error is nearly rank-independent (codebook size is the binding
  constraint: 0.2265@r4k64 vs 0.2251@r16k64), so rendering all 9 VQ configs is
  uninformative. Render-eval subset (12): lr r1/r2/r4/r8/r16/r48 fp32, lr r8 fp16,
  sq r8/r16 (8-bit), vq r8k64, r8k256, r16k256. Full 22-config storage/coefficient
  table remains in results/experiments/ablation/.

## 2026-08-02 — Stages 4/5/6 render evaluation complete (Colab session 1)

- 12-config GPU eval succeeded, 0 failed views. Full-rank sanity anchor: 101.14 dB
  vs baseline. Headline: sq_r16_b8 = 15x SH compression at -0.62 dB GT PSNR,
  SSIM 0.992 vs baseline. VQ: 200-264x, codebook-limited (~21 dB GT).
- Metrics transferred Colab→repo via chunked base64 in notebook output (values
  rounded to 4dp in transit; raw originals remain in Colab session). First attempt
  with a single-line 55KB payload was truncated by Colab's output DOM — fixed by
  compacting (rounding) + 400-char line chunks.
- Colab auto-indent mangles multiline `python -c` cells → use single-line commands
  in ad-hoc cells (repo scripts remain the source of truth).
- Open items: LPIPS pass, visual comparison crops, raw render persistence (needs
  user-approved Drive mount), second scene (optional), fresh Run-all reproduction.

## 2026-08-02 — Stage 8 in progress

- Report draft v1 written (report/report_draft.md), all sections from committed
  artifacts; literature numbers labelled as authors' reported results.
- Notebook STAGE='all' reproduction path added with hard metric assertions.
- Fresh-runtime Run-all reproduction launched (runtime deleted, notebook reloaded
  from GitHub at ae2faa2's parent 95697f7); outcome to be recorded below.
- README: one-notebook reproduction instructions + troubleshooting section.

## 2026-08-02 — Final reproduction PASSED; mandatory project complete

- Fresh-runtime Run-all (repo 95697f7): 46/46 tests, baseline render, round-trip
  assertion, sq_r16_b8 reproduced with metrics identical to the original session
  to every printed decimal. Notebook printed "STAGE all PASS".
- Closes stage-1 deferred gate, stage-7 reproduction rider, stage-8 reproduction
  requirement. Evidence: results/final_reproduction/reproduction_report.md.
- Last known-good commit for the complete mandatory project: this commit.

## 2026-11-01 — Repository synced with the final thesis

- Checked the branch against the final thesis (October 2026): all 22 storage
  configurations, the 12 rendered metric files and the 50 tests (46 unit, 4
  integration; all pass) match the thesis values.
- README project section: thesis title, key-results table, cost, caveats
  (eval=False checkpoint, non-reference SSIM/LPIPS, one scene).
- report/: added README.md marking the August drafts as superseded by the thesis.
- Open item: scripts/compare_sh_baselines.py and
  results/experiments/sh_baselines/sh_baselines.csv (thesis Section 3.5, Table 5.8)
  are referenced by the thesis but not yet committed. (Closed 2026-11-02, see below.)

## 2026-11-01 — Colab runner fixed for the current Colab image; full-training stage

- Colab now ships Python 3.13, torch 2.11+cu130, CUDA 13 and GCC 13. The CUDA
  submodules failed to build: pip build isolation hid torch from setup.py, and
  rasterizer_impl.h uses uint32_t/uintptr_t without <cstdint>.
- notebooks/colab_runner.ipynb cell 3: install the submodules with
  --no-build-isolation and add <cstdint> (rasteriser) and <float.h> (simple-knn)
  before compiling. Upstream source in the repo is unchanged; the patch is applied
  in the Colab checkout only.
- New cell 7 / TRAIN_ITERATIONS = 30000: trains a fresh checkpoint for the full
  schedule with --eval (held-out test views), renders it and re-runs sq_r16_b8 on it.
  Extension beyond the thesis, reported separately.

## 2026-11-01 — Colab reproduction PASSED on the new image; held-out training check

- Fresh Colab T4 runtime (Python 3.13.16, torch 2.11.0+cu130), commit 5754d3d:
  46/46 unit tests, model N = 1,071,462, SH round trip exact, sq_r16_b8 reproduced
  to the printed decimals (34.632 dB / SSIM 0.9919 vs base, 24.312 dB vs GT).
  "STAGE all PASS".
- Cell 7, 30,000 iterations with --eval (~55 min): N = 1,089,609, held-out baseline
  22.06 dB / 0.818; sq_r16_b8 13,471,247 B (15.5×, whole 3.63×), 34.93 dB / 0.9926 vs
  base, 21.77 dB vs held-out GT (Δ −0.29 dB, SSIM −0.004).
  Evidence: results/experiments/trained_heldout/summary.json (values from cell output).
- Cell 7 now also scores the trained baseline against GT and writes
  /content/work/repro_trained/summary.json with the deltas.

## 2026-11-02 — Same-checkpoint comparison and thesis figures committed; docs aligned

- Committed scripts/compare_sh_baselines.py and its outputs in
  results/experiments/sh_baselines/ (CSV, JSON, run log), copied unchanged from the
  thesis working folder (run of 29 Oct 2026). They match Tables 5.4, 5.7 and 5.8, the
  proxy column of Table 5.6 and the proxy validation (Pearson 0.993, Spearman 0.991,
  11 configurations). Closes the open item of 2026-11-01.
- scripts/figures/: make_diagrams.py (Figures 3.1-3.3) and make_charts.py (Figures 5.2,
  5.3) from the thesis working folder, with repo-relative paths; new
  make_analysis_figures.py for Figures 4.1-4.3 and 5.1, which had no script. Output:
  results/figures/thesis/, named by figure number. Figure 5.1 no longer overlaps the
  "uncompressed" and "rank 16, 8-bit" labels; 4.2 and 4.3 use the same style as the
  other figures. Data and values are unchanged.
- README, results/README.md: thesis table/figure → file → script map; Colab
  instructions moved from the colab-runner branch into COLAB.md on this branch.
- Notebook: badge, note that the thesis numbers come from COMMIT='95697f7' (later
  commits add files without changing src/ or the existing scripts), and an opt-in
  STAGE='baselines' for Table 3.7 step 11 (not yet run on Colab).
- requirements-cpu.txt pins the CPU versions of thesis Table 3.8; CITATION.cff added.
- No change to src/, tests/ or the existing scripts; 50 tests pass.

## 2026-11-02 — Repository tidied into its final layout

- Removed files the thesis superseded or that only tracked the development stages:
  AGENTS.md, dash.md, REQUIREMENTS.md, EXPERIMENT_PLAN.md, SUCCESS_CRITERIA.md,
  STAGE_STATUS.yaml, STATIC_AUDIT.md, STATIC_AUDIT_DISCREPANCIES.md,
  STATIC_AUDIT_FINDINGS.json, report/ (August drafts), results/stages/stage_0, 3, 4,
  5, 6 and 7 reports, and the ten working plots in results/figures/ that duplicate the
  thesis figures. All of them remain in the git history (last present in 6e01491).
- Kept the evidence the thesis relies on: results/stages/stage_1_baseline.md,
  stage_1_environment.json, stage_2_layout.md (the 38/38 pixel-identical round trip)
  and stage_2_roundtrip_metrics.json; results/final_reproduction/; this log.
- code_explanation/ moved to docs/code_explanation/, with walkthroughs added for
  compare_sh_baselines.py and the three scripts in scripts/figures/; references to the
  removed files now point to files that remain.
- .gitignore: working plots regenerated by analyse_rank.py and generate_report_assets.py
  are ignored; visual_comparisons.png (Figure B.5) stays tracked.

## 2026-11-02 — Aligned with the updated thesis draft

- The draft now reports the held-out check in the thesis itself (Section 3.6.2, Table 3.7
  step 13). README, COLAB.md, results/README.md and notebook cell 7 no longer call it
  "after the thesis"; the 2026-11-01 entries above are left as written.
- Notebook cell 4 checks the sizes and SHA-256 of tandt_db.zip and GaussianViewTest.zip
  against Appendix B.2 (Table 3.7 step 3) and stops on a mismatch. The values, also in
  data/README.md, match the hashes Hugging Face publishes for both files.
- results/README.md maps Section 3.6.2 and Appendix B.2, and records the per-view spread
  of Section 5.4.1 as a derived number (recomputed: mean loss 0.618 dB, SD 0.563 dB,
  95 % CI ±0.185 dB, range 0.098–2.989 dB over 38 views).
- For the final repository: tag thesis-v1.0 (Section 3.6, Appendix B.1); the notebook,
  COLAB.md and CITATION.cff must point at 3DGS_LRF_SH_Compression; branches main,
  dissertation, vanilla-3DGS and colab_run as listed in thesis Table B.1.

## 2026-11-02 — Moved to 3DGS_LRF_SH_Compression

- Copied the files of the working repository into this one (dissertation branch).
  The copy had missed .gitignore, .gitmodules and the four submodule links
  (SIBR_viewers, diff-gaussian-rasterization, fused-ssim, simple-knn), which the Colab
  build needs. Restored them at the same upstream commits, and stopped tracking .DS_Store.
- Notebook, COLAB.md and CITATION.cff now point at 3DGS_LRF_SH_Compression. The notebook
  clones `main` into /content/3DGS_LRF_SH_Compression; COMMIT = 'thesis-v1.0' pins the
  version described in the thesis.
- README rewritten in my own words; the research-question table now has two columns.
