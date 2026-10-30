# Development log

Newest entries at the top.

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
