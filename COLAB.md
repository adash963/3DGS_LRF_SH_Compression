# Running the thesis pipeline on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adash963/3DGS/blob/dissertation/notebooks/colab_runner.ipynb)

The notebook clones this repository on the `dissertation` branch (cell 1: `REPO_URL`,
`BRANCH`, `COMMIT`). The thesis numbers were produced at commit `95697f7`; set
`COMMIT = '95697f7'` to run exactly that code. Later commits add files but do not change
`src/` or the scripts that produced the numbers.

1. Open the badge link, sign in to Google.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all**, then **Run anyway** on the "not authored by Google" warning.

| Cell | What it does | Time on a T4 | Thesis Table 3.7 |
|---|---|---|---|
| 1–3 | GPU check, clone, install, compile the CUDA rasteriser (with the header fixes for Python 3.13 / torch 2.11 / CUDA 13) | ~5 min | steps 1–2 |
| 4–5 | Download the pretrained Tanks and Temples *train* checkpoint and images, check their sizes and SHA-256 against Appendix B.2, rewrite `cfg_args`, 46 unit tests, model inspection | ~2 min | steps 3–4 |
| 6 | Baseline render, SH round trip, `sq_r16_b8` end to end with assertions → `STAGE all PASS` | ~5 min | steps 5–6 |
| 7 | Train a new checkpoint for 30,000 iterations with held-out views, render it, re-run `sq_r16_b8`, write `/content/work/repro_trained/summary.json` | ~60 min | step 13 (Section 3.6.2) |

Parameters (cell 1):

- `STAGE = 'all'` runs cells 6 and 7; `stage1`, `stage2`, `stage4`, `stage5` and `stage7`
  run one stage each.
- `STAGE = 'baselines'` rebuilds the 22 compressed assets and runs the same-checkpoint
  comparison (Table 3.7 steps 8 and 11) on the Colab CPU, then shows how the results differ
  from the committed ones. It has not been run on Colab yet; the K-means configurations
  take most of the time, so expect well over an hour.
- `TRAIN_ITERATIONS = 0` skips cell 7.

Download `/content/work/repro_trained/` from the Files panel before the runtime ends.

Expected results: cell 6 — 34.632 dB / SSIM 0.9919 against the uncompressed render,
24.312 dB against GT, 13,670,820 B; cell 7 — 34.93 dB / 0.9926 against its uncompressed
render, Δ −0.29 dB against the held-out photographs
(see `results/experiments/trained_heldout/summary.json`).
