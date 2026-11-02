# Running the thesis pipeline on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adash963/3DGS_LRF_SH_Compression/blob/main/notebooks/colab_runner.ipynb)

I ran every GPU step of the thesis with this notebook. It clones this repository using the
parameters in its first code cell (`REPO_URL`, `BRANCH`, `COMMIT`) and, by default, runs the
tip of `main`. To run exactly the version described in the thesis, set
`COMMIT = 'thesis-v1.0'`.

1. Open the badge link and sign in to Google.
2. Go to **Runtime → Change runtime type**, choose **T4 GPU** and save.
3. Click **Runtime → Run all**. Colab warns that the notebook was not written by Google;
   click **Run anyway**.

| Cell | What it does | Time on a T4 | Thesis Table 3.7 |
|---|---|---|---|
| 1–3 | GPU check, clone, install, compile the CUDA rasteriser (with the header fixes for Python 3.13 / torch 2.11 / CUDA 13) | ~5 min | steps 1–2 |
| 4–5 | Download the pretrained Tanks and Temples *train* checkpoint and images, check their sizes and SHA-256 against Appendix B.2, rewrite `cfg_args`, 46 unit tests, model inspection | ~2 min | steps 3–4 |
| 6 | Baseline render, SH round trip, `sq_r16_b8` end to end with assertions → `STAGE all PASS` | ~5 min | steps 5–6 |
| 7 | Train a new checkpoint for 30,000 iterations with held-out views, render it, re-run `sq_r16_b8`, write `/content/work/repro_trained/summary.json` | ~60 min | step 13 (Section 3.6.2) |

Parameters:

- `STAGE = 'all'` runs cells 6 and 7. `stage1`, `stage2`, `stage4`, `stage5` and `stage7`
  run a single stage, and `train` runs only cell 7.
- `STAGE = 'baselines'` rebuilds the 22 compressed assets and runs the same-checkpoint
  comparison (Table 3.7 steps 8 and 11) on the Colab CPU, then shows any difference from
  the committed results. I have not run this one on Colab yet. The K-means configurations
  take most of the time, so expect well over an hour.
- `TRAIN_ITERATIONS = 0` skips cell 7.
- `BRANCH = 'dissertation'` runs the development branch instead of `main`.

Download `/content/work/repro_trained/` from the Files panel before the runtime ends,
otherwise the training results are lost.

What you should see: cell 6 gives 34.632 dB / SSIM 0.9919 against the uncompressed render,
24.312 dB against GT and a 13,670,820-byte file. Cell 7 gives 34.93 dB / 0.9926 against its
uncompressed render and Δ −0.29 dB against the held-out photographs
(see `results/experiments/trained_heldout/summary.json`).
