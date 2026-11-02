# Colab notebooks for my thesis

This folder holds the Google Colab notebooks for my MSc thesis, *Low-Rank Factorisation and
Quantisation of Spherical Harmonics for Storage-Efficient 3D Gaussian Splatting*. The code,
results and documentation are on the `main` branch of
[adash963/3DGS_LRF_SH_Compression](https://github.com/adash963/3DGS_LRF_SH_Compression).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adash963/3DGS_LRF_SH_Compression/blob/colab_run/colab_run/gaussian_splatting_colab.ipynb)

`gaussian_splatting_colab.ipynb` runs the thesis pipeline. It clones `main` and runs the same
cells as `notebooks/colab_runner.ipynb` there. Open it with a **T4 GPU** runtime and choose
**Run all**. It prints `STAGE all PASS` once the headline result (rank 16 + 8-bit, 13.7 MB of
SH, above 34 dB against the uncompressed render) is reproduced. Steps, timings and expected
numbers are in [COLAB.md](https://github.com/adash963/3DGS_LRF_SH_Compression/blob/main/COLAB.md).
To run exactly the version described in the thesis, set `COMMIT = 'thesis-v1.0'` in the first
code cell.

`gaussian_splatting_viewer_colab.ipynb` is camenduru's web viewer, unchanged: drag and drop a
`point_cloud.ply` into it.

I started these notebooks from camenduru's
[gaussian-splatting-colab](https://github.com/camenduru/gaussian-splatting-colab), whose
README follows.

---

🐣 Please follow me for new updates https://twitter.com/camenduru <br />
🔥 Please join our discord server https://discord.gg/k5BwmmvJJU <br />
🥳 Please join my patreon community https://patreon.com/camenduru <br />

## 🦒 Colab

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/gaussian-splatting-colab/blob/main/gaussian_splatting_colab.ipynb) | gaussian_splatting_colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YniEH63VfZPuRGTddviUvNH48cDaLqtg) | Clip Guided Gaussian Splatting Colab (thanks to [@aman_gif](https://twitter.com/aman_gif) ❤)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/gaussian-splatting-colab/blob/main/gaussian_splatting_viewer_colab.ipynb) | gaussian_splatting_viewer_colab (thanks to [@antimatter15](https://twitter.com/antimatter15) ❤)

## Viewer Tutorial
We can drag and drop our `point_cloud.ply` 🥳

## Main Repo
https://github.com/graphdeco-inria/gaussian-splatting

## Page
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

## Paper
https://arxiv.org/abs/2308.04079

## Main Repo Viewer
https://github.com/antimatter15/splat thanks to @antimatter15 ❤

## Output
https://github.com/camenduru/gaussian-splatting-colab/assets/54370274/003ee472-6c1d-44e0-990a-6c795effe46f

https://github.com/camenduru/gaussian-splatting-colab/assets/54370274/e6d1674f-1ba5-4f46-aa30-be6e54dea45e
