# `metrics.py` (upstream)

## Purpose

Upstream's own evaluation script: reads a model's `renders/` and `gt/` directories and
reports PSNR, SSIM and LPIPS.

**This project does not use it.** We wrote our own
(`src/sh_compression/metrics.py`), and this document explains why — a question a
marker could reasonably ask.

## Line-by-line

### Lines 12–22 — Imports

```python
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
```

Uses the codebase's own metric implementations: `ssim` from
[loss_utils.md](loss_utils.md), `psnr` from [image_utils.md](image_utils.md), and the
bundled `lpipsPyTorch`.

### Lines 24–35 — `readImages`

```python
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names
```

Loads every render and its same-named ground truth onto the GPU.

**Two properties here are why we did not reuse this function.**

1. **`os.listdir` is unsorted** (line 28). Filesystem order is arbitrary, so per-view
   results come back in an unpredictable order. Aggregate means are unaffected, but
   per-view records — which our evaluation retains and the audit later verified — would
   not be reliably comparable across runs. Our `compare_dirs` sorts explicitly.

2. **Everything is loaded onto the GPU at once** (lines 31–32). For 38 views this is
   fine; it is also unnecessary, and it makes the function unusable on a CPU-only
   machine. Our metrics run in NumPy, so the same code path works locally and in
   Colab.

There is also no error handling: a missing or corrupt file raises and aborts the whole
evaluation. Ours records the failure per view and continues, which is what makes
"0 failed views" a positive claim rather than an absence of evidence.

### Lines 37 onward — `evaluate`

```python
def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    ...
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            ...
```

Iterates over model directories, and within each, over method subdirectories
(`ours_30000`, etc.), computing mean PSNR/SSIM/LPIPS and writing `results.json` and
`per_view.json` into the model directory.

The structure assumes upstream's directory conventions and writes results **inside the
model directory** — which for us would mean writing into the baseline checkpoint we
deliberately never modify.

## Why we wrote our own

Four reasons, all worth stating in the report:

1. **Two reference frames.** Upstream compares renders against ground truth only. Our
   evaluation additionally compares **compressed renders against baseline renders**,
   which isolates compression loss from the model's own reconstruction error. That
   comparison is the dissertation's primary evidence, and this script cannot produce
   it.

2. **Deterministic ordering and per-view retention.** Sorted filenames, filename-based
   pairing, and every per-view row preserved — the properties that let the static
   audit re-derive our summary means from stored data.

3. **Failure recording.** Failures become entries in `failed_views` rather than
   aborting the run.

4. **CPU portability and float64 accumulation.** Our metrics run without CUDA and
   accumulate in float64, which matters in the high-PSNR regime where our headline
   measurements sit.

**The implementations remain comparable.** Our PSNR is algebraically identical to
upstream's (see [image_utils.md](image_utils.md)); our SSIM uses scikit-image with the
same standard constants and window as upstream's. So a reader can compare our absolute
numbers against 3DGS paper figures without worrying that a different metric definition
is responsible for any difference.

## Why it matters

For the report's methodology section: the choice not to reuse upstream's evaluation was
deliberate and motivated, not incidental. The dual-reference comparison in particular
is a methodological improvement over what this script offers, and it is the reason the
dissertation can separate "how much did compression cost" from "how good is the
underlying model".
