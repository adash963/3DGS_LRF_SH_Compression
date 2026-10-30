# `render.py`

## Purpose

The rendering entry point. Loads a trained model, renders the train and/or test camera
sets, and writes PNGs alongside their ground-truth counterparts.

**Every image in this dissertation was produced by this script**, invoked identically
for every configuration. It is completely unmodified.

## Line-by-line

### Lines 12–27 — Imports

```python
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
```

Lines 23–27 probe for the sparse-Adam optimiser. Despite the name, the flag is reused
as the `separate_sh` argument (see line 77) — if the newer rasteriser is installed,
DC and rest coefficients are passed separately. Our Colab environment has it, so our
renders take the `separate_sh` path.

### Lines 30–46 — `render_set`

```python
def render_set(model_path, name, iteration, views, gaussians, pipeline, background,
               train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
```

**Lines 31–32 define the output paths** our pipeline depends on:

```
<model>/test/ours_30000/renders/    rendered images
<model>/test/ours_30000/gt/         ground-truth images
```

Our `render_model.py` reads from the first, and `run_render_eval.py` uses both as
comparison references. The `ours_{iteration}` naming is why our staging directory
must place the PLY under `point_cloud/iteration_30000/`.

**That the ground truth is written here matters**: the same script emits renders and
their references, at the same resolution, in the same order — so the comparison never
depends on separately locating and matching source photographs.

```python
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background,
                           use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]
```

Line 38 calls into `gaussian_renderer.render` (documented in
[gaussian_renderer.md](gaussian_renderer.md)) and keeps only the `"render"` key.

Line 39 takes the first three channels of the ground truth, dropping alpha if present.

```python
        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
```

Lines 41–43 crop to the right half when exposure training is enabled. **This is
disabled in our configuration** (`train_test_exp=False` in the `cfg_args` our
`fix_cfg_args.py` writes), so our renders are full-frame. Had it been enabled, all
metrics would have been computed on half-images — a silent and serious difference.

```python
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
```

**Lines 45–46 are why our filename-based matching works.** Images are named by
zero-padded *index* — `00000.png` through `00037.png` — not by source filename.

Since the camera list order is deterministic (shuffle is disabled, see below), index
`k` refers to the same camera in every render run. Our `compare_dirs` matches images
by filename, which therefore matches them by camera — the property that makes
per-view comparison valid across configurations.

### Lines 48–60 — `render_sets`

```python
def render_sets(dataset, iteration, pipeline, skip_train, skip_test, separate_sh):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
```

- Line 49: `torch.no_grad()` — inference only, no autograd graph.
- Line 50 constructs the model with the degree from config (3 for our checkpoint).
- **Line 51: `shuffle=False`** — critical for us. Camera order is deterministic, so
  index-based filenames are stable across runs and configurations.
- Lines 53–54: background is black, since `white_background=False` in our config.
  Identical for every configuration, so it cannot confound comparisons.

```python
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), ...)
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), ...)
```

Lines 56–60 render each split. We always pass `--skip_train`, so only the test set
renders — 38 views rather than 301, which is why a render takes ~17 seconds.

**This is where the `eval=True` fix pays off.** With the checkpoint's original
`eval=False`, `getTestCameras()` returns an empty list, `render_set` loops zero times,
and the script exits successfully having produced nothing. That silent no-op is the
third problem `fix_cfg_args.py` solves, and the reason our `render_model.py`
explicitly checks that renders appeared.

### Lines 62–77 — Entry point

```python
if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
```

**Line 65 is the origin of the second Stage 1 bug.** `ModelParams(parser,
sentinel=True)` sets every model-related command-line default to `None`. The merge
logic in `get_combined_args` then keeps only non-`None` values, which means **all**
model configuration must come from the `cfg_args` file.

That is why a `cfg_args` missing `depths` or `train_test_exp` produces
`AttributeError: 'GroupParams' object has no attribute 'depths'` rather than falling
back to a default — there are no defaults. Our `fix_cfg_args.py` writes the complete
field set precisely because of this line.

`--iteration` defaults to −1, meaning "use the highest available", which resolves to
30000 for our checkpoint.

```python
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
```

`safe_state` seeds the RNG (fixed seed 0) and sets the CUDA device. Rendering is
deterministic anyway, but the fixed seed removes any doubt — which is consistent with
our reproduction run producing metrics identical to the original to every printed
decimal.

Line 77 passes `SPARSE_ADAM_AVAILABLE` as `separate_sh`, as noted above.

## Why it matters

Three properties of this script underpin the dissertation's evaluation protocol, all
obtained for free by not modifying it:

1. **Identical rendering for every configuration.** Same script, same arguments, same
   background, same cameras — only the PLY contents differ.
2. **Deterministic view ordering** (`shuffle=False`) plus index-based filenames, which
   is what makes filename-matched per-view comparison correct.
3. **Ground truth emitted alongside renders**, at matching resolution and ordering,
   removing an entire class of misalignment error from the evaluation.

It is also the script whose failures drove both Stage 1 bugs — the `eval()` on
`cfg_args` and the sentinel parser — and the reason `scripts/fix_cfg_args.py` exists.
