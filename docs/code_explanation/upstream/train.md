# `train.py`

## Purpose

The optimisation loop that produces a 3DGS model from posed images. 285 lines, the
largest single script upstream.

**This project never runs it.** We load a pretrained checkpoint. Documenting it
matters anyway, because it explains *how the coefficients we compress came to be* —
and several of its design choices corroborate our findings.

## Structure

### Lines 12–40 — Imports and setup

```python
import os, torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False
```

Note it imports `ModelParams` **without** `sentinel=True` — unlike `render.py`. This
is why training has working command-line defaults while rendering requires a complete
`cfg_args`, the asymmetry behind our Stage 1 bug.

### Lines ~40–70 — Initialisation

```python
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
```

Creates the model and calls `create_from_pcd` (via `Scene`, since no checkpoint
exists), initialising Gaussians from the COLMAP sparse point cloud.

**Recall from [gaussian_model.md](gaussian_model.md): all higher-order SH coefficients
start at exactly zero.** Only the DC term is initialised, from point colour.

### The training loop

The loop body, roughly:

```python
    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, ...)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()
        ...
```

**The SH degree schedule is directly relevant to this dissertation.**
`oneupSHdegree()` raises `active_sh_degree` by one every 1000 iterations, capped at
`max_sh_degree = 3`. So training is coarse-to-fine in the *angular* dimension:

- iterations 0–999: degree 0 only — flat colour, no view dependence
- 1000–1999: degree 1 added
- 2000–2999: degree 2 added
- 3000+: full degree 3

Higher-order coefficients therefore spend the first 3000 iterations at exactly zero
and are only refined for the remaining 27,000 — at 1/20th the DC learning rate
(`feature_lr / 20.0`).

The loss (`0.8 × L1 + 0.2 × (1 − SSIM)`) is worth noting for the report: **the model
was never optimised for LPIPS**, which contextualises the baseline's LPIPS of 0.142
against ground truth.

### Densification

```python
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(...)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                                scene.cameras_extent, size_threshold, radii)
```

Between iterations 500 and 15,000, Gaussians are split, cloned and pruned based on
screen-space gradients. **After iteration 15,000 the population is frozen** — which is
why our checkpoint has a fixed 1,071,462 Gaussians, and why "same geometry across all
configurations" is a stable property of the checkpoint rather than something we
enforce.

### Saving

```python
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
```

Calls `Scene.save` → `gaussian_model.save_ply`, writing
`point_cloud/iteration_30000/point_cloud.ply` — **the file our entire project reads**.

## Why it matters

Three ways this file supports the dissertation, despite never being executed.

**1. It explains why the SH matrix is low-rank.** Three design choices push in the same
direction: higher-order coefficients initialise at zero, are introduced only after 3000
iterations, and train at 1/20th the DC learning rate. The optimisation treats
view-dependent detail as a small correction on a dominant base colour — exactly the
structure that yields effective rank 12 of 48, with rank 1 alone capturing 90% of
energy.

**2. It bounds what our results can claim.** The model was optimised for L1 + SSIM at
full precision, with no awareness that its coefficients would later be compressed.
That is what makes our approach genuinely *post-hoc*, and it is the honest reason our
extreme-compression results fall short of published systems that fine-tune after
quantising — a point the report makes in its discussion rather than leaving implicit.

**3. It defines the artefact we consume.** The 30,000-iteration checkpoint with its
frozen population and full degree-3 SH is the fixed input to every experiment; nothing
in our pipeline could change it even in principle, since we never invoke this script.
