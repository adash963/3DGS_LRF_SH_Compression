# `gaussian_renderer/__init__.py`

## Purpose

The bridge between the Python model and the CUDA rasteriser. One function, `render`,
which packages a `GaussianModel` and a camera into the arguments the compiled
rasteriser expects and returns the rendered image.

**This is the code path every configuration in the dissertation goes through** —
baseline, round-trip, low-rank, scalar, and VQ alike, entirely unmodified.

## Line-by-line

### Lines 12–18 — Imports and signature

```python
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,
           scaling_modifier = 1.0, separate_sh = False, override_color = None,
           use_trained_exp=False):
```

Line 14 imports the **compiled CUDA extension** — the module built during Colab setup
(`pip install ./submodules/diff-gaussian-rasterization`). This is the differentiable
rasteriser from the 3DGS paper.

### Lines 25–30 — Screen-space gradient placeholder

```python
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
```

A zero tensor whose gradient captures 2-D screen-space positions during training —
used by the densification logic to decide where to add Gaussians.

Irrelevant to this project (we never train), but it runs anyway. The `+ 0` forces a
new node in the autograd graph so `retain_grad` has something to attach to; the bare
`except` handles inference mode where gradients are disabled.

### Lines 32–52 — Rasteriser configuration

```python
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
```

Everything the CUDA kernel needs: image dimensions, field of view as tangent
half-angles, background colour, and the view and projection matrices.

**Two settings matter for the dissertation's fairness claim:**

- `sh_degree=pc.active_sh_degree` — tells the kernel how many SH coefficients to
  evaluate. Always 3 for our models, identical across every configuration.
- `campos=viewpoint_camera.camera_center` — the camera position, needed to compute
  per-Gaussian viewing directions for SH evaluation.

Because these settings derive entirely from the camera and the model's declared
degree, they are identical for baseline and compressed renders. Nothing in the
render configuration varies between configurations.

### Lines 54–68 — Geometry

```python
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
```

Positions, opacities, scales and rotations — all read through the model's activated
properties (`get_scaling` applies `exp`, `get_opacity` applies `sigmoid`).

**These are exactly the fields our `insert_sh` leaves untouched.** The values passed
here are bit-identical between a baseline render and any compressed render, which is
the mechanism behind "same geometry, different colour".

### Lines 70–87 — Colour: the SH path

```python
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
```

**The colour branch — where this project's compressed data enters the renderer.**

Two paths exist:

- **Python path** (lines 75–80, when `convert_SHs_python` is set): evaluates SH in
  PyTorch using `eval_sh`. Line 76's `transpose(1,2).view(-1, 3, 16)` produces
  exactly the `[N, 3, 16]` canonical shape our `sh_layout` uses — independent
  confirmation of the layout. Lines 77–78 compute per-Gaussian viewing directions
  (Gaussian position minus camera position, normalised). Line 80 adds 0.5 and clamps,
  matching `SH2RGB`.
- **CUDA path** (lines 82–85, the default and what we use): passes raw coefficients
  to the kernel, which evaluates SH on the GPU.

Our pipeline uses the CUDA path with `separate_sh` handling, so the reconstructed
coefficients are consumed directly by the rasteriser — no Python-side colour
processing that could differ between configurations.

### Lines 89–110 — The rasterisation call

```python
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D, means2D = means2D,
            dc = dc, shs = shs, colors_precomp = colors_precomp,
            opacities = opacity, scales = scales, rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D, means2D = means2D,
            shs = shs, colors_precomp = colors_precomp,
            opacities = opacity, scales = scales, rotations = rotations,
            cov3D_precomp = cov3D_precomp)
```

The actual CUDA call. Inside the kernel: project each 3-D Gaussian to a 2-D screen
ellipse, sort by depth, evaluate SH for the viewing direction, and alpha-composite
front to back.

Two call signatures depending on whether DC and rest coefficients are passed
separately — a performance optimisation in the newer rasteriser, mathematically
equivalent to the combined form.

Returns the image, per-Gaussian screen radii (zero for culled Gaussians), and a depth
map.

### Lines 112–128 — Post-processing and return

```python
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]

    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    return out
```

Optional per-image exposure compensation — **disabled in our configuration**
(`train_test_exp=False` in the `cfg_args` our `fix_cfg_args.py` writes). Leaving it
off is important for comparability: a per-image colour transform would confound the
compression measurement.

Line 119 clamps to `[0, 1]`, which is why our `load_image` heuristic works and why no
render ever contains out-of-range values.

## Why it matters

This file is the reason the dissertation can claim a fair comparison.

Every configuration passes through this identical function with identical settings.
The *only* difference between a baseline render and a compressed render is the
numerical content of `pc.get_features` — the SH coefficients. Geometry, opacity,
camera, projection, background, degree, and post-processing are all bit-identical.

That isolation is what allows any measured quality difference to be attributed to the
compression rather than to a rendering difference — and it is achieved by *not
modifying this file*, which is why "unmodified baseline renderer" appears throughout
the report.
