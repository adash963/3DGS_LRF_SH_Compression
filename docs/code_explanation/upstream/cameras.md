# `scene/cameras.py`

## Purpose

The `Camera` class — holds a view's pose, field of view, ground-truth image, and the
precomputed matrices the rasteriser needs. One instance per evaluation view; 38 of
them exist during each of our renders.

## Line-by-line

### Lines 12–17 — Imports and class

```python
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
```

Subclassing `nn.Module` is mostly vestigial — cameras hold no trainable parameters in
our path — but it provides convenient device management.

### Lines 20–33 — Constructor and pose

```python
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image,
                 invdepthmap, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
```

Pose (`R`, `T`) and field of view come from the dataset reader. `image_name` preserves
the original filename — worth noting, because `render.py` does **not** use it for
output naming, preferring the index. That choice is what makes our filename-based
matching correspond to camera identity.

### Lines 35–41 — Device selection

```python
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")
```

`data_device` comes from config — `"cuda"` in ours. Images are stored on GPU, which is
why loading 301 cameras consumes noticeable VRAM and takes several seconds (the
"Loading Training Cameras" pause in every log).

### Lines 43–50 — Ground-truth image

```python
        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))
```

Line 43 loads and normalises to `[0,1]` in `[C,H,W]` layout (see
[general_utils.md](general_utils.md)). Line 44 keeps RGB; lines 46–50 build an alpha
mask, defaulting to all-ones when the image has no alpha channel — which is the case
for our JPEG inputs.

### Lines 52–56 — Exposure-mode masking (disabled for us)

```python
        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0
```

Under exposure training, half of each test image is masked out — the left half used
for fitting exposure, the right for evaluation.

**Disabled in our configuration** (`train_test_exp=False`). Had it been enabled, our
metrics would have been computed on half-images, and the numbers would not be
comparable to standard 3DGS evaluation. This is one of the reasons `fix_cfg_args.py`
writes that field explicitly rather than leaving it to chance.

### Lines 58–60 — Image storage and dimensions

```python
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
```

`original_image` is what `render.py` writes to the `gt/` directory — so our
ground-truth comparison images come from here, already clamped to `[0,1]` and at the
same resolution as the render.

Dimensions are taken from the image itself, then passed to the rasteriser as
`image_height`/`image_width`. Renders therefore always match ground truth in size,
which is why our metric shape check never fires in practice.

### Later lines — Matrices

The constructor goes on to build:

```python
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar,
                                                     fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(
                                    self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
```

- `world_view_transform` — world to camera, from
  [graphics_utils.md](graphics_utils.md). The `.transpose(0,1)` converts to the
  row-vector convention the CUDA kernel expects.
- `projection_matrix` — the perspective projection.
- `full_proj_transform` — their product, passed to the rasteriser as `projmatrix`.
- **`camera_center`** — extracted by inverting the view matrix. This is passed to the
  rasteriser as `campos` and is what the kernel uses to compute per-Gaussian viewing
  directions for **SH evaluation**.

That last one is the direct link to this project: `camera_center` determines the
direction vector fed to the SH basis functions documented in
[sh_utils.md](sh_utils.md), which is what turns our compressed coefficients into
pixels.

## Why it matters

Every quantity here derives from the dataset and the camera pose — **never from the
model**. Two consequences for the dissertation:

1. **Cameras are identical across configurations by construction.** The same 38
   `Camera` objects, the same matrices, the same `camera_center`, whether rendering
   the baseline or a 264×-compressed variant.
2. **Ground truth is fixed and shared.** `original_image` is written out alongside
   each render at matching resolution, so every comparison uses the same reference
   pixels.

Together these remove camera and reference variability from the experiment entirely,
which is what allows measured differences to be attributed to compression alone.
