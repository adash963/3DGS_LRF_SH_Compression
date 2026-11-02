# `utils/camera_utils.py`

## Purpose

Constructs `Camera` objects from `CameraInfo` records, handling image loading and —
importantly for us — **resolution selection**. One line in this file silently
determines the resolution at which every metric in the dissertation was computed.

## Line-by-line

### Lines 12–19 — Imports and state

```python
from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2

WARNED = False
```

`WARNED` is a module-level flag so the large-image warning prints once rather than 301
times.

### Lines 21–40 — Image and depth loading

```python
def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    image = Image.open(cam_info.image_path)

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)
        except FileNotFoundError:
            ...
    else:
        invdepthmap = None
```

Line 22 opens the ground-truth photograph. Lines 24–40 load an inverse depth map for
depth regularisation — **skipped entirely for us**, since our `cfg_args` sets
`depths=''`, making `cam_info.depth_path` empty.

That is worth noting: the `depths` field whose absence caused our Stage 1 crash ends
up unused. It had to be present for the config merge to succeed, but its value being
empty is what we want.

### Lines 42–60 — Resolution logic

```python
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
```

**This block determines the resolution of every image in the dissertation.**

Our config sets `resolution = -1`, which takes the `else` branch at line 46:

- If the source image is **wider than 1600 pixels**, it is downscaled so width becomes
  1600 (line 53), and the informational warning prints once.
- Otherwise it is used at native resolution (line 55).

Two consequences worth stating precisely in the report:

1. **Metrics are computed at whatever resolution this produces**, not at the source
   photographs' native resolution. For T&T `train`, source images exceed 1600 px wide,
   so renders are the downscaled version. Both render and ground truth pass through
   the same path, so the comparison is internally consistent — but a reader comparing
   our absolute PSNR against another paper's should know the resolution convention.

2. **It is identical for every configuration.** `resolution` comes from config, not
   from the model, so baseline and compressed renders are always the same size. This
   is what makes our metric shape check never fire.

The `resolution_scale` parameter supports multi-resolution training; `Scene` passes
`[1.0]`, so it has no effect here.

### Remaining lines — `cameraList_from_camInfos` and `camera_to_JSON`

```python
def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))
    return camera_list
```

A straightforward loop preserving input order — which is what carries the dataset
reader's deterministic ordering through to the render loop, and ultimately to our
index-named output files.

`camera_to_JSON` serialises camera parameters for the SIBR viewer; only used on first
training run, never in our path.

## Why it matters

This file is where an easily-overlooked experimental variable is fixed. Resolution
affects PSNR substantially — the same reconstruction scores differently at different
scales — so a project that varied it between configurations would produce meaningless
comparisons.

Here it is pinned by config (`resolution=-1`, applied identically everywhere), which
is one more control the dissertation obtains by invoking the unmodified renderer the
same way every time.

For the report, the honest framing is: all metrics were computed at the renderer's
default resolution handling (native, capped at 1600 px width), identically for every
configuration and for ground truth.
