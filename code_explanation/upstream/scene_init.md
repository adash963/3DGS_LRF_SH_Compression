# `scene/__init__.py`

## Purpose

The `Scene` class — ties together the dataset (cameras and images), the Gaussian model,
and the train/test split. It is what `render.py` constructs to get everything needed
for rendering.

**This file is where the 38 evaluation views are chosen**, which makes it central to
the dissertation's evaluation protocol.

## Line-by-line

### Lines 12–19 — Imports

```python
import os, random, json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
```

`random` (line 13) is used only for the optional camera shuffle — significant for us
because we need it *not* to happen.

### Lines 25–38 — Constructor and checkpoint selection

```python
    def __init__(self, args : ModelParams, gaussians : GaussianModel,
                 load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
```

Note `shuffle=True` is the **default**, but `render.py` explicitly passes
`shuffle=False`. That single argument is what gives our evaluation deterministic view
ordering.

Lines 34–35: `load_iteration=-1` (render.py's default) triggers a directory scan for
the highest iteration, resolving to 30000 for our checkpoint. This is the "Loading
trained model at iteration 30000" line visible in every Colab log.

### Lines 43–49 — Dataset type detection

```python
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images,
                                                          args.depths, args.eval,
                                                          args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path,
                                                           args.white_background,
                                                           args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"
```

Detects dataset format by looking for characteristic files: a `sparse/` directory
means COLMAP, a `transforms_train.json` means Blender/NeRF-synthetic. Our T&T `train`
scene takes the COLMAP branch.

**Line 44 is where the first Stage 1 crash surfaced.** It reads `args.depths` — and
because `render.py` builds its parser with `sentinel=True`, that attribute exists only
if `cfg_args` supplies it. The 2023-era checkpoint predates the field, producing:

```
AttributeError: 'GroupParams' object has no attribute 'depths'
```

The traceback pointed here, but the *cause* was in `arguments/__init__.py`'s merge
logic — which is why the fix was to write a complete `cfg_args` rather than to patch
this line.

**`args.eval` on the same line selects the train/test split.** This is the parameter
our `fix_cfg_args.py` forces to `True`.

### Lines 51–63 — First-run bookkeeping (skipped for us)

```python
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(...) as dest_file:
                dest_file.write(src_file.read())
            ...
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
```

Copies the input point cloud and writes `cameras.json` — only when starting fresh
training. We always load a checkpoint, so `loaded_iter` is set and this block is
skipped entirely.

That matters practically: it means rendering never writes into the model directory
beyond its own render outputs, so our staging approach cannot corrupt the baseline.

### Lines 65–67 — The shuffle

```python
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
```

**The lines that must not execute for our evaluation to be valid.**

`render.py` passes `shuffle=False`, so camera order stays as the dataset reader
produced it — sorted and deterministic. Since `render_set` names output files by
*index*, a shuffle would mean `00012.png` referred to different cameras in different
runs, silently invalidating every per-view comparison.

This is a case where the correctness of our protocol depends on an upstream default
being overridden — worth knowing, since it is not obvious from our own code.

### Lines 69–75 — Camera construction

```python
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)
```

Builds `Camera` objects, loading each image from disk. This is the slow part of
startup — the "Loading Training Cameras / Loading Test Cameras" pause of several
seconds visible in every Colab log, repeated for each of the 12 configurations.

`resolution_scales=[1.0]` means full resolution. Identical for every configuration, so
all comparisons happen at the same resolution — one more control obtained by not
varying the invocation.

### Lines 77–83 — Loading the Gaussians

```python
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras,
                                           self.cameras_extent)
```

**Lines 78–81 define the path our staging directory must replicate:**

```
<model_path>/point_cloud/iteration_<N>/point_cloud.ply
```

This is exactly what `scripts/render_model.py` constructs. Knowing this path is what
allowed variant PLYs to be rendered through the unmodified renderer rather than
requiring renderer changes.

`load_ply` is the method documented in [gaussian_model.md](gaussian_model.md) — the
inverse of `save_ply`, and the reader our written files must satisfy.

### Lines 85–100 — Save and accessors

```python
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        ...

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
```

`save` is training-only. The two accessors are what `render.py` calls to obtain each
camera set; with `--skip_train`, only `getTestCameras()` is used, returning the 38
evaluation views.

## The train/test split

The split itself is implemented in `dataset_readers.py`, but its consequence is
visible here through `args.eval`:

- **`eval=False`** — all 301 cameras go to train, test is empty. `--skip_train` then
  renders nothing.
- **`eval=True`** — every 8th camera goes to test (the "LLFF hold" convention), giving
  301 ÷ 8 ≈ 38 test views.

That is where the number 38 comes from, and the "------------LLFF HOLD-------------"
line in the Colab logs is the split being applied.

**The methodological caveat this creates** is recorded in the report: the checkpoint
was *trained* with `eval=False`, meaning all 301 cameras were seen during
optimisation. Forcing `eval=True` at render time creates a test split from cameras the
model already knows.

The consequence is precise. Ground-truth PSNR here measures *fit*, not generalisation,
so it cannot be quoted as evidence of novel-view synthesis quality. But the
dissertation's central measurement — compressed renders versus baseline renders on
identical views — is entirely unaffected, since both sides face the same cameras with
the same training history.

## Why it matters

This file supplies three things the evaluation depends on:

1. **The 38 test views**, via `args.eval` and the LLFF-hold split.
2. **Deterministic ordering**, via `shuffle=False` — without which index-named files
   would not correspond across runs.
3. **The model directory layout**, which our staging approach replicates so that
   variant PLYs render through the unmodified renderer.

It is also where the `depths` crash surfaced, making it a useful reference when
explaining the Stage 1 debugging in the report.
