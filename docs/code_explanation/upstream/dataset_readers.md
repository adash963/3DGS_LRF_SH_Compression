# `scene/dataset_readers.py`

## Purpose

Reads datasets into a common `SceneInfo` structure, handling both COLMAP
reconstructions and Blender/NeRF-synthetic JSON. **This file implements the train/test
split**, which determines the 38 evaluation views used throughout the dissertation.

314 lines. This document covers the parts that shaped our evaluation.

## Line-by-line

### Lines 26–46 — The data structures

```python
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    ...
    image_name: str
    ...
    is_test: bool
```

```python
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
```

`CameraInfo` carries rotation, translation, field of view, the image, and — crucially
— an `is_test` flag. `SceneInfo` bundles everything `Scene` needs.

### Lines 48–68 — `getNerfppNorm`

```python
def getNerfppNorm(cam_info):
    ...
```

Computes the scene's spatial extent (a centre and radius) from the camera positions.
Used to scale learning rates during training and to set densification thresholds.

Irrelevant to our rendering path, but note it is computed from **train cameras only**
(line 203) — a detail that matters for training, not for us.

### Lines 145–186 — `readColmapSceneInfo` and the split

```python
def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
```

The COLMAP entry point, taking our T&T `train` scene.

**The `llffhold=8` default is the origin of the number 38.**

```python
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            ...
            test_cam_names_list = [name for idx, name in enumerate(cam_names)
                                   if idx % llffhold == 0]
```

Lines 181–186 implement the split: **every 8th camera** becomes a test view. This is
the "LLFF hold" convention, named after the LLFF dataset paper and standard across
NeRF-family evaluation.

The `print` on line 183 is the `------------LLFF HOLD-------------` line visible in
every one of our Colab logs — a useful confirmation that the split actually ran.

For our scene: 301 cameras, every 8th held out, giving **38 test views** (indices
0, 8, 16, … 296).

```python
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
```

Lines 200–201 partition the cameras. Note line 200: when `train_test_exp` is enabled,
test cameras are *also* included in training (the exposure-compensation regime, which
trains on the left half of each test image and evaluates on the right). **We disable
this**, so the split is a clean partition.

**This is where `eval=False` would have broken everything.** With `eval` false, the
hold-out never happens, `test_cam_infos` is empty, and `render.py --skip_train`
renders zero images while exiting successfully. That silent no-op is the third problem
our `fix_cfg_args.py` solves — and it is why `scripts/render_model.py` explicitly
verifies that renders appeared.

### Lines 220–226 — Assembling `SceneInfo`

```python
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
```

Cameras are returned **in COLMAP order**, sorted by name. Combined with
`shuffle=False` in `Scene`, this is what makes our index-named render files
(`00000.png` … `00037.png`) refer to the same cameras in every run — the property
underpinning per-view comparison across configurations.

### Lines 228–283 — Blender path (unused)

```python
def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background,
                              is_test, extension=".png"):
```

Reads NeRF-synthetic datasets from `transforms_train.json` / `transforms_test.json`.
Not our path — T&T is COLMAP — but note the split there is explicit (separate files)
rather than derived by hold-out.

### Lines 312–315 — The dispatch table

```python
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}
```

The registry `scene/__init__.py` indexes into after detecting the dataset type.

## The methodological caveat, precisely stated

This file creates the situation the report must be careful about.

The checkpoint was **trained** with `eval=False`, meaning `test_cam_infos` was empty
and all 301 cameras went into training. Our `fix_cfg_args.py` sets `eval=True` at
render time, creating a test split from cameras the model has already seen.

The consequences, stated exactly:

- **Ground-truth PSNR (24.93 dB baseline) measures fit, not generalisation.** It
  cannot be quoted as novel-view synthesis quality, and the report does not do so.
- **The dissertation's actual measurement is unaffected.** Every comparison is
  compressed renders versus baseline renders on identical cameras. Both sides share
  the same training history, so the compression effect is measured cleanly.

This is why the report emphasises the *vs-baseline* numbers (34.63 dB for the headline
configuration) as the primary evidence, with vs-ground-truth reported as context and
an upper bound.

## Why it matters

Three things this file determines:

1. **The 38 evaluation views**, via the `llffhold=8` convention.
2. **Deterministic camera ordering**, which makes index-named files comparable across
   runs.
3. **The train/test caveat**, which is documented in the report's limitations rather
   than left for a marker to discover.

It is also the file that explains why `eval=True` had to be forced — without it, the
entire evaluation would have silently produced nothing.
