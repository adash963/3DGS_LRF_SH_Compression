# `scene/gaussian_model.py`

## Purpose

The core data structure of 3DGS: the collection of Gaussians, their parameters, their
optimisation, and their serialisation to and from PLY.

**This is the most important upstream file for this dissertation.** Its `save_ply`
and `load_ply` methods define where SH coefficients live on disk, and reading them is
how our `sh_layout.py` layout was determined — not guessed, not inferred from a
paper, but read from the code that writes the files.

473 lines, of which roughly 200 are training-only (densification, pruning, optimiser
management) and irrelevant to this project. This document covers the parts that
matter, and says so where it skips.

## Line-by-line

### Lines 12–28 — Imports

```python
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os, json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
```

`plyfile` (line 19) is the same library our `ply_io.py` uses — which is why files we
write are format-compatible with what this class reads.

`simple_knn._C` (line 21) is the second CUDA extension compiled during setup, used
only at initialisation to size Gaussians from neighbour distances.

### Lines 32–47 — `setup_functions`: the activations

```python
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
```

**Important for understanding what is stored on disk.** Parameters are stored in an
unconstrained space and activated on access:

- **scaling**: stored as log, activated with `exp` — guarantees positive scale
- **opacity**: stored as logit, activated with `sigmoid` — guarantees `[0,1]`
- **rotation**: stored as raw 4-vector, activated by normalising — guarantees unit
  quaternion

So the `scale_0..2` and `opacity` columns in a PLY are *pre-activation* values, which
is why they can be negative or large. Our code never interprets them — it copies them
through untouched — so this never mattered operationally, but it explains why the
synthetic test fixtures can use arbitrary random values without being invalid.

**SH coefficients have no activation.** They are stored and used directly, which is
precisely why they can be linearly factorised: no non-linearity intervenes between
what we compress and what the renderer consumes.

### Lines 50–66 — `__init__`

```python
    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        ...
```

**Lines 55–56 are the crux of the whole project.** SH coefficients are stored in
**two separate tensors**:

- `_features_dc` — the DC (degree-0) term, shape `[N, 1, 3]`
- `_features_rest` — degrees 1–3, shape `[N, 15, 3]`

Note the shape convention here is `[N, coefficients, channels]` — coefficient-major
in memory. This is the opposite of the on-disk order, and the transposition between
them is exactly what `save_ply` performs (see below). Missing that transposition is
the permutation bug our layout tests exist to rule out.

`active_sh_degree` starts at 0 and increases during training (a coarse-to-fine
schedule); for a loaded checkpoint it is set to `max_sh_degree = 3`.

### Lines 114–126 — Feature accessors

```python
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest
```

`get_features` concatenates along dim 1 (the coefficient axis), producing `[N, 16, 3]`
— all 16 coefficients per channel. This is what the renderer receives.

The separate accessors support the `separate_sh` optimisation in newer rasteriser
versions.

### Lines 149–176 — `create_from_pcd` (training only)

Initialises Gaussians from a COLMAP point cloud. Not used by this project — we load a
pretrained checkpoint — but two lines are informative:

```python
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
```

**All higher-order coefficients start at exactly zero** (line 155); only the DC term
is initialised, from the point's RGB colour. View-dependent effects are learned
entirely during optimisation.

This is relevant context for the dissertation's findings: higher-order coefficients
grow from zero only insofar as the data demands them, which partly explains why they
carry small magnitudes and why the SH matrix turns out to be strongly low-rank.

```python
        dist2 = torch.clamp_min(distCUDA2(...), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
```

Line 159–160 is where `simple_knn` is used: each Gaussian's initial size is derived
from the distance to its nearest neighbours, so dense regions get small Gaussians.

### Lines 183–190 — Learning rates (training only)

```python
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
```

Worth noting even though we never train: **higher-order SH coefficients are optimised
at 1/20th the learning rate of the DC term.** The original authors deliberately
constrained view-dependent detail to change slowly.

This is corroborating evidence for our compression results — the training procedure
itself treats higher-order coefficients as a small correction on top of base colour,
which is consistent with them contributing little energy and being the first thing an
SVD discards.

### Lines 225–237 — `construct_list_of_attributes`

```python
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
```

**The definitive column order**, and the source of `FIELD_ORDER` in our
`tests/conftest.py`:

```
x, y, z, nx, ny, nz, f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3
```

62 columns for a degree-3 model. Counts are derived from tensor shapes:
`_features_dc.shape[1] * shape[2] = 1 × 3 = 3` DC fields, and
`_features_rest.shape[1] * shape[2] = 15 × 3 = 45` rest fields.

Normals (`nx, ny, nz`) are always written as zeros — 3DGS does not use them; they
exist for PLY viewer compatibility.

### Lines 239–256 — `save_ply`: **where the layout is decided**

```python
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ...
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
```

**Line 245 is the single most consequential line for this dissertation.**

`_features_rest` has shape `[N, 15, 3]` — coefficient-major. The chain
`.transpose(1, 2)` converts it to `[N, 3, 15]` — **channel-major** — and
`.flatten(start_dim=1)` then flattens to `[N, 45]` in row-major order.

The result: `f_rest_{c*15 + j}` is channel `c`, coefficient `j`. Channels are
contiguous — all 15 of channel 0's coefficients, then all 15 of channel 1's, then
channel 2's.

**This is the origin of the channel-major claim in our `sh_layout.py`.** Had this
line omitted the transpose, the on-disk order would have been coefficient-major, and
our `from_ply_fields` reshape would scramble colours across channels while raising no
error whatsoever.

That risk is exactly why `test_sh_layout.py::test_ply_field_mapping_no_permutation`
exists — it verifies our reading of this line with distinct known values at all 45
positions.

Line 253 concatenates everything in the order `construct_list_of_attributes` declares,
which is why our writer must preserve field order to produce compatible files.

### Lines 263–314 — `load_ply`: the inverse

```python
    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        ...
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]), ...), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
```

Lines 281–284 read the DC terms explicitly by name into `[N, 3, 1]` — channel index
first. This confirms `f_dc_{c}` is channel `c`, matching our `DC_FIELDS` mapping.

```python
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
```

- **Line 287 sorts numerically**, not lexicographically — `f_rest_2` before
  `f_rest_10`. A lexicographic sort would silently scramble the coefficients. Our
  `REST_FIELDS` list is generated in numeric order for the same reason.
- **Line 288 is the degree assertion**: `3 × 16 − 3 = 45`. This is the upstream
  equivalent of the degree check in our `read_gaussian_ply`.
- **Line 293 reshapes to `[N, 3, 15]`** — channel-major, exactly inverting the
  transpose in `save_ply` and independently confirming the layout.

```python
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
```

Lines 308–309 transpose back to the internal `[N, coefficients, channels]` layout,
completing the round trip.

Note `device="cuda"` — this class always loads to GPU, which is why our `ply_io.py`
exists at all: we need to read and write these files on a CPU-only machine, using
NumPy rather than this class.

### Lines 316–473 — Training machinery (not used)

The remainder implements optimisation: `replace_tensor_to_optimizer`,
`_prune_optimizer`, `prune_points`, `densification_postfix`, `densify_and_split`,
`densify_and_clone`, `add_densification_stats`. These grow and prune the Gaussian
population during training.

None of it executes in this project. We load a fully-trained checkpoint and render;
the population is fixed at 1,071,462 Gaussians throughout, which is what makes
"same geometry, different colour" a controlled comparison.

## Why it matters

Three things this file gave the dissertation:

1. **The layout, read from source rather than assumed.** `save_ply` line 245 and
   `load_ply` line 293 together establish channel-major ordering. Our entire
   `sh_layout.py` is a faithful reimplementation of those two lines, and
   `results/stages/stage_0_requirements.md` cites them as the evidence.

2. **The degree check.** Line 288's assertion is mirrored in our reader, so a
   lower-degree model is rejected rather than misinterpreted.

3. **Corroborating context for the findings.** SH coefficients have no activation
   (so they are linearly compressible), higher-order terms initialise at zero and
   train at 1/20th learning rate (so they are small), and the DC term dominates —
   all consistent with the measured effective rank of 12.

The file also defines the boundary of this project's contribution: everything here is
upstream and unmodified. Our work reads what `save_ply` wrote, compresses it, and
writes something `load_ply` can read — without changing a line of this file.
