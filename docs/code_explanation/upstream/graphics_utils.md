# `utils/graphics_utils.py`

## Purpose

Camera and projection mathematics: world-to-view matrices, the perspective projection
matrix, and field-of-view conversions. These build the `viewmatrix` and `projmatrix`
that the CUDA rasteriser uses to place Gaussians on screen.

Not modified or called by this project, but it defines the geometry that makes every
render deterministic and identical across configurations.

## Line-by-line

### Lines 17–20 — `BasicPointCloud`

```python
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
```

A simple container for COLMAP's sparse reconstruction — the initialisation input for
training. Unused by us (we load a trained checkpoint).

### Lines 22–29 — `geom_transform_points`

```python
def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)
```

Standard homogeneous transformation. Line 25 appends a column of ones to make
`[x, y, z, 1]`, line 26 applies the 4×4 matrix, and lines 28–29 perform the
perspective divide.

The `+ 0.0000001` on line 28 guards against division by zero for points exactly on the
camera plane — a small epsilon rather than a branch, which keeps the operation
vectorised.

### Lines 31–36 — `getWorld2View`

```python
def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)
```

Builds the world-to-camera matrix from a rotation `R` and translation `t`.

**`R.transpose()` on line 33** is the important detail: for a rotation matrix, the
transpose equals the inverse. COLMAP stores camera-to-world rotation, so transposing
gives world-to-camera — the direction the renderer needs.

### Lines 38–49 — `getWorld2View2`

```python
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)
```

The same, with optional scene translation and scaling. It inverts to camera-to-world
(line 44), adjusts the camera centre (lines 45–47), then inverts back.

Used to normalise scene scale. For our evaluation the parameters are defaults
(no translation, unit scale), so it reduces to `getWorld2View` — and, importantly, it
is identical for every configuration.

### Lines 51–71 — `getProjectionMatrix`

```python
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
```

The standard OpenGL-style perspective projection matrix, derived from the view
frustum. Lines 55–58 compute frustum bounds at the near plane from the field of view;
lines 64–70 fill the matrix entries.

Because the frustum is symmetric here (`bottom = -top`, `left = -right`), entries
`P[0,2]` and `P[1,2]` evaluate to zero — the general form is retained for clarity.

The product of this and the view matrix becomes `full_proj_transform`, passed to the
rasteriser as `projmatrix`.

### Lines 73–77 — Field-of-view conversions

```python
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
```

Convert between focal length in pixels and angular field of view — exact inverses.

COLMAP reports focal lengths; the renderer wants field of view. `focal2fov` performs
that conversion when cameras are loaded, and the result becomes `FoVx`/`FoVy` in the
`Camera` object, which `gaussian_renderer.render` turns into `tanfovx`/`tanfovy`.

## Why it matters

This file is the reason camera geometry is a non-issue for the dissertation's
comparisons.

Every configuration renders through the same cameras, which produce the same view and
projection matrices via these functions, which produce the same `viewmatrix` and
`projmatrix` in the rasteriser settings. The mathematics is deterministic and depends
only on the dataset, never on the model.

So when the report says "identical cameras for all configurations", that is
guaranteed by construction: the camera pipeline never sees the SH coefficients at all.
