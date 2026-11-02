# `utils/general_utils.py`

## Purpose

Assorted helpers: activation inverses, the learning-rate schedule, quaternion and
covariance construction, and process initialisation. Two functions here directly
affect this project — `safe_state`, which seeds every RNG, and `build_rotation`,
which shows how the `rot_0..3` columns we preserve are interpreted.

## Line-by-line

### Lines 18–19 — `inverse_sigmoid`

```python
def inverse_sigmoid(x):
    return torch.log(x/(1-x))
```

The logit function, inverse of `torch.sigmoid`. Used to convert a desired opacity into
the raw stored value.

This explains the `opacity` column in a PLY: it holds a **logit**, not a probability,
so values outside `[0,1]` are normal and expected. Our code never interprets it — it
is copied through as opaque data — which is why our synthetic fixtures can use
arbitrary random values without being invalid.

### Lines 21–27 — `PILtoTorch`

```python
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
```

Loads a PIL image into a PyTorch tensor: divide by 255 to reach `[0, 1]`, then permute
`[H,W,C]` to `[C,H,W]`.

**The same two conversions our `metrics.load_image` performs** (in reverse for the
permute, since we work in NumPy's `[H,W,C]` convention). Ground-truth images pass
through here on their way to being written out by `render.py`, which is why our
loader's `[0,1]` normalisation matches what upstream produced.

### Lines 29–62 — `get_expon_lr_func` (training only)

```python
def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0,
                      max_steps=1000000):
    def helper(step):
        ...
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper
```

Log-linear learning-rate decay (adapted from Plenoxels/JaxNeRF), returning a closure
over the schedule parameters. Line 59 interpolates in log space, giving exponential
decay from `lr_init` to `lr_final`.

Used only during training. Irrelevant to us beyond confirming that our checkpoint was
produced by a standard schedule.

### Lines 64–76 — `strip_lowerdiag` / `strip_symmetric`

```python
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    ...
    return uncertainty
```

Packs a symmetric 3×3 covariance matrix into 6 values, exploiting symmetry to avoid
storing the redundant lower triangle. Used when covariance is precomputed in Python
rather than derived in CUDA — not our path.

### Lines 78–99 — `build_rotation`

```python
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]; x = q[:, 1]; y = q[:, 2]; z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    ...
    return R
```

Converts quaternions to 3×3 rotation matrices, normalising first (lines 79–81) so
stored values need not be unit quaternions.

**This is what the `rot_0..3` columns mean** — a quaternion in `(w, x, y, z)` order,
as line 85's naming shows (`r` is the scalar part). Lines 90–98 are the standard
quaternion-to-matrix formula.

Relevant to us only as confirmation of what we are preserving: our `insert_sh` copies
these four columns by reference, so every Gaussian's orientation is bit-identical
between baseline and compressed renders.

### Lines 101–110 — `build_scaling_rotation`

```python
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)
    L[:,0,0] = s[:,0]; L[:,1,1] = s[:,1]; L[:,2,2] = s[:,2]
    L = R @ L
    return L
```

Builds `L = R·S`, whose product `L·Lᵀ` is the 3-D covariance — the anisotropic ellipse
shape of each Gaussian. This is the decomposition described in the 3DGS paper:
covariance is parameterised by a rotation and three scales, guaranteeing it stays
positive semi-definite under optimisation.

Again, only context for us: scale and rotation are among the fields we preserve
exactly.

### Lines 112–133 — `safe_state`

```python
def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent
        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(
                        str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)
        def flush(self):
            old_f.flush()
    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
```

Two jobs.

**Lines 113–128 wrap stdout to timestamp every line.** This is the source of the
`[02/08 09:30:31]` markers throughout our Colab logs — useful evidence, since they let
the reproduction run's timings be compared against the original.

**Lines 130–133 seed every RNG to 0** — Python's `random`, NumPy, and PyTorch — and
pin the CUDA device.

This is a genuine contribution to the dissertation's determinism claim. Rendering is
deterministic anyway (no stochastic operations in the forward pass), but fixed seeds
remove any residual doubt, and it is consistent with our fresh-runtime reproduction
producing metrics identical to the original run **to every printed decimal**.

Note this is called by `render.py` before any work begins, so every one of our renders
ran with identical RNG state.

## Why it matters

Three points for the report:

1. **`safe_state`'s seeding** supports the determinism claim, complementing our own
   fixed seeds for K-means.
2. **The activation functions** explain why the non-SH columns we preserve contain
   values that look implausible (negative scales, out-of-range opacities) — they are
   pre-activation, and correctly so.
3. **`PILtoTorch`'s normalisation** matches our `load_image`, confirming that renders
   and ground truth reach our metrics in the same `[0,1]` convention they were written
   in.
