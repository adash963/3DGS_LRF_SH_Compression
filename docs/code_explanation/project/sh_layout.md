# `src/sh_compression/sh_layout.py`

## Purpose

This file is the single source of truth for **where each spherical-harmonic
coefficient lives**. Every other module in the project defers to it rather than
indexing arrays by hand.

That matters more than it sounds. A 3DGS model stores 48 SH numbers per Gaussian,
and there are two plausible ways to lay them out (channel-major or
coefficient-major). Pick the wrong one and everything still *runs* — matrices have
the right shapes, the SVD converges, files save — but the red channel's data ends
up mixed with the blue channel's. Renders come out subtly wrong in a way that is
extremely hard to trace back. Centralising the layout here, and unit-testing it,
is what makes the project's "we are compressing the right numbers" claim defensible.

## Line-by-line

### Lines 1–9 — Module docstring

```python
"""Canonical SH layout for degree-3 3DGS models.

Canonical tensor form: ``[N, 3, 16]`` — channel-major, coefficient 0 is the DC term.
Flat form: ``[N, 48]`` with coefficient j of channel c at column ``c*16 + j``.

PLY mapping (verified against scene/gaussian_model.py save_ply/load_ply):
  f_dc_{c}          -> [:, c, 0]
  f_rest_{c*15 + j} -> [:, c, 1 + j]   for j in 0..14
"""
```

The docstring records the *decision* and its *evidence*. Two shapes are defined:

- **Canonical** `[N, 3, 16]` — N Gaussians, 3 colour channels, 16 coefficients each.
  Human-readable; you can ask for "channel 1, coefficient 5" directly.
- **Flat** `[N, 48]` — the same data as a 2-D matrix, which is what SVD needs.

The "verified against" note is important: this mapping was not guessed, it was read
out of the upstream `save_ply`/`load_ply` code. The DC term (coefficient 0) is the
view-independent base colour; coefficients 1–15 encode how colour changes with
viewing angle.

### Line 11 — Import

```python
import numpy as np
```

NumPy only. This module deliberately has no dependency on PyTorch, plyfile, or
anything else, so it can be imported anywhere cheaply.

### Lines 13–20 — Layout constants

```python
SH_DEGREE = 3
N_COEFFS = (SH_DEGREE + 1) ** 2          # 16
N_CHANNELS = 3
FLAT_DIM = N_CHANNELS * N_COEFFS          # 48
N_REST = N_COEFFS - 1                     # 15
DC_FIELDS = [f"f_dc_{c}" for c in range(N_CHANNELS)]
REST_FIELDS = [f"f_rest_{i}" for i in range(N_CHANNELS * N_REST)]
ORDERING = "channel_major_dc_first"
```

- `SH_DEGREE = 3` — the degree used by standard 3DGS.
- `N_COEFFS = (degree+1)² = 16` — this is the standard count of real spherical
  harmonic basis functions up to a given degree: 1 (degree 0) + 3 (degree 1) +
  5 (degree 2) + 7 (degree 3) = 16.
- `FLAT_DIM = 3 × 16 = 48` — the per-Gaussian feature count, and the width of the
  matrix `X` in the dissertation's `X ≈ U·V` formulation.
- `N_REST = 15` — the non-DC coefficients, stored separately by upstream.
- `DC_FIELDS` evaluates to `['f_dc_0', 'f_dc_1', 'f_dc_2']` — the PLY property
  names for the base colour of each channel.
- `REST_FIELDS` evaluates to `['f_rest_0', ..., 'f_rest_44']` — 45 names
  (3 channels × 15 coefficients).
- `ORDERING` is a string stamped into every saved asset. If the layout convention
  ever changed, old files would still declare which convention they used, so a
  reader could detect the mismatch instead of silently misinterpreting data.

### Lines 23–26 — `flatten`

```python
def flatten(sh: np.ndarray) -> np.ndarray:
    """[N, 3, 16] -> [N, 48] (row-major, lossless)."""
    _check_canonical(sh)
    return np.ascontiguousarray(sh.reshape(sh.shape[0], FLAT_DIM))
```

Converts canonical form to the matrix form the SVD consumes. The shape is validated
first (line 25) so a wrong-shaped input fails immediately with a clear message
rather than producing a silently misinterpreted matrix.

`reshape` in NumPy's default row-major (C) order walks the last axis fastest, so
element `[n, c, j]` lands at flat column `c*16 + j` — exactly the documented
convention. `ascontiguousarray` guarantees a dense memory layout, which keeps
downstream BLAS operations fast and avoids surprises if the caller passed a view
into a larger array.

This operation is lossless and exactly invertible — a property directly asserted by
`test_sh_layout.py::test_flatten_unflatten_lossless`.

### Lines 29–33 — `unflatten`

```python
def unflatten(flat: np.ndarray) -> np.ndarray:
    """[N, 48] -> [N, 3, 16] (row-major, lossless)."""
    if flat.ndim != 2 or flat.shape[1] != FLAT_DIM:
        raise ValueError(f"expected [N, {FLAT_DIM}], got {flat.shape}")
    return np.ascontiguousarray(flat.reshape(flat.shape[0], N_CHANNELS, N_COEFFS))
```

The exact inverse of `flatten`. The validation is written inline rather than reusing
`_check_canonical`, because the expected shape is different (2-D here, 3-D there).

This is called after reconstruction — the compressed pipeline produces a flat
`[N, 48]` matrix, and this turns it back into the form `insert_sh` needs.

### Lines 36–49 — `from_ply_fields`

```python
def from_ply_fields(dc: np.ndarray, rest: np.ndarray) -> np.ndarray:
    n = dc.shape[0]
    if dc.shape != (n, N_CHANNELS):
        raise ValueError(f"dc shape {dc.shape} != ({n}, {N_CHANNELS})")
    if rest.shape != (n, N_CHANNELS * N_REST):
        raise ValueError(f"rest shape {rest.shape} != ({n}, {N_CHANNELS * N_REST})")
    sh = np.empty((n, N_CHANNELS, N_COEFFS), dtype=np.result_type(dc, rest))
    sh[:, :, 0] = dc
    sh[:, :, 1:] = rest.reshape(n, N_CHANNELS, N_REST)
    return sh
```

**This is the critical function** — where the PLY's flat list of named columns
becomes structured data.

- Lines 42–45 validate both inputs. `dc` must be `[N, 3]`, `rest` must be `[N, 45]`.
- Line 46 allocates the output. `np.result_type(dc, rest)` picks a dtype that can
  hold both inputs without loss — if one were float64 and the other float32, you'd
  get float64 rather than silently truncating.
- Line 47 places the DC terms at coefficient index 0 of each channel. `sh[:, :, 0]`
  selects "all Gaussians, all channels, coefficient 0", whose shape `[N, 3]` matches
  `dc` exactly.
- **Line 48 is the load-bearing line.** `rest.reshape(n, 3, 15)` interprets the 45
  flat columns as 3 groups of 15 — i.e. it asserts that `f_rest_0..14` are channel
  0, `f_rest_15..29` are channel 1, `f_rest_30..44` are channel 2. This is the
  channel-major claim. If upstream had instead interleaved channels
  (coefficient-major), this reshape would scramble colours while raising no error.

The claim is verified in `test_sh_layout.py::test_ply_field_mapping_no_permutation`,
which puts distinct known values in every position and checks each one arrives where
the documentation says it should.

### Lines 52–57 — `to_ply_fields`

```python
def to_ply_fields(sh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _check_canonical(sh)
    dc = np.ascontiguousarray(sh[:, :, 0])
    rest = np.ascontiguousarray(sh[:, :, 1:].reshape(sh.shape[0], N_CHANNELS * N_REST))
    return dc, rest
```

The exact inverse of `from_ply_fields`, used when writing a model back to disk.
Line 55 extracts the DC slice; line 56 takes coefficients 1–15 across all channels
and flattens them back to 45 columns in the same channel-major order.

Because this is a precise inverse, `from_ply_fields(*to_ply_fields(x))` returns `x`
unchanged — which is one link in the chain that produces the project's bit-exact
round-trip result.

### Lines 60–63 — `validate_finite`

```python
def validate_finite(sh: np.ndarray, name: str = "SH") -> None:
    if not np.isfinite(sh).all():
        bad = int(np.size(sh) - np.count_nonzero(np.isfinite(sh)))
        raise ValueError(f"{name} contains {bad} non-finite values (NaN/Inf)")
```

A guard against NaN and infinity. These are genuinely dangerous here: a single NaN
coefficient propagates through SVD to contaminate the *entire* factorisation (every
singular vector becomes NaN), and NaN Gaussians render as black or invisible.
Catching them at the boundary means the failure is reported where it originated.

Line 62 counts the bad values — `np.size` gives the total, `count_nonzero(isfinite)`
gives the good ones, so the difference is the bad count. Reporting the count rather
than just "contains NaN" helps distinguish one corrupt Gaussian from wholesale
corruption.

### Lines 66–68 — `_check_canonical`

```python
def _check_canonical(sh: np.ndarray) -> None:
    if sh.ndim != 3 or sh.shape[1] != N_CHANNELS or sh.shape[2] != N_COEFFS:
        raise ValueError(f"expected [N, {N_CHANNELS}, {N_COEFFS}], got {sh.shape}")
```

Private shape validator, shared by `flatten` and `to_ply_fields`. The leading
underscore marks it as internal. It checks all three conditions — that the array is
3-D, and that both trailing dimensions are correct — and reports the actual shape,
which usually makes the mistake obvious immediately.

## Why it matters

This 68-line file underpins the dissertation's most important verification result.
The claim "extraction and reinsertion is bit-exact, therefore we are provably
manipulating the correct coefficients" rests on:

1. `from_ply_fields` / `to_ply_fields` being exact inverses,
2. `flatten` / `unflatten` being exact inverses,
3. all four being unit-tested against explicit known index mappings.

Together these mean any error observed later in the pipeline must come from the
compression itself, never from data handling — which is precisely the confound that
would otherwise be impossible to rule out.
