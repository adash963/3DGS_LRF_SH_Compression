# `utils/sh_utils.py`

## Purpose

Evaluates spherical harmonic basis functions — the mathematics that turns the 48
stored coefficients into an actual RGB colour for a given viewing direction.

**This file explains what the numbers this project compresses actually mean.**
Nothing here is called during our evaluation (the CUDA rasteriser has its own
implementation), but understanding it is essential to interpreting the results.

Origin: adapted from the PlenOctree authors, under a BSD-style licence (lines 1–22).

## Line-by-line

### Lines 24–54 — The normalisation constants

```python
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, ...]
C3 = [-0.5900435899266435, 2.890611442640554, ...]
C4 = [...]
```

Precomputed normalisation constants for real spherical harmonics, grouped by degree.
They are not arbitrary — each is derived from the standard SH normalisation:

- `C0 = 0.2820947...` = `1/(2√π)` — the degree-0 constant
- `C1 = 0.4886025...` = `√(3/(4π))` — the degree-1 constant
- `C2` has 5 entries (degree 2 has 5 basis functions), `C3` has 7, `C4` has 9

Counting: 1 + 3 + 5 + 7 = 16 basis functions for degree 3, which is exactly the
`N_COEFFS` constant in our `sh_layout.py`. The number 48 in `X ∈ R^(N×48)` traces
directly back to this list.

Hardcoding these avoids recomputing factorials and square roots per Gaussian per
frame — with a million Gaussians, that matters.

### Lines 57–72 — `eval_sh` signature and validation

```python
def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff
```

Takes coefficients and viewing directions, returns colour.

**Note the expected shape: `[..., C, (deg+1)**2]`** — channels before coefficients.
This is the same channel-major convention our `sh_layout.py` documents, corroborating
the layout decision from a second place in the codebase.

`dirs` must be unit vectors — the polynomials below assume `x²+y²+z² = 1`.

### Line 74 — The DC term

```python
    result = C0 * sh[..., 0]
```

**The single most important line for understanding this project's results.**

Coefficient 0 multiplied by a constant — no dependence on direction at all. This is
the view-independent base colour: what the Gaussian looks like from anywhere.

This is why rank-1 compression still produces a recognisable scene (18.38 dB rather
than noise). If the factorisation captures mostly the DC component, the scene keeps
its basic colours and loses only view-dependent effects.

### Lines 75–80 — Degree 1

```python
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])
```

Three coefficients, each multiplied by one direction component. These produce a
**linear gradient** across the viewing sphere — the Gaussian appears slightly
different from the left than from the right.

The alternating signs come from the standard real-SH sign convention, not from
anything scene-specific.

### Lines 82–90 — Degree 2

```python
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])
```

Five coefficients on quadratic terms. Line 83–84 precompute the products once and
reuse them — a meaningful optimisation at scale.

These capture **broad directional variation**: a surface that is brighter along one
axis and dimmer along another, such as a soft sheen.

### Lines 92–100 — Degree 3

```python
            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                ...
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])
```

Seven coefficients on cubic terms, completing the 16 used by standard 3DGS. These
encode the **sharpest view-dependent effects** — tight specular highlights, the glint
that moves across a surface as the camera orbits.

### Lines 102–112 — Degree 4 (unused)

Nine more coefficients, supported by the function but not used by 3DGS, which stops
at degree 3.

### Lines 114–118 — Conversions

```python
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5
```

Convert between RGB and the DC coefficient. The `- 0.5` / `+ 0.5` centres colour
around mid-grey, so the DC coefficient represents deviation from grey rather than
absolute brightness. Used to initialise Gaussians from point-cloud colours during
training.

## Why this matters for the dissertation

This file provides the physical explanation for the project's most interesting
finding: **coefficient-space error and image error diverge.**

The reason is visible in the structure above. All 16 coefficients contribute to the
final colour, but they do so through basis functions of very different character:

- **Coefficient 0** is a constant — it sets the base colour everywhere.
- **Coefficients 1–3** vary linearly and smoothly.
- **Coefficients 9–15** vary rapidly, producing localised highlights.

The higher-order coefficients typically have small magnitudes, so an SVD — which
minimises total squared error — discards them first. That is why rank-1 retains 90%
of the matrix *energy* yet loses 5 dB of image quality: the discarded 10% is
concentrated in exactly the terms that produce specular detail, which the eye and the
metrics both notice.

It also explains the observed failure mode in the visual comparisons. Under extreme
VQ, geometry stays perfectly sharp (positions and scales are untouched) while
view-dependent colour flattens — the green hillside washing to grey haze is precisely
what happens when higher-degree terms are lost and only something close to the DC
term survives.

Finally, the constants here are the origin of the number 48. Degree 3 gives 16 basis
functions; three colour channels gives 48 values per Gaussian; that is the width of
the matrix `X` in the dissertation's formulation, and the reason the maximum possible
rank is 48.
