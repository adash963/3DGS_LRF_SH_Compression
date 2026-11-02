# `src/sh_compression/extraction.py`

## Purpose

The orchestration layer. Every other core module does one job well; this file
composes them into the four operations the scripts and notebook actually call:
extract, verify, compress, decompress.

If you want to understand the pipeline in one sitting, read this file — it is the
whole workflow in 72 lines.

## Line-by-line

### Lines 1–10 — Imports

```python
"""High-level pipeline operations used by scripts and the Colab runner."""

import numpy as np
from . import sh_layout
from .factorization import svd_factorize
from .ply_io import GaussianPly, extract_sh, insert_sh, read_gaussian_ply, write_gaussian_ply
from .quantization import kmeans_vq, scalar_quantize
from .serialization import CompressedSH
```

Imports from every other module — this is the layer that knows how they fit together,
so the lower-level modules never need to know about each other.

### Lines 13–17 — `extract_flat_sh`

```python
def extract_flat_sh(ply_path: str) -> tuple[GaussianPly, np.ndarray]:
    """Load model, return (model, X [N,48] float32)."""
    model = read_gaussian_ply(ply_path)
    sh = extract_sh(model)
    return model, sh_layout.flatten(sh)
```

Three steps: read the PLY, pull out the SH tensor `[N, 3, 16]`, flatten to `[N, 48]`.

It returns **both** the model and the matrix, because callers almost always need the
model too — to know the point count, to get the file's non-SH fields for later
reinsertion, or to report file size. Returning both avoids a second expensive
read of a 265 MB file.

### Lines 20–37 — `roundtrip_unchanged`

```python
def roundtrip_unchanged(src_ply: str, dst_ply: str) -> dict:
    """Extract SH and reinsert unchanged; save; report coefficient differences."""
    model, flat = extract_flat_sh(src_ply)
    model2 = insert_sh(model, sh_layout.unflatten(flat))
    write_gaussian_ply(model2, dst_ply)
    reread = read_gaussian_ply(dst_ply)
    flat2 = sh_layout.flatten(extract_sh(reread))
```

**This function produces the project's Stage 2 verification result** — the strongest
evidence in the dissertation.

The chain is deliberately complete: extract (line 23) → unflatten and reinsert
(line 24) → **write to disk** (line 25) → **read back from disk** (line 26) →
extract again (line 27). Including the disk round trip is what makes this meaningful;
an in-memory check would not catch a serialisation bug, a dtype conversion, or a
column-ordering mistake in the writer.

```python
    non_sh_equal = all(
        np.array_equal(
            model.fields[f].astype(np.float32), reread.fields[f].astype(np.float32)
        )
        for f in model.non_sh_field_names()
    )
```

Lines 28–33 verify geometry survived. Every non-SH field is compared between the
original and the re-read file.

Precision matters here, and the static audit was right to flag the report's wording:
this compares **float32 values** via `np.array_equal`, after an explicit `astype`. It
does not compare raw byte buffers, dtypes, or PLY headers. The honest claim is "all
non-SH float32 field values compared equal after write and read" — which is what the
experiment actually establishes.

```python
    return {
        "n_points": model.num_points,
        "max_abs_coeff_diff": float(np.max(np.abs(flat - flat2))),
        "non_sh_fields_identical": bool(non_sh_equal),
    }
```

Lines 34–38 return the verification result. `max_abs_coeff_diff` is the worst-case
error over all 51 million coefficients — a maximum rather than a mean, because a mean
would hide a single catastrophically wrong value.

**On the real model this returns exactly `0.0`**, which is the claim that anchors the
whole project: any error measured later must come from compression, never from data
handling.

### Lines 40–60 — `compress`

```python
def compress(flat_sh: np.ndarray, rank: int, mode: str = "lowrank",
             codebook_size: int = 256, scalar_bits: int = 8, seed: int = 0,
             factor_dtype: str = "float32", source_model: str = "") -> CompressedSH:
    u, v = svd_factorize(flat_sh, rank)
    np_dtype = {"float32": np.float32, "float16": np.float16}[factor_dtype]
    asset = CompressedSH(
        mode=mode, rank=rank, n_points=flat_sh.shape[0],
        v=v.astype(np_dtype), source_model=source_model, factor_dtype=factor_dtype,
    )
```

The single entry point for building any compressed configuration.

Line 44 factorises — this always happens, in every mode. Line 45 maps the dtype
string to a NumPy type via dict lookup, which raises `KeyError` on an unknown value
rather than silently defaulting.

```python
    if mode == "lowrank":
        asset.u = u.astype(np_dtype)
    elif mode == "vq":
        asset.vq = kmeans_vq(u, codebook_size, seed=seed)
    elif mode == "scalar":
        asset.scalar = scalar_quantize(u, bits=scalar_bits)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return asset
```

Lines 51–59 branch on mode: store `U` directly, or quantise it two different ways.
The structure makes the experimental design explicit — all three modes share
identical factorisation, differing *only* in how `U` is stored. That is what makes
the ablation a controlled comparison: any quality difference between `lr_r16_fp32`
and `sq_r16_b8` is attributable to quantisation alone.

Note `V` is never quantised in any mode (line 47 stores it as-is, subject only to the
dtype choice), for the reason given in the quantisation document: it is tiny, and
every Gaussian depends on it.

### Lines 63–72 — `decompress_to_ply`

```python
def decompress_to_ply(asset: CompressedSH, baseline_ply: str, out_ply: str) -> None:
    """Reconstruct SH from the asset and write a renderable PLY.

    The output PLY is a temporary rendering artefact — its size is NOT the
    compressed size (the .npz asset is).
    """
    model = read_gaussian_ply(baseline_ply)
    if model.num_points != asset.n_points:
        raise ValueError(
            f"asset has {asset.n_points} points but model has {model.num_points}"
        )
    flat = asset.reconstruct_flat()
    write_gaussian_ply(insert_sh(model, sh_layout.unflatten(flat)), out_ply)
```

Turns a compressed asset back into something the CUDA renderer can load.

**The docstring warning is doing real work.** The output PLY is always full size
(265 MB) because it contains uncompressed geometry plus reconstructed SH. Someone
measuring *that* file and reporting it as the compressed size would report roughly
1× compression and conclude the method failed. The brief explicitly called out this
confusion as a thing to avoid, and the warning is repeated in
`scripts/reconstruct_model.py`'s output.

Lines 69–72 check the asset matches the model. Applying compression built from one
scene to a different scene would otherwise produce a shape mismatch deep in the
reconstruction, or worse, silently work if the point counts coincidentally matched.

Line 71 decodes and reconstructs; line 72 chains `insert_sh` (preserving all
geometry) with `write_gaussian_ply`.

## Why it matters

This file defines the experiment. Two design decisions embedded here shape every
result in the dissertation:

1. **The verification chain includes disk I/O** — so `max_abs_coeff_diff == 0.0` is
   a statement about the complete pipeline, not just in-memory arithmetic.
2. **All modes share one factorisation path** — so the ablation isolates the effect
   of quantisation rather than confounding it with other differences.

The explicit separation between the `.npz` asset (the compressed representation) and
the reconstructed PLY (a temporary render artefact) is maintained in code, in
docstrings, and in script output, because conflating them is the single easiest way
to produce a meaningless compression number.
