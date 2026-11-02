# `src/sh_compression/serialization.py`

## Purpose

Defines the compressed asset format — the single `.npz` file that **is** the
project's deliverable. Every compression ratio in the dissertation is
`baseline SH bytes ÷ the size of a file produced here`.

This file also carries the project's most important measurement discipline: the
reported size is the *actual file size on disk*, obtained from the filesystem, never
an arithmetic estimate.

## Line-by-line

### Lines 1–20 — Docstring, imports, mode list

```python
"""Serialisable compressed-SH asset (single .npz file, versioned).

Modes:
  lowrank   — U (fp32/fp16) + V
  vq        — codebook + indices (U quantised) + V
  scalar    — scalar-quantised U codes + scale/offset + V
"""
import json, os
from dataclasses import dataclass
import numpy as np
from . import FORMAT_VERSION
from .factorization import reconstruct as _lowrank_reconstruct
from .quantization import ScalarQuantized, VQQuantized
from .sh_layout import FLAT_DIM, ORDERING, SH_DEGREE

MODES = ("lowrank", "vq", "scalar")
```

Three storage modes share one container so the evaluation pipeline can treat any
configuration uniformly. `V` is present in all three — only the representation of
`U` changes.

### Lines 23–35 — The `CompressedSH` dataclass

```python
@dataclass
class CompressedSH:
    mode: str
    rank: int
    n_points: int
    v: np.ndarray                       # [r, 48] float32/float16
    u: np.ndarray | None = None         # lowrank mode: [N, r]
    vq: VQQuantized | None = None       # vq mode
    scalar: ScalarQuantized | None = None  # scalar mode
    source_model: str = ""
    factor_dtype: str = "float32"
```

Exactly one of `u` / `vq` / `scalar` is populated, selected by `mode`. `source_model`
records which PLY the asset came from — provenance that makes it possible to check an
asset is being applied to the model it was built from.

### Lines 38–49 — `reconstruct_flat`

```python
    def reconstruct_flat(self) -> np.ndarray:
        if self.mode == "lowrank":
            u = self.u
        elif self.mode == "vq":
            u = self.vq.decode()
        elif self.mode == "scalar":
            u = self.scalar.decode()
        else:
            raise ValueError(f"unknown mode {self.mode}")
        return _lowrank_reconstruct(np.asarray(u), self.v)
```

The decode path, and a neat illustration of the design: each mode only needs to
produce a `[N, r]` matrix `U`, after which reconstruction is identical everywhere —
`U @ V`. Quantisation is *transparent* to the factorisation.

### Lines 52–79 — `save`

```python
    def save(self, path: str) -> int:
        meta = {
            "format_version": FORMAT_VERSION,
            "mode": self.mode,
            "rank": int(self.rank),
            "n_points": int(self.n_points),
            "flat_dim": FLAT_DIM,
            "sh_degree": SH_DEGREE,
            "ordering": ORDERING,
            "factor_dtype": self.factor_dtype,
            "source_model": self.source_model,
        }
```

The metadata block makes the file **self-describing**. `flat_dim`, `sh_degree` and
`ordering` mean a reader can verify the file matches its own conventions rather than
assuming compatibility — the protection against silently misinterpreting an asset
written under a different layout.

```python
        arrays: dict[str, np.ndarray] = {"v": self.v}
        if self.mode == "lowrank":
            arrays["u"] = self.u
        elif self.mode == "vq":
            arrays["codebook"] = self.vq.codebook
            arrays["indices"] = self.vq.indices
            meta["codebook_size"] = int(self.vq.codebook_size)
            meta["seed"] = int(self.vq.seed)
        elif self.mode == "scalar":
            arrays["codes"] = self.scalar.codes
            arrays["scale"] = self.scalar.scale
            arrays["offset"] = self.scalar.offset
            meta["bits"] = int(self.scalar.bits)
```

Mode-specific payloads. Note the VQ branch stores `seed` — enough information to
*regenerate* the codebook, not merely to use it, which supports the reproducibility
claim.

```python
        arrays["meta_json"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )
        np.savez_compressed(path, **arrays)
        real = path if path.endswith(".npz") else path + ".npz"
        return os.path.getsize(real)
```

- Line 74 smuggles JSON metadata into the archive by encoding it as a uint8 array —
  `.npz` stores only arrays, so this is how structured metadata rides along.
- Line 77 writes with **zip compression**. Worth being precise about: the reported
  size is post-compression, which is why it does not equal the sum of raw array
  sizes. For `sq_r16_b8`, raw payload is ~17.1 MB but the file is 13.7 MB — zip finds
  redundancy in the quantised integers.
- Lines 78–79 handle NumPy's habit of appending `.npz` if absent, then **return the
  real size from the filesystem**. This single line is what backs the dissertation's
  "measured, not estimated" claim, and `test_serialization.py` asserts the returned
  value equals `os.path.getsize`.

### Lines 82–110 — `load`

```python
    @classmethod
    def load(cls, path: str) -> "CompressedSH":
        with np.load(path) as z:
            meta = json.loads(bytes(z["meta_json"].tobytes()).decode("utf-8"))
            if meta.get("format_version") != FORMAT_VERSION:
                raise ValueError(
                    f"unsupported format_version {meta.get('format_version')!r}, "
                    f"expected {FORMAT_VERSION}"
                )
            if meta["mode"] not in MODES:
                raise ValueError(f"unknown mode {meta['mode']!r}")
            if meta["flat_dim"] != FLAT_DIM or meta["sh_degree"] != SH_DEGREE:
                raise ValueError("SH layout mismatch with this codebase")
```

Three validation gates before any data is trusted: version, mode, and layout
compatibility. The version check is the standard defence against a future format
change silently misreading old files; `.get()` rather than `[...]` means a file
lacking the key produces the clear version error rather than a `KeyError`.

```python
            obj = cls(
                mode=meta["mode"], rank=meta["rank"], n_points=meta["n_points"],
                v=z["v"], source_model=meta.get("source_model", ""),
                factor_dtype=meta.get("factor_dtype", "float32"),
            )
            if obj.mode == "lowrank":
                obj.u = z["u"]
            elif obj.mode == "vq":
                obj.vq = VQQuantized(...)
            elif obj.mode == "scalar":
                obj.scalar = ScalarQuantized(...)
        _validate(obj)
        return obj
```

Reconstructs the object mode by mode, then runs structural validation. The `with`
block ensures the file handle closes even on error.

### Lines 113–127 — `component_bytes`

```python
    def component_bytes(self) -> dict[str, int]:
        """Raw (uncompressed-array) byte accounting per component + metadata."""
        comp = {"v": int(self.v.nbytes)}
        if self.mode == "lowrank":
            comp["u"] = int(self.u.nbytes)
        elif self.mode == "vq":
            comp["codebook"] = int(self.vq.codebook.nbytes)
            comp["indices"] = int(self.vq.indices.nbytes)
        elif self.mode == "scalar":
            comp["codes"] = int(self.scalar.codes.nbytes)
            comp["scale_offset"] = int(self.scalar.scale.nbytes + self.scalar.offset.nbytes)
        comp["metadata"] = 512  # bounded JSON header estimate; file size is authoritative
        return comp
```

Breaks storage down by component — the source of the storage-breakdown figure, and
the evidence for "indices dominate under VQ".

**An important caveat the static audit identified.** These are *raw* array sizes
(`nbytes`), but `save()` returns the *zip-compressed* file size. The two do not
match, and the components must therefore not be presented as a partition of the
compressed file. The docstring and the trailing comment say this, but the report's
figure caption needs the same clarification. The 512-byte metadata figure is a fixed
estimate, not a measurement.

### Lines 130–141 — `_validate`

```python
def _validate(obj: CompressedSH) -> None:
    if obj.v.shape != (obj.rank, FLAT_DIM):
        raise ValueError(f"corrupt asset: V shape {obj.v.shape} != ({obj.rank}, {FLAT_DIM})")
    if obj.mode == "lowrank" and obj.u.shape != (obj.n_points, obj.rank):
        raise ValueError("corrupt asset: U shape mismatch")
    if obj.mode == "vq":
        if obj.vq.indices.shape != (obj.n_points,):
            raise ValueError("corrupt asset: indices shape mismatch")
        if obj.vq.indices.max(initial=0) >= obj.vq.codebook.shape[0]:
            raise ValueError("corrupt asset: index out of codebook bounds")
    if obj.mode == "scalar" and obj.scalar.codes.shape != (obj.n_points, obj.rank):
        raise ValueError("corrupt asset: codes shape mismatch")
```

Cross-checks that array shapes agree with the declared metadata. This catches
truncation and tampering — `test_serialization.py::test_corrupt_shape_rejected`
truncates `U` inside a saved file and confirms loading fails loudly rather than
producing a wrong-sized reconstruction.

The index bounds check on line 138 is the most valuable: an out-of-range index would
otherwise raise an obscure indexing error deep inside `decode`, or in a language
without bounds checking, read arbitrary memory.

## Why it matters

Three dissertation claims live in this file:

1. **"Measured, not theoretical"** — `save` returns `os.path.getsize`.
2. **"Assets load in a clean process"** — `load` performs full validation from the
   file alone, with no reliance on in-memory state.
3. **"Deterministic"** — the VQ seed is stored, so a codebook can be regenerated.

The gap the audit correctly flagged is that `component_bytes` reports raw sizes while
the headline number is compressed — a labelling issue in the report, not an error in
any measurement.
