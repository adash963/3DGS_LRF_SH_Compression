# `src/sh_compression/ply_io.py`

## Purpose

Reads and writes 3DGS point-cloud PLY files. Its defining constraint: **everything
that is not an SH coefficient must survive untouched.** Positions, rotations,
scales, opacities and normals define the scene's geometry; this project compresses
only colour, so any accidental change to those fields would invalidate every
comparison.

The design achieves this by never interpreting non-SH fields at all — they are read
as opaque arrays, carried through, and written back.

## Line-by-line

### Lines 1–13 — Docstring and imports

```python
"""Read/write 3DGS point-cloud PLY files preserving every non-SH field exactly.
...
"""
from dataclasses import dataclass, field
import numpy as np
from plyfile import PlyData, PlyElement
from . import sh_layout
```

`plyfile` is the same library upstream uses, so files written here are byte-format
compatible with what the renderer expects. Importing `sh_layout` (not reimplementing
the mapping) keeps the layout decision in exactly one place.

### Lines 16–27 — The `GaussianPly` container

```python
@dataclass
class GaussianPly:
    field_names: list[str]
    fields: dict[str, np.ndarray] = field(repr=False)
```

A deliberately dumb container: an ordered list of column names plus a name→array
dict. Two details matter:

- **`field_names` is separate from `fields`.** Dict ordering alone would be fragile;
  keeping an explicit ordered list guarantees columns are written back in the same
  order they were read, which keeps output files structurally identical to inputs.
- **`repr=False` on `fields`** stops Python printing a million-element array if
  someone inspects the object in a debugger.

```python
    @property
    def num_points(self) -> int:
        return len(self.fields[self.field_names[0]])
```

Point count inferred from the first column. Safe because `read_gaussian_ply`
validates that all columns have equal length before constructing the object.

```python
    def non_sh_field_names(self) -> list[str]:
        sh_names = set(sh_layout.DC_FIELDS) | set(sh_layout.REST_FIELDS)
        return [f for f in self.field_names if f not in sh_names]
```

Returns everything except the 48 SH columns — the fields that must be preserved.
Used by the round-trip verification to check that geometry was untouched. Building
a `set` first makes the membership test O(1) rather than scanning a 48-element list
per field.

### Lines 30–51 — `read_gaussian_ply`

```python
def read_gaussian_ply(path: str) -> GaussianPly:
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"{path}: no 'vertex' element — not a 3DGS point cloud")
    el = ply["vertex"]
    names = [p.name for p in el.properties]
```

Line 31 parses the file. Lines 32–33 check it is a point cloud at all — a PLY could
legitimately contain a triangle mesh instead, and failing here gives a much clearer
message than an obscure `KeyError` later. Line 35 captures the column order as
stored on disk.

```python
    missing = [f for f in sh_layout.DC_FIELDS if f not in names]
    if missing:
        raise ValueError(f"{path}: missing DC fields {missing}")
```

Lines 36–38 confirm the base-colour columns exist. Their absence means this is not a
3DGS model.

```python
    rest_present = [f for f in names if f.startswith("f_rest_")]
    if len(rest_present) != len(sh_layout.REST_FIELDS):
        raise ValueError(
            f"{path}: found {len(rest_present)} f_rest fields, expected "
            f"{len(sh_layout.REST_FIELDS)} (degree-{sh_layout.SH_DEGREE} SH); "
            "model has a different SH degree"
        )
```

Lines 39–45 are a **degree check**. 3DGS can be trained at lower SH degree: degree 2
yields 9 coefficients (24 rest fields), degree 1 yields 4 (9 rest fields). This
project's mathematics assumes 48 features throughout, so a degree-2 model must be
rejected rather than silently mangled. The message names the actual cause ("model
has a different SH degree") instead of leaving the user to infer it from a count.

```python
    fields = {n: np.asarray(el[n]) for n in names}
    n = len(fields[names[0]])
    for k, v in fields.items():
        if len(v) != n:
            raise ValueError(f"{path}: field {k} length {len(v)} != {n}")
    return GaussianPly(field_names=names, fields=fields)
```

Line 46 pulls every column into a NumPy array — including columns this project does
not understand, which is exactly the point. Lines 47–50 verify all columns share one
length, establishing the invariant `num_points` relies on.

### Lines 54–61 — `write_gaussian_ply`

```python
def write_gaussian_ply(model: GaussianPly, path: str) -> None:
    n = model.num_points
    dtype = [(name, "f4") for name in model.field_names]
    arr = np.empty(n, dtype=dtype)
    for name in model.field_names:
        arr[name] = model.fields[name].astype(np.float32)
    PlyElement.describe(arr, "vertex")
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)
```

Line 56 builds a NumPy *structured dtype* — a record type with one named float32
(`f4`) field per column, matching how 3DGS stores everything. Line 57 allocates the
record array, and lines 58–59 fill it column by column **in `field_names` order**,
preserving the input file's layout.

Line 60 is a no-op: its return value is discarded and line 61 calls
`PlyElement.describe` again. Harmless (it just does the descriptive work twice), but
it is dead code and could be deleted.

A caveat worth being precise about: this writes float32 for every column. Round-trip
equality is therefore *float32 value equality*, not raw byte identity of the original
file — the static audit correctly flagged the stronger wording as unsupported.

### Lines 64–70 — `extract_sh`

```python
def extract_sh(model: GaussianPly) -> np.ndarray:
    dc = np.stack([model.fields[f] for f in sh_layout.DC_FIELDS], axis=1)
    rest = np.stack([model.fields[f] for f in sh_layout.REST_FIELDS], axis=1)
    sh = sh_layout.from_ply_fields(dc, rest).astype(np.float32)
    sh_layout.validate_finite(sh)
    return sh
```

Line 66 stacks the three DC columns into `[N, 3]`; line 67 stacks the 45 rest columns
into `[N, 45]`. `axis=1` is what makes each Gaussian a row. Crucially both iterate
over the **constant lists** from `sh_layout`, not over whatever order the file
happened to use — so a file listing its columns unusually still produces correctly
ordered data.

Line 68 delegates the structural assembly to `sh_layout` and normalises to float32.
Line 69 rejects NaN/Inf at the earliest point, before any of it can reach the SVD.

### Lines 73–84 — `insert_sh`

```python
def insert_sh(model: GaussianPly, sh: np.ndarray) -> GaussianPly:
    sh_layout.validate_finite(sh)
    if sh.shape[0] != model.num_points:
        raise ValueError(f"SH rows {sh.shape[0]} != model points {model.num_points}")
    dc, rest = sh_layout.to_ply_fields(sh.astype(np.float32))
    fields = dict(model.fields)
    for c, name in enumerate(sh_layout.DC_FIELDS):
        fields[name] = np.ascontiguousarray(dc[:, c])
    for i, name in enumerate(sh_layout.REST_FIELDS):
        fields[name] = np.ascontiguousarray(rest[:, i])
    return GaussianPly(field_names=list(model.field_names), fields=fields)
```

The inverse operation, and **the function that guarantees geometry preservation**.

- Line 75 rejects non-finite replacement coefficients — important because a
  reconstruction bug could produce NaN, and this stops it reaching a saved file.
- Lines 76–77 check the row count matches. Compressing model A and applying it to
  model B is a real mistake this catches.
- Line 79 splits the canonical tensor back into PLY column layout.
- **Line 80 is the key line:** `dict(model.fields)` makes a *shallow* copy. The new
  dict has its own key→value mapping, but non-SH values are the *same array objects*
  as the input's. Lines 81–84 then rebind only the 48 SH keys to new arrays.

  The consequence: non-SH fields are not copied, not converted, not touched — they
  are literally the same objects. Preservation is structural, not something that
  depends on a correct copy operation.
- Line 84 returns a new `GaussianPly` with a fresh `field_names` list (so mutating
  one object's column order cannot affect the other), completing a
  non-destructive operation — the input model remains valid and unchanged.

## Why it matters

This file is where the project's central experimental control is implemented. Every
comparison in the dissertation is of the form "same geometry, different colour
encoding". The shallow-copy-and-rebind pattern in `insert_sh` is what makes that
claim true by construction rather than by careful bookkeeping, and
`test_ply_roundtrip.py::test_insert_preserves_non_sh` asserts it directly.
