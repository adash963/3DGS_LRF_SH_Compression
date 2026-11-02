"""Read/write 3DGS point-cloud PLY files preserving every non-SH field exactly.

A model is represented as a ``GaussianPly``: ordered field names + a dict of 1-D
float32 numpy arrays, exactly as stored on disk. SH extraction/insertion goes through
sh_layout's canonical [N,3,16] form.
"""

from dataclasses import dataclass, field

import numpy as np
from plyfile import PlyData, PlyElement

from . import sh_layout


@dataclass
class GaussianPly:
    field_names: list[str]
    fields: dict[str, np.ndarray] = field(repr=False)

    @property
    def num_points(self) -> int:
        return len(self.fields[self.field_names[0]])

    def non_sh_field_names(self) -> list[str]:
        sh_names = set(sh_layout.DC_FIELDS) | set(sh_layout.REST_FIELDS)
        return [f for f in self.field_names if f not in sh_names]


def read_gaussian_ply(path: str) -> GaussianPly:
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"{path}: no 'vertex' element — not a 3DGS point cloud")
    el = ply["vertex"]
    names = [p.name for p in el.properties]
    missing = [f for f in sh_layout.DC_FIELDS if f not in names]
    if missing:
        raise ValueError(f"{path}: missing DC fields {missing}")
    rest_present = [f for f in names if f.startswith("f_rest_")]
    if len(rest_present) != len(sh_layout.REST_FIELDS):
        raise ValueError(
            f"{path}: found {len(rest_present)} f_rest fields, expected "
            f"{len(sh_layout.REST_FIELDS)} (degree-{sh_layout.SH_DEGREE} SH); "
            "model has a different SH degree"
        )
    fields = {n: np.asarray(el[n]) for n in names}
    n = len(fields[names[0]])
    for k, v in fields.items():
        if len(v) != n:
            raise ValueError(f"{path}: field {k} length {len(v)} != {n}")
    return GaussianPly(field_names=names, fields=fields)


def write_gaussian_ply(model: GaussianPly, path: str) -> None:
    n = model.num_points
    dtype = [(name, "f4") for name in model.field_names]
    arr = np.empty(n, dtype=dtype)
    for name in model.field_names:
        arr[name] = model.fields[name].astype(np.float32)
    PlyElement.describe(arr, "vertex")
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def extract_sh(model: GaussianPly) -> np.ndarray:
    """Return canonical SH tensor [N, 3, 16] (float32 copy)."""
    dc = np.stack([model.fields[f] for f in sh_layout.DC_FIELDS], axis=1)
    rest = np.stack([model.fields[f] for f in sh_layout.REST_FIELDS], axis=1)
    sh = sh_layout.from_ply_fields(dc, rest).astype(np.float32)
    sh_layout.validate_finite(sh)
    return sh


def insert_sh(model: GaussianPly, sh: np.ndarray) -> GaussianPly:
    """Return a new GaussianPly with SH replaced; all other fields shared unchanged."""
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
