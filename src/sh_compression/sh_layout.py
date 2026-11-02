"""Canonical SH layout for degree-3 3DGS models.

Canonical tensor form: ``[N, 3, 16]`` — channel-major, coefficient 0 is the DC term.
Flat form: ``[N, 48]`` with coefficient j of channel c at column ``c*16 + j``.

PLY mapping (verified against scene/gaussian_model.py save_ply/load_ply):
  f_dc_{c}          -> [:, c, 0]
  f_rest_{c*15 + j} -> [:, c, 1 + j]   for j in 0..14
"""

import numpy as np

SH_DEGREE = 3
N_COEFFS = (SH_DEGREE + 1) ** 2          # 16
N_CHANNELS = 3
FLAT_DIM = N_CHANNELS * N_COEFFS          # 48
N_REST = N_COEFFS - 1                     # 15
DC_FIELDS = [f"f_dc_{c}" for c in range(N_CHANNELS)]
REST_FIELDS = [f"f_rest_{i}" for i in range(N_CHANNELS * N_REST)]
ORDERING = "channel_major_dc_first"


def flatten(sh: np.ndarray) -> np.ndarray:
    """[N, 3, 16] -> [N, 48] (row-major, lossless)."""
    _check_canonical(sh)
    return np.ascontiguousarray(sh.reshape(sh.shape[0], FLAT_DIM))


def unflatten(flat: np.ndarray) -> np.ndarray:
    """[N, 48] -> [N, 3, 16] (row-major, lossless)."""
    if flat.ndim != 2 or flat.shape[1] != FLAT_DIM:
        raise ValueError(f"expected [N, {FLAT_DIM}], got {flat.shape}")
    return np.ascontiguousarray(flat.reshape(flat.shape[0], N_CHANNELS, N_COEFFS))


def from_ply_fields(dc: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Assemble canonical [N,3,16] from PLY-ordered arrays.

    dc: [N, 3] columns f_dc_0..2; rest: [N, 45] columns f_rest_0..44.
    """
    n = dc.shape[0]
    if dc.shape != (n, N_CHANNELS):
        raise ValueError(f"dc shape {dc.shape} != ({n}, {N_CHANNELS})")
    if rest.shape != (n, N_CHANNELS * N_REST):
        raise ValueError(f"rest shape {rest.shape} != ({n}, {N_CHANNELS * N_REST})")
    sh = np.empty((n, N_CHANNELS, N_COEFFS), dtype=np.result_type(dc, rest))
    sh[:, :, 0] = dc
    sh[:, :, 1:] = rest.reshape(n, N_CHANNELS, N_REST)
    return sh


def to_ply_fields(sh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonical [N,3,16] -> (dc [N,3], rest [N,45]) in PLY column order."""
    _check_canonical(sh)
    dc = np.ascontiguousarray(sh[:, :, 0])
    rest = np.ascontiguousarray(sh[:, :, 1:].reshape(sh.shape[0], N_CHANNELS * N_REST))
    return dc, rest


def validate_finite(sh: np.ndarray, name: str = "SH") -> None:
    if not np.isfinite(sh).all():
        bad = int(np.size(sh) - np.count_nonzero(np.isfinite(sh)))
        raise ValueError(f"{name} contains {bad} non-finite values (NaN/Inf)")


def _check_canonical(sh: np.ndarray) -> None:
    if sh.ndim != 3 or sh.shape[1] != N_CHANNELS or sh.shape[2] != N_COEFFS:
        raise ValueError(f"expected [N, {N_CHANNELS}, {N_COEFFS}], got {sh.shape}")
