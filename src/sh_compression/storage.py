"""Byte accounting: measured file sizes and compression ratios."""

import os

from .sh_layout import FLAT_DIM


def file_bytes(path: str) -> int:
    return os.path.getsize(path)


def baseline_sh_bytes(n_points: int) -> int:
    """Uncompressed SH storage in the baseline PLY: N x 48 x float32."""
    return n_points * FLAT_DIM * 4


def ratios(n_points: int, compressed_sh_file_bytes: int,
           baseline_ply_bytes: int | None = None) -> dict:
    sh_base = baseline_sh_bytes(n_points)
    out = {
        "baseline_sh_bytes": sh_base,
        "compressed_sh_bytes": compressed_sh_file_bytes,
        "sh_compression_ratio": sh_base / compressed_sh_file_bytes,
    }
    if baseline_ply_bytes is not None:
        non_sh = baseline_ply_bytes - sh_base
        out["baseline_ply_bytes"] = baseline_ply_bytes
        out["whole_model_bytes_compressed"] = non_sh + compressed_sh_file_bytes
        out["whole_model_ratio"] = baseline_ply_bytes / (non_sh + compressed_sh_file_bytes)
    return out
