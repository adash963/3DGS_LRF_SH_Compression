"""High-level pipeline operations used by scripts and the Colab runner."""

import numpy as np

from . import sh_layout
from .factorization import svd_factorize
from .ply_io import GaussianPly, extract_sh, insert_sh, read_gaussian_ply, write_gaussian_ply
from .quantization import kmeans_vq, scalar_quantize
from .serialization import CompressedSH


def extract_flat_sh(ply_path: str) -> tuple[GaussianPly, np.ndarray]:
    """Load model, return (model, X [N,48] float32)."""
    model = read_gaussian_ply(ply_path)
    sh = extract_sh(model)
    return model, sh_layout.flatten(sh)


def roundtrip_unchanged(src_ply: str, dst_ply: str) -> dict:
    """Extract SH and reinsert unchanged; save; report coefficient differences."""
    model, flat = extract_flat_sh(src_ply)
    model2 = insert_sh(model, sh_layout.unflatten(flat))
    write_gaussian_ply(model2, dst_ply)
    reread = read_gaussian_ply(dst_ply)
    flat2 = sh_layout.flatten(extract_sh(reread))
    non_sh_equal = all(
        np.array_equal(
            model.fields[f].astype(np.float32), reread.fields[f].astype(np.float32)
        )
        for f in model.non_sh_field_names()
    )
    return {
        "n_points": model.num_points,
        "max_abs_coeff_diff": float(np.max(np.abs(flat - flat2))),
        "non_sh_fields_identical": bool(non_sh_equal),
    }


def compress(flat_sh: np.ndarray, rank: int, mode: str = "lowrank",
             codebook_size: int = 256, scalar_bits: int = 8, seed: int = 0,
             factor_dtype: str = "float32", source_model: str = "") -> CompressedSH:
    """Factorise and optionally quantise; returns an in-memory CompressedSH asset."""
    u, v = svd_factorize(flat_sh, rank)
    np_dtype = {"float32": np.float32, "float16": np.float16}[factor_dtype]
    asset = CompressedSH(
        mode=mode, rank=rank, n_points=flat_sh.shape[0],
        v=v.astype(np_dtype), source_model=source_model, factor_dtype=factor_dtype,
    )
    if mode == "lowrank":
        asset.u = u.astype(np_dtype)
    elif mode == "vq":
        asset.vq = kmeans_vq(u, codebook_size, seed=seed)
    elif mode == "scalar":
        asset.scalar = scalar_quantize(u, bits=scalar_bits)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return asset


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
