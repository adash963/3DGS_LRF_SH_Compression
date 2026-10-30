"""Quantisation of the per-Gaussian U factor: scalar (uniform) and K-means VQ."""

from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2


# ---------------------------------------------------------------- scalar

@dataclass
class ScalarQuantized:
    codes: np.ndarray          # uint8/uint16 [N, r]
    scale: np.ndarray          # float32 [r]
    offset: np.ndarray         # float32 [r]
    bits: int

    def decode(self) -> np.ndarray:
        return (self.codes.astype(np.float32) * self.scale + self.offset).astype(np.float32)


def scalar_quantize(x: np.ndarray, bits: int = 8) -> ScalarQuantized:
    """Per-column uniform quantisation to the given bit width (8 or 16)."""
    if bits not in (8, 16):
        raise ValueError(f"bits must be 8 or 16, got {bits}")
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("input must be a finite 2-D matrix")
    levels = (1 << bits) - 1
    lo = x.min(axis=0).astype(np.float64)
    hi = x.max(axis=0).astype(np.float64)
    span = np.where(hi > lo, hi - lo, 1.0)
    codes = np.round((x - lo) / span * levels)
    dtype = np.uint8 if bits == 8 else np.uint16
    return ScalarQuantized(
        codes=codes.astype(dtype),
        scale=(span / levels).astype(np.float32),
        offset=lo.astype(np.float32),
        bits=bits,
    )


# ---------------------------------------------------------------- K-means VQ

@dataclass
class VQQuantized:
    codebook: np.ndarray       # float32 [K, r]
    indices: np.ndarray        # uint8/uint16/uint32 [N]
    codebook_size: int
    seed: int

    @property
    def index_bits(self) -> int:
        return int(np.ceil(np.log2(max(self.codebook_size, 2))))

    def decode(self) -> np.ndarray:
        if self.indices.max(initial=0) >= self.codebook.shape[0]:
            raise ValueError("index out of codebook bounds")
        return self.codebook[self.indices].astype(np.float32)


def _index_dtype(k: int):
    if k <= 256:
        return np.uint8
    if k <= 65536:
        return np.uint16
    return np.uint32


def kmeans_vq(x: np.ndarray, codebook_size: int, seed: int = 0,
              iters: int = 50) -> VQQuantized:
    """K-means vector quantisation of the rows of x [N, r].

    Deterministic for a fixed seed. If N <= codebook_size, the codebook is the set of
    unique rows (exact reproduction).
    """
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("input must be a finite 2-D matrix")
    if codebook_size < 1:
        raise ValueError(f"codebook_size must be >= 1, got {codebook_size}")
    x64 = np.asarray(x, dtype=np.float64)
    unique_rows = np.unique(x64, axis=0)
    if len(unique_rows) <= codebook_size:
        codebook = unique_rows
        # exact assignment to nearest (== identical) unique row
        idx = _assign(x64, codebook)
    else:
        codebook, idx = kmeans2(
            x64, codebook_size, iter=iters, minit="++", seed=seed
        )
        # kmeans2 can leave empty clusters; reassign them to the farthest points
        codebook, idx = _fix_empty_clusters(x64, codebook, idx)
    k = codebook.shape[0]
    return VQQuantized(
        codebook=codebook.astype(np.float32),
        indices=idx.astype(_index_dtype(k)),
        codebook_size=k,
        seed=seed,
    )


def _assign(x: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Nearest-codeword assignment, chunked to bound memory."""
    out = np.empty(len(x), dtype=np.int64)
    step = max(1, 2_000_000 // max(1, codebook.shape[0]))
    for s in range(0, len(x), step):
        e = min(s + step, len(x))
        d = ((x[s:e, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
        out[s:e] = d.argmin(axis=1)
    return out


def _fix_empty_clusters(x, codebook, idx):
    counts = np.bincount(idx, minlength=codebook.shape[0])
    empty = np.flatnonzero(counts == 0)
    if len(empty) == 0:
        return codebook, idx
    # move each empty centroid onto the point currently worst-represented
    d = ((x - codebook[idx]) ** 2).sum(axis=1)
    worst = np.argsort(d)[::-1]
    codebook = codebook.copy()
    for i, c in enumerate(empty):
        codebook[c] = x[worst[i % len(worst)]]
    idx = _assign(x, codebook)
    return codebook, idx
