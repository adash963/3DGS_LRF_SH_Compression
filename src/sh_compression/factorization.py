"""Truncated-SVD low-rank factorisation of the scene SH matrix X [N, 48]."""

from dataclasses import dataclass

import numpy as np

from .sh_layout import FLAT_DIM


def _check_input(x: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError(f"expected 2-D matrix, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("input matrix contains NaN/Inf")


def svd_factorize(x: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """X ~= U V with U [N, r] (singular values folded in) and V [r, D].

    The input is never mutated. rank must be in 1..min(N, D).
    """
    _check_input(x)
    max_rank = min(x.shape)
    if not (isinstance(rank, (int, np.integer)) and 1 <= rank <= max_rank):
        raise ValueError(f"rank must be an int in 1..{max_rank}, got {rank!r}")
    u, s, vt = np.linalg.svd(np.asarray(x, dtype=np.float64), full_matrices=False)
    u_r = (u[:, :rank] * s[:rank]).astype(np.float32)
    v_r = vt[:rank].astype(np.float32)
    return u_r, v_r


def reconstruct(u: np.ndarray, v: np.ndarray, chunk_size: int = 1_000_000) -> np.ndarray:
    """U @ V computed in row chunks (float32 output)."""
    n = u.shape[0]
    out = np.empty((n, v.shape[1]), dtype=np.float32)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        out[start:stop] = u[start:stop].astype(np.float32) @ v.astype(np.float32)
    return out


@dataclass
class RankAnalysis:
    singular_values: np.ndarray
    ranks: list[int]
    rel_frobenius_error: list[float]
    mse: list[float]
    max_abs_error: list[float]
    explained_energy: list[float]
    theoretical_bytes_fp32: list[int]

    def as_rows(self) -> list[dict]:
        return [
            {
                "rank": r,
                "rel_frobenius_error": self.rel_frobenius_error[i],
                "mse": self.mse[i],
                "max_abs_error": self.max_abs_error[i],
                "explained_energy": self.explained_energy[i],
                "theoretical_bytes_fp32": self.theoretical_bytes_fp32[i],
            }
            for i, r in enumerate(self.ranks)
        ]


def analyse_ranks(x: np.ndarray, ranks: list[int]) -> RankAnalysis:
    """Singular spectrum + per-rank reconstruction errors. Skips ranks > max rank."""
    _check_input(x)
    x64 = np.asarray(x, dtype=np.float64)
    u, s, vt = np.linalg.svd(x64, full_matrices=False)
    max_rank = len(s)
    usable = [r for r in ranks if 1 <= r <= max_rank]
    total_energy = float(np.sum(s**2))
    fro_x = float(np.linalg.norm(x64))
    res = RankAnalysis(s.copy(), usable, [], [], [], [], [])
    for r in usable:
        recon = (u[:, :r] * s[:r]) @ vt[:r]
        diff = x64 - recon
        res.rel_frobenius_error.append(
            float(np.linalg.norm(diff)) / fro_x if fro_x > 0 else 0.0
        )
        res.mse.append(float(np.mean(diff**2)))
        res.max_abs_error.append(float(np.max(np.abs(diff))) if diff.size else 0.0)
        res.explained_energy.append(
            float(np.sum(s[:r] ** 2)) / total_energy if total_energy > 0 else 1.0
        )
        res.theoretical_bytes_fp32.append(4 * (x.shape[0] * r + r * x.shape[1]))
    return res


def effective_rank(s: np.ndarray, energy_threshold: float = 0.99) -> int:
    """Smallest r whose cumulative energy reaches the threshold."""
    energy = np.cumsum(s**2) / np.sum(s**2)
    return int(np.searchsorted(energy, energy_threshold) + 1)


def make_synthetic(
    n: int, dim: int = FLAT_DIM, rank: int | None = None,
    noise: float = 0.0, seed: int = 0, constant: bool = False,
) -> np.ndarray:
    """Synthetic test matrices: exact low rank, noisy, full-rank, or constant rows."""
    rng = np.random.default_rng(seed)
    if constant:
        return np.tile(rng.normal(size=(1, dim)), (n, 1)).astype(np.float32)
    if rank is None:
        x = rng.normal(size=(n, dim))
    else:
        x = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, dim))
    if noise > 0:
        x = x + rng.normal(scale=noise, size=x.shape)
    return x.astype(np.float32)
