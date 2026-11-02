"""Serialisable compressed-SH asset (single .npz file, versioned).

Modes:
  lowrank   — U (fp32/fp16) + V
  vq        — codebook + indices (U quantised) + V
  scalar    — scalar-quantised U codes + scale/offset + V
"""

import json
import os
from dataclasses import dataclass

import numpy as np

from . import FORMAT_VERSION
from .factorization import reconstruct as _lowrank_reconstruct
from .quantization import ScalarQuantized, VQQuantized
from .sh_layout import FLAT_DIM, ORDERING, SH_DEGREE

MODES = ("lowrank", "vq", "scalar")


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

    # ------------------------------------------------------------ reconstruct
    def reconstruct_flat(self) -> np.ndarray:
        """Return reconstructed X [N, 48] float32."""
        if self.mode == "lowrank":
            u = self.u
        elif self.mode == "vq":
            u = self.vq.decode()
        elif self.mode == "scalar":
            u = self.scalar.decode()
        else:
            raise ValueError(f"unknown mode {self.mode}")
        return _lowrank_reconstruct(np.asarray(u), self.v)

    # ------------------------------------------------------------ save / load
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
        arrays["meta_json"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )
        np.savez_compressed(path, **arrays)
        real = path if path.endswith(".npz") else path + ".npz"
        return os.path.getsize(real)

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
            obj = cls(
                mode=meta["mode"], rank=meta["rank"], n_points=meta["n_points"],
                v=z["v"], source_model=meta.get("source_model", ""),
                factor_dtype=meta.get("factor_dtype", "float32"),
            )
            if obj.mode == "lowrank":
                obj.u = z["u"]
            elif obj.mode == "vq":
                obj.vq = VQQuantized(
                    codebook=z["codebook"], indices=z["indices"],
                    codebook_size=meta["codebook_size"], seed=meta.get("seed", 0),
                )
            elif obj.mode == "scalar":
                obj.scalar = ScalarQuantized(
                    codes=z["codes"], scale=z["scale"], offset=z["offset"],
                    bits=meta["bits"],
                )
        _validate(obj)
        return obj

    # ------------------------------------------------------------ accounting
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
