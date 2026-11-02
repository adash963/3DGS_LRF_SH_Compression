import json
import os

import numpy as np
import pytest

from sh_compression.extraction import compress
from sh_compression.factorization import make_synthetic
from sh_compression.serialization import CompressedSH


@pytest.fixture
def flat():
    return make_synthetic(300, rank=6, noise=0.05, seed=0)


@pytest.mark.parametrize("mode,kwargs", [
    ("lowrank", {}),
    ("lowrank", {"factor_dtype": "float16"}),
    ("vq", {"codebook_size": 32}),
    ("scalar", {"scalar_bits": 8}),
])
def test_save_load_preserves_reconstruction(tmp_path, flat, mode, kwargs):
    asset = compress(flat, rank=6, mode=mode, **kwargs)
    path = str(tmp_path / "asset.npz")
    reported = asset.save(path)
    assert reported == os.path.getsize(path)  # reported size == actual bytes
    loaded = CompressedSH.load(path)
    assert np.allclose(loaded.reconstruct_flat(), asset.reconstruct_flat(), atol=1e-6)
    assert loaded.mode == mode and loaded.rank == 6 and loaded.n_points == 300


def test_full_rank_lowrank_matches_original(tmp_path, flat):
    asset = compress(flat, rank=48, mode="lowrank")
    rec = asset.reconstruct_flat()
    assert np.linalg.norm(rec - flat) / np.linalg.norm(flat) < 1e-5


def test_component_bytes_accounting(flat):
    asset = compress(flat, rank=6, mode="vq", codebook_size=32)
    comp = asset.component_bytes()
    assert set(comp) == {"v", "codebook", "indices", "metadata"}
    assert comp["indices"] == 300  # uint8 per point
    assert comp["v"] == 6 * 48 * 4


def test_unsupported_version_rejected(tmp_path, flat):
    asset = compress(flat, rank=4)
    path = str(tmp_path / "asset.npz")
    asset.save(path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(bytes(arrays["meta_json"].tobytes()).decode())
    meta["format_version"] = 999
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="format_version"):
        CompressedSH.load(path)


def test_corrupt_shape_rejected(tmp_path, flat):
    asset = compress(flat, rank=4)
    path = str(tmp_path / "asset.npz")
    asset.save(path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["u"] = arrays["u"][:100]  # truncate U
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="corrupt"):
        CompressedSH.load(path)


def test_deterministic_output(tmp_path, flat):
    a1 = compress(flat, rank=6, mode="vq", codebook_size=32, seed=5)
    a2 = compress(flat, rank=6, mode="vq", codebook_size=32, seed=5)
    assert np.array_equal(a1.vq.indices, a2.vq.indices)
    assert np.array_equal(a1.vq.codebook, a2.vq.codebook)
