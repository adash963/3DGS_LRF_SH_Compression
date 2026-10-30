"""End-to-end (CPU) pipeline test on a synthetic model with genuine low-rank SH."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.conftest import FIELD_ORDER, make_tiny_model  # noqa: E402

from sh_compression import sh_layout  # noqa: E402
from sh_compression.extraction import compress, decompress_to_ply, extract_flat_sh  # noqa: E402
from sh_compression.ply_io import GaussianPly, write_gaussian_ply  # noqa: E402
from sh_compression.serialization import CompressedSH  # noqa: E402


@pytest.fixture
def lowrank_ply(tmp_path):
    """Synthetic model whose SH truly has rank 4 (plus mild noise)."""
    rng = np.random.default_rng(0)
    n = 2000
    model = make_tiny_model(n=n, seed=1)
    x = (rng.normal(size=(n, 4)) @ rng.normal(size=(4, 48))
         + rng.normal(scale=1e-3, size=(n, 48))).astype(np.float32)
    sh = sh_layout.unflatten(x)
    dc, rest = sh_layout.to_ply_fields(sh)
    for c, name in enumerate(sh_layout.DC_FIELDS):
        model.fields[name] = np.ascontiguousarray(dc[:, c])
    for i, name in enumerate(sh_layout.REST_FIELDS):
        model.fields[name] = np.ascontiguousarray(rest[:, i])
    path = str(tmp_path / "lowrank.ply")
    write_gaussian_ply(model, path)
    return path


@pytest.mark.parametrize("mode,kwargs,max_rel_err", [
    ("lowrank", {}, 1e-2),
    ("vq", {"codebook_size": 256}, 1.0),
    ("scalar", {"scalar_bits": 8}, 5e-2),
])
def test_compress_save_load_reconstruct_render_ply(tmp_path, lowrank_ply,
                                                   mode, kwargs, max_rel_err):
    _, flat = extract_flat_sh(lowrank_ply)
    asset = compress(flat, rank=4, mode=mode, **kwargs)
    apath = str(tmp_path / "asset.npz")
    reported = asset.save(apath)
    assert reported == os.path.getsize(apath)

    loaded = CompressedSH.load(apath)
    out_ply = str(tmp_path / "recon.ply")
    decompress_to_ply(loaded, lowrank_ply, out_ply)

    _, flat2 = extract_flat_sh(out_ply)
    rel = np.linalg.norm(flat2 - flat) / np.linalg.norm(flat)
    assert rel < max_rel_err
    # non-SH fields byte-preserved through the whole chain
    from sh_compression.ply_io import read_gaussian_ply
    a, b = read_gaussian_ply(lowrank_ply), read_gaussian_ply(out_ply)
    for f in a.non_sh_field_names():
        assert np.array_equal(a.fields[f], b.fields[f])


def test_wrong_point_count_rejected(tmp_path, lowrank_ply):
    _, flat = extract_flat_sh(lowrank_ply)
    asset = compress(flat[:100], rank=4)
    with pytest.raises(ValueError, match="points"):
        decompress_to_ply(asset, lowrank_ply, str(tmp_path / "x.ply"))
