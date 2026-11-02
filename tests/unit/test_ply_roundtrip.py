import numpy as np
import pytest

from sh_compression import sh_layout as L
from sh_compression.extraction import extract_flat_sh, roundtrip_unchanged
from sh_compression.ply_io import extract_sh, insert_sh, read_gaussian_ply


def test_read_write_roundtrip(tiny_model, tiny_ply):
    reread = read_gaussian_ply(tiny_ply)
    assert reread.field_names == tiny_model.field_names
    for f in tiny_model.field_names:
        assert np.array_equal(reread.fields[f], tiny_model.fields[f])


def test_extract_matches_ply_fields(tiny_model):
    sh = extract_sh(tiny_model)
    for c in range(3):
        assert np.array_equal(sh[:, c, 0], tiny_model.fields[f"f_dc_{c}"])
        for j in range(15):
            assert np.array_equal(sh[:, c, 1 + j], tiny_model.fields[f"f_rest_{c*15+j}"])


def test_insert_preserves_non_sh(tiny_model):
    sh = extract_sh(tiny_model)
    out = insert_sh(tiny_model, sh * 2.0)
    for f in tiny_model.non_sh_field_names():
        assert np.array_equal(out.fields[f], tiny_model.fields[f])
    assert np.allclose(extract_sh(out), sh * 2.0)


def test_full_roundtrip_via_disk(tiny_ply, tmp_path):
    report = roundtrip_unchanged(tiny_ply, str(tmp_path / "rt.ply"))
    assert report["max_abs_coeff_diff"] == 0.0
    assert report["non_sh_fields_identical"]


def test_extract_flat_shape(tiny_ply):
    _, flat = extract_flat_sh(tiny_ply)
    assert flat.shape == (100, L.FLAT_DIM) and flat.dtype == np.float32


def test_wrong_degree_rejected(tmp_path, tiny_model):
    from sh_compression.ply_io import GaussianPly, write_gaussian_ply

    names = [f for f in tiny_model.field_names if f != "f_rest_44"]
    broken = GaussianPly(names, {k: tiny_model.fields[k] for k in names})
    p = str(tmp_path / "broken.ply")
    write_gaussian_ply(broken, p)
    with pytest.raises(ValueError, match="SH degree"):
        read_gaussian_ply(p)


def test_nan_model_rejected(tiny_model):
    tiny_model.fields["f_dc_0"] = tiny_model.fields["f_dc_0"].copy()
    tiny_model.fields["f_dc_0"][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        extract_sh(tiny_model)
