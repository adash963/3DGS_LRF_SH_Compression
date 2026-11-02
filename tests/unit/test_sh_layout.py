import numpy as np
import pytest

from sh_compression import sh_layout as L


def test_constants():
    assert L.N_COEFFS == 16 and L.FLAT_DIM == 48 and len(L.REST_FIELDS) == 45


def test_flatten_unflatten_lossless():
    rng = np.random.default_rng(0)
    sh = rng.normal(size=(50, 3, 16)).astype(np.float32)
    assert np.array_equal(L.unflatten(L.flatten(sh)), sh)


def test_flat_column_mapping():
    """Coefficient j of channel c must land at flat column c*16+j."""
    sh = np.zeros((1, 3, 16), dtype=np.float32)
    sh[0, 1, 5] = 7.0
    flat = L.flatten(sh)
    assert flat[0, 1 * 16 + 5] == 7.0 and np.count_nonzero(flat) == 1


def test_ply_field_mapping_no_permutation():
    """f_dc_c -> [:,c,0]; f_rest_{c*15+j} -> [:,c,1+j]."""
    n = 4
    dc = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    rest = 100 + np.arange(n * 45, dtype=np.float32).reshape(n, 45)
    sh = L.from_ply_fields(dc, rest)
    assert np.array_equal(sh[:, :, 0], dc)
    for c in range(3):
        for j in range(15):
            assert np.array_equal(sh[:, c, 1 + j], rest[:, c * 15 + j])
    dc2, rest2 = L.to_ply_fields(sh)
    assert np.array_equal(dc2, dc) and np.array_equal(rest2, rest)


def test_bad_shapes_rejected():
    with pytest.raises(ValueError):
        L.flatten(np.zeros((5, 3, 15)))
    with pytest.raises(ValueError):
        L.unflatten(np.zeros((5, 47)))
    with pytest.raises(ValueError):
        L.from_ply_fields(np.zeros((5, 3)), np.zeros((5, 44)))


def test_nan_detection():
    sh = np.zeros((2, 3, 16))
    sh[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        L.validate_finite(sh)
