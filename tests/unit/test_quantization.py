import numpy as np
import pytest

from sh_compression.quantization import kmeans_vq, scalar_quantize


def test_scalar_roundtrip_error_bounded():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 8)).astype(np.float32)
    q = scalar_quantize(x, bits=8)
    span = x.max(axis=0) - x.min(axis=0)
    assert np.all(np.abs(q.decode() - x) <= span / 255 * 0.51 + 1e-6)


def test_scalar_16bit_better_than_8bit():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 8)).astype(np.float32)
    e8 = np.abs(scalar_quantize(x, 8).decode() - x).max()
    e16 = np.abs(scalar_quantize(x, 16).decode() - x).max()
    assert e16 < e8


def test_scalar_invalid():
    with pytest.raises(ValueError):
        scalar_quantize(np.zeros((5, 2)), bits=7)
    bad = np.zeros((5, 2)); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        scalar_quantize(bad)


def test_vq_roundtrip_and_bounds():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1000, 8)).astype(np.float32)
    q = kmeans_vq(x, 64, seed=0)
    dec = q.decode()
    assert dec.shape == x.shape
    assert q.indices.max() < q.codebook.shape[0]
    assert q.index_bits == 6
    # VQ error must beat the trivial single-centroid quantiser
    single = kmeans_vq(x, 1, seed=0)
    assert np.mean((dec - x) ** 2) < np.mean((single.decode() - x) ** 2)


def test_vq_deterministic():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 4)).astype(np.float32)
    q1, q2 = kmeans_vq(x, 32, seed=7), kmeans_vq(x, 32, seed=7)
    assert np.array_equal(q1.indices, q2.indices)
    assert np.array_equal(q1.codebook, q2.codebook)


def test_vq_unique_vectors_exact():
    """A codebook holding every unique vector reproduces the input exactly."""
    rng = np.random.default_rng(4)
    base = rng.normal(size=(10, 4)).astype(np.float32)
    x = base[rng.integers(0, 10, size=300)]
    q = kmeans_vq(x, 16, seed=0)  # 10 unique rows <= 16
    assert np.allclose(q.decode(), x, atol=1e-6)


def test_vq_no_empty_clusters():
    rng = np.random.default_rng(5)
    # pathological: tight cluster + few outliers encourages empty clusters
    x = np.vstack([rng.normal(scale=1e-4, size=(500, 4)),
                   rng.normal(loc=100, size=(3, 4))]).astype(np.float32)
    q = kmeans_vq(x, 32, seed=0)
    counts = np.bincount(q.indices, minlength=q.codebook.shape[0])
    assert (counts > 0).all()


def test_vq_larger_codebook_not_worse():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(2000, 8)).astype(np.float32)
    e = {k: float(np.mean((kmeans_vq(x, k, seed=0).decode() - x) ** 2))
         for k in (16, 64, 256)}
    assert e[256] <= e[64] * 1.05 and e[64] <= e[16] * 1.05


def test_vq_invalid():
    with pytest.raises(ValueError):
        kmeans_vq(np.zeros((10, 2)), 0)
    bad = np.zeros((10, 2)); bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        kmeans_vq(bad, 4)
