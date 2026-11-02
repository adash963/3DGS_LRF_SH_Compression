import numpy as np
import pytest

from sh_compression.metrics import psnr, ssim
from sh_compression.storage import baseline_sh_bytes, ratios


def test_identical_images_perfect():
    rng = np.random.default_rng(0)
    img = rng.random((32, 32, 3)).astype(np.float32)
    assert psnr(img, img) == float("inf")
    assert ssim(img, img) == pytest.approx(1.0)


def test_perturbed_images_worse():
    rng = np.random.default_rng(1)
    img = rng.random((64, 64, 3)).astype(np.float32)
    small = np.clip(img + rng.normal(scale=0.01, size=img.shape), 0, 1).astype(np.float32)
    big = np.clip(img + rng.normal(scale=0.2, size=img.shape), 0, 1).astype(np.float32)
    assert psnr(img, small) > psnr(img, big)
    assert ssim(img, small) > ssim(img, big)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        psnr(np.zeros((4, 4, 3)), np.zeros((5, 5, 3)))


def test_storage_ratios():
    n = 1000
    assert baseline_sh_bytes(n) == n * 48 * 4
    r = ratios(n, compressed_sh_file_bytes=n * 48, baseline_ply_bytes=n * 62 * 4)
    assert r["sh_compression_ratio"] == pytest.approx(4.0)
    assert r["whole_model_bytes_compressed"] == n * 14 * 4 + n * 48
    assert 1.0 < r["whole_model_ratio"] < 4.0
