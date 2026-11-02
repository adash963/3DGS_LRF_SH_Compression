import numpy as np
import pytest

from sh_compression.factorization import (
    analyse_ranks, effective_rank, make_synthetic, reconstruct, svd_factorize,
)

RANKS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]


def rel_err(x, u, v):
    return np.linalg.norm(x - reconstruct(u, v)) / np.linalg.norm(x)


@pytest.mark.parametrize("true_rank", [1, 2, 4])
def test_exact_rank_recovers(true_rank):
    x = make_synthetic(200, rank=true_rank, seed=1)
    u, v = svd_factorize(x, true_rank)
    assert rel_err(x, u, v) < 1e-6


def test_rank3_worse_than_rank4_on_rank4_data():
    x = make_synthetic(200, rank=4, seed=2)
    e3 = rel_err(x, *svd_factorize(x, 3))
    e4 = rel_err(x, *svd_factorize(x, 4))
    assert e4 < 1e-6 < e3


def test_error_monotone_energy_monotone():
    x = make_synthetic(300, rank=4, noise=0.1, seed=3)
    res = analyse_ranks(x, RANKS)
    errs, energy = res.rel_frobenius_error, res.explained_energy
    assert all(errs[i + 1] <= errs[i] + 1e-12 for i in range(len(errs) - 1))
    assert all(energy[i + 1] >= energy[i] - 1e-12 for i in range(len(energy) - 1))


def test_full_rank_reconstruction_exact():
    x = make_synthetic(100, seed=4)  # full-rank random
    u, v = svd_factorize(x, 48)
    assert rel_err(x, u, v) < 1e-5


def test_constant_data_is_rank_one():
    x = make_synthetic(150, constant=True, seed=5)
    u, v = svd_factorize(x, 1)
    assert rel_err(x, u, v) < 1e-6
    assert effective_rank(analyse_ranks(x, [1]).singular_values) == 1


def test_ranks_above_max_skipped():
    x = make_synthetic(10, seed=6)  # max rank 10
    res = analyse_ranks(x, RANKS)
    assert max(res.ranks) <= 10


def test_invalid_inputs_rejected():
    x = make_synthetic(50, seed=7)
    for bad in (0, -1, 49, 2.5):
        with pytest.raises(ValueError):
            svd_factorize(x, bad)
    bad_x = x.copy()
    bad_x[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        svd_factorize(bad_x, 4)
    with pytest.raises(ValueError):
        svd_factorize(x.ravel(), 4)


def test_input_not_mutated():
    x = make_synthetic(50, seed=8)
    before = x.copy()
    svd_factorize(x, 4)
    analyse_ranks(x, [1, 4])
    assert np.array_equal(x, before)


def test_chunked_reconstruction_agrees():
    x = make_synthetic(1000, rank=6, seed=9)
    u, v = svd_factorize(x, 6)
    assert np.allclose(reconstruct(u, v, chunk_size=100), reconstruct(u, v), atol=1e-6)
