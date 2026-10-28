# Stage 3 verification — synthetic data and effective-rank analysis

Date: 2026-08-02. Commit: a9feb6b.

## Synthetic mathematics (tests/unit/test_factorization.py, 10 tests)

- exact rank-1/2/4 data reconstructs at matching rank, rel. Frobenius error < 1e-6 ✓
- rank 3 error > rank 4 error on exact rank-4 data ✓
- error monotone non-increasing, energy monotone non-decreasing in rank ✓
- full-rank reconstruction matches input (< 1e-5) ✓
- constant data is rank 1 ✓
- ranks above max skipped; invalid ranks / NaN / 1-D inputs rejected ✓
- input matrices never mutated ✓; chunked == unbatched reconstruction ✓

All pass on macOS (arm64) and Colab (x86_64) — cross-platform numerical consistency.
(No CUDA SVD path is used; factorisation is CPU float64 by design.)

## Real-scene rank analysis (T&T train, N = 1,071,462)

From results/experiments/rank_analysis/ (CSVs) and results/figures/:

    effective rank (99% cumulative energy): 12 of 48
    rank  1: rel_err 0.3148, energy 0.9009
    rank  4: rel_err 0.1755, energy 0.9692
    rank  8: rel_err 0.1327, energy 0.9824
    rank 12: rel_err 0.0936, energy 0.9912
    rank 16: rel_err 0.0651, energy 0.9958
    rank 48: rel_err 0.0000, energy 1.0000

Full-rank (48) reconstruction of the real matrix is numerically exact. STAGE 3 PASSED.
