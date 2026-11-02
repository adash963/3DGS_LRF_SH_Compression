# `tests/unit/test_serialization.py`

## Purpose

Verifies the compressed asset format: that saved files reload correctly, that the
reported size is the real file size, that corrupted or version-mismatched files are
rejected, and that byte accounting is right.

## Line-by-line

### Lines 12–15 — Shared fixture

```python
@pytest.fixture
def flat():
    return make_synthetic(300, rank=6, noise=0.05, seed=0)
```

A 300×48 matrix with genuine rank-6 structure plus light noise — realistic enough
that rank-6 compression is meaningful, small enough to be instant.

### Lines 18–31 — `test_save_load_preserves_reconstruction`

```python
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
```

**The central format test**, run against all four storage variants.

Line 27 is the one that backs the dissertation's measurement discipline:
`reported == os.path.getsize(path)`. Every compression ratio in the report divides by
a number that came from `save()`, and this asserts that number is the filesystem's,
not an estimate. If `save()` ever returned a computed value, this test would fail.

Line 29 checks the round trip preserves *reconstruction*, not just bytes. That is the
property that actually matters — a file could differ in irrelevant ways yet still
reconstruct identically, and conversely could load without error yet reconstruct
wrongly.

Line 30 confirms the metadata survived.

Parametrising over all four modes means a serialisation bug affecting only, say, the
scalar branch cannot hide.

### Lines 34–38 — `test_full_rank_lowrank_matches_original`

```python
def test_full_rank_lowrank_matches_original(tmp_path, flat):
    asset = compress(flat, rank=48, mode="lowrank")
    rec = asset.reconstruct_flat()
    assert np.linalg.norm(rec - flat) / np.linalg.norm(flat) < 1e-5
```

Full-rank compression through the *asset* path must reproduce the input. This is the
synthetic analogue of the real-model result where `lr_r48_fp32` rendered at 101 dB
against the baseline.

It differs from the equivalent test in `test_factorization.py` by going through
`CompressedSH` rather than calling the factoriser directly — so it also exercises the
container and its reconstruction dispatch.

### Lines 41–47 — `test_component_bytes_accounting`

```python
def test_component_bytes_accounting(flat):
    asset = compress(flat, rank=6, mode="vq", codebook_size=32)
    comp = asset.component_bytes()
    assert set(comp) == {"v", "codebook", "indices", "metadata"}
    assert comp["indices"] == 300  # uint8 per point
    assert comp["v"] == 6 * 48 * 4
```

Verifies the storage breakdown against hand-computed values.

- `indices == 300` — one uint8 per point, confirming the dtype selection picked uint8
  for a 32-entry codebook.
- `v == 6 * 48 * 4` — rank 6, 48 columns, 4 bytes per float32.

Asserting exact expected byte counts is what makes the storage-breakdown figure
trustworthy: the numbers are checked against arithmetic a reader can verify, not just
against themselves.

Note this tests **raw** component sizes, which — as the static audit observed — do
not sum to the compressed file size. The test is correct about what it measures; the
report needs to label the figure accordingly.

### Lines 50–63 — `test_unsupported_version_rejected`

```python
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
```

Saves a valid asset, **surgically rewrites its version to 999**, and confirms loading
fails.

This exercises the guard described in [init.md](init.md). Editing a real file rather
than mocking the check means the whole path is tested — metadata encoding, decoding,
and comparison.

`match="format_version"` ensures the failure is the version check specifically, not
some incidental error caused by the file surgery.

### Lines 66–76 — `test_corrupt_shape_rejected`

```python
def test_corrupt_shape_rejected(tmp_path, flat):
    asset = compress(flat, rank=4)
    ...
    arrays["u"] = arrays["u"][:100]  # truncate U
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="corrupt"):
        CompressedSH.load(path)
```

Truncates `U` from 300 rows to 100 and confirms `_validate` catches the mismatch
against the declared `n_points`.

This simulates real corruption — an interrupted write, a truncated download. Without
validation the asset would load and then reconstruct only 100 of 300 Gaussians, or
raise an obscure shape error much later.

### Lines 79–84 — `test_deterministic_output`

```python
def test_deterministic_output(tmp_path, flat):
    a1 = compress(flat, rank=6, mode="vq", codebook_size=32, seed=5)
    a2 = compress(flat, rank=6, mode="vq", codebook_size=32, seed=5)
    assert np.array_equal(a1.vq.indices, a2.vq.indices)
    assert np.array_equal(a1.vq.codebook, a2.vq.codebook)
```

Determinism again, this time through the full `compress` entry point rather than the
quantiser alone — confirming the seed is actually threaded through the pipeline
rather than dropped somewhere in between.

## Why it matters

Three dissertation claims are tested here:

1. **"Reported size equals actual file size"** — line 27, in all four modes.
2. **"Assets load correctly in a clean process"** — every load in this file works
   from the file alone, with no reliance on the object that wrote it.
3. **"Corrupted files are detected where practical"** — version and shape corruption
   both produce clear, specific failures.

The version and corruption tests are cheap insurance of the kind that only proves its
value when something goes wrong — but the failure they prevent (silently
reconstructing wrong values from a subtly bad file) is exactly the kind that would
undermine every result downstream.
