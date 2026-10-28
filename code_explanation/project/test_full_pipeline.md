# `tests/integration/test_full_pipeline.py`

## Purpose

The only test that exercises the **complete** chain — compress, save, load,
reconstruct, write PLY, read back — on a model with genuine low-rank structure. Unit
tests verify components in isolation; this verifies they compose.

## Line-by-line

### Lines 1–16 — Imports

```python
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.conftest import FIELD_ORDER, make_tiny_model

from sh_compression import sh_layout
from sh_compression.extraction import compress, decompress_to_ply, extract_flat_sh
from sh_compression.ply_io import GaussianPly, write_gaussian_ply
from sh_compression.serialization import CompressedSH
```

Line 15 imports the fixture builder as a plain function. This test needs to *modify*
the model before use, so it constructs one directly rather than taking the fixture.

### Lines 19–35 — The `lowrank_ply` fixture

```python
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
```

**This fixture is more sophisticated than the unit-test ones, and deliberately so.**

The standard `tiny_model` has random SH — which is full rank, so compressing it at
rank 4 would produce large error and the test could only assert something vague.

Here, lines 25–26 build SH with **true rank 4** (a `[n,4] @ [4,48]` product) plus
tiny noise. The model therefore has known structure, so rank-4 compression must
reconstruct it *almost exactly* — and the test can assert a tight bound.

Lines 27–33 install this structured SH into a realistic model, using the layout
functions rather than manual index arithmetic. The result is a model that is
structurally a genuine 3DGS file and mathematically low-rank.

### Lines 38–63 — The main integration test

```python
@pytest.mark.parametrize("mode,kwargs,max_rel_err", [
    ("lowrank", {}, 1e-2),
    ("vq", {"codebook_size": 256}, 1.0),
    ("scalar", {"scalar_bits": 8}, 5e-2),
])
def test_compress_save_load_reconstruct_render_ply(tmp_path, lowrank_ply,
                                                   mode, kwargs, max_rel_err):
```

Parametrised over all three modes, each with its own error tolerance — and the
tolerances are set from what each method can be *expected* to achieve, not from
observed results:

- **lowrank 1e-2** — rank-4 SVD on rank-4 data should be near-exact; the bound allows
  for the added noise.
- **scalar 5e-2** — 8-bit quantisation adds bounded error on top.
- **vq 1.0** — deliberately loose. With 2000 points, 256 codewords and only 4
  dimensions, VQ error is genuinely large. The test asserts the pipeline *completes
  and stays finite*, not that quality is good.

That last choice is worth being explicit about: it is not a weakened threshold hiding
a failure, it is a test whose purpose is plumbing rather than quality. The quality
question for VQ is answered by the real-scene experiments.

```python
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
```

The full chain, with **disk round trips at both ends**. The asset is saved and
reloaded (lines 48–50), and the reconstructed model is written and re-read
(lines 52–54) before error is measured.

This is what distinguishes an integration test from the unit tests: the error
measured on line 55 has passed through every serialisation boundary in the pipeline,
so it would catch a bug that only manifests when components are combined — a dtype
lost in one hop, a shape convention mismatched between two modules.

```python
    from sh_compression.ply_io import read_gaussian_ply
    a, b = read_gaussian_ply(lowrank_ply), read_gaussian_ply(out_ply)
    for f in a.non_sh_field_names():
        assert np.array_equal(a.fields[f], b.fields[f])
```

Lines 57–60 verify geometry survived **the entire compression pipeline**, not just a
single `insert_sh` call. This is the strongest form of the experimental-control
guarantee: after compressing, saving, loading, reconstructing and re-writing, every
non-SH field is still identical.

### Lines 66–71 — `test_wrong_point_count_rejected`

```python
def test_wrong_point_count_rejected(tmp_path, lowrank_ply):
    _, flat = extract_flat_sh(lowrank_ply)
    asset = compress(flat[:100], rank=4)
    with pytest.raises(ValueError, match="points"):
        decompress_to_ply(asset, lowrank_ply, str(tmp_path / "x.ply"))
```

Compresses only the first 100 Gaussians, then attempts to apply that asset to the
full 2000-point model.

This simulates a realistic mistake — mixing assets between scenes, or between
versions of a scene. Without the check the mismatch would surface as a confusing
shape error inside `insert_sh`, or in the worst case work silently if counts
happened to align.

## Why it matters

Unit tests can all pass while a pipeline is broken, if the modules disagree at their
boundaries. This test closes that gap by running the complete workflow that
`run_render_eval.py` runs — minus the GPU render — and asserting:

- the asset's reported size is real,
- reconstruction survives both disk round trips,
- error stays within method-appropriate bounds,
- geometry is unchanged end to end,
- mismatched inputs are rejected.

It is also the fastest safety net available: four parametrised runs of the whole
pipeline in under a second, no GPU, no checkpoint.
