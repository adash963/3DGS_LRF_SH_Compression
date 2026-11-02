# `scripts/compress_model.py`

## Purpose

Builds one compressed asset from one model and reports its measured size and
coefficient-space error. This is the single-configuration workhorse; `run_ablation.py`
does the same thing in bulk.

## Line-by-line

### Lines 1–12 — Header and imports

```python
"""Compress a model's SH: factorise (+ optional quantisation), save asset, report bytes."""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sh_compression.extraction import compress, extract_flat_sh
from sh_compression.storage import ratios
```

### Lines 15–25 — Arguments

```python
    ap.add_argument("--ply", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--mode", choices=["lowrank", "vq", "scalar"], default="lowrank")
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--scalar-bits", type=int, default=8)
    ap.add_argument("--factor-dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output .npz asset path")
```

The full configuration surface. `choices=` on `--mode` and `--factor-dtype` means
argparse rejects typos with a helpful message rather than letting them reach the
library and fail obscurely.

`--seed` defaults to 0 rather than being random — determinism by default, which is
what makes the VQ results reproducible.

### Lines 27–35 — Compression

```python
    _, flat = extract_flat_sh(args.ply)
    asset = compress(
        flat, rank=args.rank, mode=args.mode, codebook_size=args.codebook_size,
        scalar_bits=args.scalar_bits, seed=args.seed, factor_dtype=args.factor_dtype,
        source_model=os.path.abspath(args.ply),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    size = asset.save(args.out)
```

Line 33 records `os.path.abspath(args.ply)` as provenance inside the asset — the
absolute path resolves ambiguity about which checkpoint was used.

**Line 35 captures the measured size.** `asset.save()` returns
`os.path.getsize(...)`, so `size` is a filesystem fact, and every ratio computed
below inherits that property.

### Lines 37–52 — The report

```python
    rec = asset.reconstruct_flat()
    diff = rec - flat
    report = {
        "config": vars(args),
        "n_points": int(flat.shape[0]),
        "coeff_mse": float(np.mean(diff**2)),
        "coeff_rel_frobenius_error": float(np.linalg.norm(diff) / np.linalg.norm(flat)),
        "coeff_max_abs_error": float(np.max(np.abs(diff))),
        "asset_file_bytes": size,
        "component_bytes": asset.component_bytes(),
        **ratios(flat.shape[0], size, baseline_ply_bytes=os.path.getsize(args.ply)),
    }
```

Line 37 reconstructs **immediately after compressing** and line 38 measures the
error against the original. This is a round-trip check in miniature: it verifies the
asset decodes correctly before anything downstream depends on it.

- `vars(args)` embeds the entire configuration in the output, so a metrics file is
  self-describing — a reader never has to guess which settings produced it.
- Three error views (MSE, relative Frobenius, max absolute) for the reasons given in
  [factorization.md](factorization.md).
- `component_bytes()` records the storage breakdown — noting the caveat that these
  are **raw** array sizes while `asset_file_bytes` is zip-compressed.
- `**ratios(...)` merges the compression-ratio fields in. Passing
  `os.path.getsize(args.ply)` enables the whole-model calculation.

```python
    out_json = os.path.splitext(args.out)[0] + ".metrics.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2))
```

The metrics file is named after the asset (`foo.npz` → `foo.metrics.json`), keeping
them associated on disk.

Line 55 prints everything except `config` — the configuration was just typed on the
command line, so echoing it back adds noise. The file keeps it for the record; the
console shows only what was learned.

## Why it matters

This script embodies the project's measurement discipline in miniature. Every asset
it produces arrives with:

- its **measured** file size (not an estimate),
- its coefficient-space error (verified by immediate reconstruction),
- both compression ratios,
- and the complete configuration that produced it.

Nothing downstream has to remember how an asset was made, and no number in the
results can be traced to anything other than a file this script wrote.
