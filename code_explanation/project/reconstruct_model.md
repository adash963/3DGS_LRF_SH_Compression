# `scripts/reconstruct_model.py`

## Purpose

Turns a compressed `.npz` asset back into a renderable PLY. Small, but it carries one
of the project's most important warnings.

## Line-by-line

### Lines 1–11 — Header and imports

```python
"""Reconstruct a renderable PLY from a compressed asset + the baseline model."""
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sh_compression.extraction import decompress_to_ply
from sh_compression.serialization import CompressedSH
```

Note it needs **two** inputs: the asset and the baseline model. The asset holds only
compressed *colour*; geometry comes from the baseline. That division is the whole
point of the method — geometry is never touched, so it is never stored twice.

### Lines 14–23 — `main`

```python
    ap.add_argument("--asset", required=True)
    ap.add_argument("--baseline-ply", required=True)
    ap.add_argument("--out-ply", required=True)
    args = ap.parse_args()
    asset = CompressedSH.load(args.asset)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_ply)), exist_ok=True)
    decompress_to_ply(asset, args.baseline_ply, args.out_ply)
```

Line 18 loads with full validation — version check, mode check, layout check, shape
checks (see [serialization.md](serialization.md)). A corrupted or incompatible asset
fails here with a clear message rather than producing a broken PLY.

Line 20 delegates to the pipeline function, which also verifies the asset's point
count matches the model's.

### Lines 21–22 — The warning

```python
    print(f"wrote {args.out_ply} ({os.path.getsize(args.out_ply):,} bytes) — "
          "temporary render artefact, NOT the compressed size")
```

**This is the important line.** The output PLY is always ~265 MB, because it contains
uncompressed geometry plus *reconstructed* (not compressed) SH coefficients.

Someone measuring that file and reporting it as the compressed size would compute a
compression ratio near 1.0 and conclude the method achieved nothing. The brief called
this out explicitly as an error to avoid:

> Do not confuse the size of a temporary reconstructed PLY file with the deployed
> compressed representation size.

So the warning is printed **on every invocation**, alongside the size itself. Anyone
reading the console output sees the number and the caveat together — the caveat
cannot be missed by someone skimming.

The same warning appears in `decompress_to_ply`'s docstring, in the report's
limitations, and in `dash.md`. Repeating it at every layer is deliberate: it is the
single most plausible way for an outside reader to misinterpret this project's
results.

## Why it matters

The file demonstrates the storage model. Deployment would ship:

- the compressed SH asset — 13.7 MB for the headline configuration,
- plus geometry — 60 MB, unchanged by this work,

and reconstruct the full PLY at load time. The 265 MB reconstructed file is an
intermediate that exists only because the unmodified renderer reads PLYs; a
production integration would decode into GPU memory directly, never writing it.

That is also why the whole-model ratio (3.6×) is *calculated* rather than measured —
the composite artefact it describes is never emitted by this pipeline, a distinction
the static audit was right to insist on.
