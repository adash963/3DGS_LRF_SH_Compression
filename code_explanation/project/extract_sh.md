# `scripts/extract_sh.py`

## Purpose

Dumps a model's SH matrix to a standalone `.npz` for offline analysis. A utility
rather than a pipeline stage — nothing in the main experiment depends on it — but it
makes the 205 MB matrix portable without carrying the 265 MB model around.

## Line-by-line

### Lines 1–11 — Header and imports

```python
"""Extract the flat SH matrix from a model PLY into an .npz for offline analysis."""
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sh_compression.extraction import extract_flat_sh
```

### Lines 14–24 — `main`

```python
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    _, flat = extract_flat_sh(args.ply)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, sh_flat=flat)
```

Line 18 does the work via the shared pipeline function — the same code path the
experiments use, so an extracted matrix is exactly what the compressor would see.

Line 19 creates the parent directory. `os.path.abspath` first, so a bare filename
like `out.npz` yields a valid parent rather than an empty string.

Line 20 saves under the key `sh_flat`, making the archive self-describing:
`np.load(path)["sh_flat"]` retrieves it.

```python
    print(f"wrote {args.out}: shape {flat.shape}, "
          f"{os.path.getsize(args.out if args.out.endswith('.npz') else args.out + '.npz'):,} bytes")
```

The conditional inside the f-string handles NumPy's habit of appending `.npz` when
the path lacks it — without it, `getsize` would fail on a path the user typed without
the extension. Slightly awkward inline, but correct.

## Why it matters

Two practical uses:

- **Portability.** The extracted matrix compresses to a fraction of the model and
  contains everything the factorisation needs, so rank analysis can be explored on a
  machine that never holds the full checkpoint.
- **Independent verification.** A marker or colleague can load the `.npz` in a
  notebook and run their own SVD, confirming the effective-rank result without
  trusting this project's code at all.

The second point is the more valuable one for a dissertation: it lowers the cost of
someone checking the central empirical claim.
