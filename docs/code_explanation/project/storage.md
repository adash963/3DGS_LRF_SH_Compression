# `src/sh_compression/storage.py`

## Purpose

Byte accounting and compression ratios. Only 30 lines, but it defines what
"compression ratio" *means* in this dissertation — and because that term is used
inconsistently across the 3DGS compression literature, defining it precisely and in
one place is a substantive contribution to the report's defensibility.

## Line-by-line

### Lines 1–7 — Imports

```python
"""Byte accounting: measured file sizes and compression ratios."""

import os
from .sh_layout import FLAT_DIM
```

`os` for filesystem queries, `FLAT_DIM` (48) so the baseline arithmetic cannot drift
out of step with the layout definition.

### Lines 8–10 — `file_bytes`

```python
def file_bytes(path: str) -> int:
    return os.path.getsize(path)
```

A one-line wrapper over `os.path.getsize`. Trivial, but it exists to make the
intent greppable: every size in the results traces back to a filesystem query, and
this function is the vocabulary for that.

### Lines 13–16 — `baseline_sh_bytes`

```python
def baseline_sh_bytes(n_points: int) -> int:
    """Uncompressed SH storage in the baseline PLY: N x 48 x float32."""
    return n_points * FLAT_DIM * 4
```

The denominator-defining function. For the studied scene:
`1,071,462 × 48 × 4 = 205,720,704 bytes`.

This is *computed* rather than measured, and that is correct: the SH block is not a
separate file, it is 48 of the 62 columns interleaved inside the PLY. But the
arithmetic is exact rather than approximate — the PLY format stores these as
contiguous float32 properties, so N × 48 × 4 is precisely the number of bytes those
columns occupy.

### Lines 19–30 — `ratios`

```python
def ratios(n_points: int, compressed_sh_file_bytes: int,
           baseline_ply_bytes: int | None = None) -> dict:
    sh_base = baseline_sh_bytes(n_points)
    out = {
        "baseline_sh_bytes": sh_base,
        "compressed_sh_bytes": compressed_sh_file_bytes,
        "sh_compression_ratio": sh_base / compressed_sh_file_bytes,
    }
```

**The headline ratio.** `sh_compression_ratio` = uncompressed SH bytes ÷ measured
`.npz` file size. For `sq_r16_b8`: `205,720,704 / 13,670,820 = 15.05`.

The parameter name `compressed_sh_file_bytes` is doing deliberate work — the word
*file* signals this must come from `os.path.getsize`, not from an estimate. Callers
pass the return value of `CompressedSH.save()`, which is exactly that.

Note the output dict returns the inputs alongside the ratio, so a saved metrics file
records the numerator and denominator, not just the quotient. A reader can verify the
arithmetic without re-running anything.

```python
    if baseline_ply_bytes is not None:
        non_sh = baseline_ply_bytes - sh_base
        out["baseline_ply_bytes"] = baseline_ply_bytes
        out["whole_model_bytes_compressed"] = non_sh + compressed_sh_file_bytes
        out["whole_model_ratio"] = baseline_ply_bytes / (non_sh + compressed_sh_file_bytes)
    return out
```

**The whole-model ratio**, computed only when the baseline file size is supplied.

- Line 25 derives the non-SH portion by subtraction: everything in the PLY that is
  not spherical harmonics — positions, normals, opacity, scales, rotations. For the
  studied scene: `265,724,108 − 205,720,704 = 60,003,404 bytes`.
- Line 27 forms the composite: uncompressed geometry plus compressed colour. For
  `sq_r16_b8`: `60,003,404 + 13,670,820 = 73,674,224 bytes`.
- Line 28 divides: `265,724,108 / 73,674,224 = 3.61×`.

**An important honesty point the static audit raised.** The 73.7 MB figure is a
*calculated deployable composite*, not a file that exists on disk. No artefact of
that size is emitted anywhere in the pipeline — the reconstructed PLY is always full
size, because it contains reconstructed (not compressed) SH. The field name
`whole_model_bytes_compressed` is arguably too assertive; the report should describe
it as a calculated composite size, which is what it is.

Reporting both ratios is the honest approach. The SH-only ratio (15×) measures what
this project actually does. The whole-model ratio (3.6×) is the number that matters
to someone shipping a scene, and is necessarily smaller because geometry — untouched
by this work — becomes the dominant cost once colour is compressed. Quoting only the
larger number would overstate the practical benefit; quoting only the smaller would
understate the method's effect on its actual target.

## Why it matters

Compression papers vary in what they divide by, which makes cross-paper comparison
treacherous. By defining both ratios explicitly, computing them from measured file
sizes, and reporting them side by side, this file lets the dissertation state
precisely what its numbers mean.

It also underpins the report's most important comparison caveat: this project's 15×
is *SH-only and post-hoc*, whereas published figures like LightGaussian's reported
15× are *whole-model with retraining*. Those are different quantities, and having
both defined in code makes the distinction easy to state accurately rather than
accidentally claiming parity.
