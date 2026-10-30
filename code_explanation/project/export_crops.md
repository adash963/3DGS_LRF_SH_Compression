# `scripts/export_crops.py`

## Purpose

Prints base64-encoded JPEG crops of rendered views to stdout, so images can be moved
out of a Colab runtime **without credentials** — no Drive mount, no upload token, no
authentication of any kind.

This is a data-transfer workaround, and its existence is itself a documented
engineering finding.

## The problem it solves

Renders live in the Colab runtime's ephemeral filesystem. Getting them into the
repository normally means mounting Google Drive (which requires an interactive
authorisation the user must click) or configuring credentials inside the notebook
(which should never be done).

The alternative used here: print the image as text, read that text back through the
browser automation that was already driving the notebook, and decode it locally.
Slow and inelegant, but it needs no permissions and creates no credential exposure.

## Line-by-line

### Lines 1–12 — Header and imports

```python
"""Print base64 JPEG crops of rendered views for transfer through notebook output.

Used by the Colab session to move small visual-comparison crops into the repo
without credentials. Markers: IMG:<name>:START / IMG:<name>:END per image.
"""
import argparse, base64, io, os
from PIL import Image
```

The marker convention is the protocol: each image is delimited so a regular
expression can extract exactly one image's payload from a page of mixed output.

### Lines 15–25 — Arguments

```python
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--renders-root", required=True, help="dir containing <config>/ render dirs")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--view", default="00012.png")
    ap.add_argument("--box", type=int, nargs=4, default=[550, 450, 950, 800],
                    help="left top right bottom")
    ap.add_argument("--quality", type=int, default=85)
```

`--box` takes four integers for the crop rectangle. **The default is wrong for these
renders** — it extends to y=800 while the test views are only ~545 pixels tall, so
the first run produced images that were half black. The actual figure was generated
with `--box 300 120 700 470`.

The default was left unchanged rather than silently corrected, since the correct box
is scene- and view-specific; the lesson is recorded in `dash.md`'s engineering notes.

`--quality 85` is a size/fidelity compromise. JPEG is lossy, which is acceptable for
a visual comparison figure but would not be for anything measured — the metrics are
computed on the original PNGs, never on these crops.

### Lines 26–32 — Source resolution

```python
    sources = {"gt": os.path.join(args.gt_dir, args.view),
               "baseline": os.path.join(args.baseline_dir, args.view)}
    for c in args.configs:
        sources[c] = os.path.join(args.renders_root, c, args.view)
```

Ground truth and baseline come from the checkpoint's own output directories; every
other configuration comes from `<renders_root>/<config>/`. All use the **same view
filename**, so the crops are directly comparable — the same camera, the same pixels,
differing only in the model that produced them.

### Lines 34–42 — Encode and emit

```python
    for name, path in sources.items():
        buf = io.BytesIO()
        Image.open(path).convert("RGB").crop(tuple(args.box)).save(
            buf, "JPEG", quality=args.quality)
        b = base64.b64encode(buf.getvalue()).decode()
        print(f"IMG:{name}:START")
        for i in range(0, len(b), 400):
            print(b[i:i + 400])
        print(f"IMG:{name}:END")
```

- Line 36 uses an in-memory buffer, avoiding temporary files.
- `convert("RGB")` normalises away any alpha channel, which JPEG cannot store.
- **Lines 40–41 chunk the base64 into 400-character lines.** This is the critical
  detail. An earlier attempt printed a single 55 KB base64 string and Colab's output
  DOM silently truncated it — the decode then failed with an "incorrect padding"
  error, which is a confusing symptom for a truncation cause. Chunking keeps every
  line short enough to survive the renderer.

## Why it matters

Two contributions.

**It preserved a security boundary.** The alternative approaches all involve
credentials in a notebook or an interactive authorisation. This needs neither, so
image transfer never required handling the user's Google credentials.

**It documented a real constraint.** The truncation failure and its fix are recorded
in `DEVELOPMENT_LOG.md` and `dash.md`. Anyone else driving Colab programmatically
will hit the same limit, and the diagnosis — silent DOM truncation presenting as a
base64 padding error — is not obvious from the symptom.

The images it produced are the six panels of `results/figures/visual_comparisons.png`,
including the VQ K=64 failure case where the hillside desaturates to grey haze.
