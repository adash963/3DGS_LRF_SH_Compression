# `scripts/evaluate_model.py`

## Purpose

Compares two directories of rendered images and writes per-view plus mean metrics.
A thin CLI wrapper over `metrics.compare_dirs`, used for one-off comparisons — the
bulk evaluation runs inside `run_render_eval.py`.

## Line-by-line

### Lines 1–10 — Header and imports

```python
"""Compute PSNR/SSIM/(LPIPS) between two render directories."""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sh_compression.metrics import compare_dirs, save_metrics
```

### Lines 13–21 — Arguments

```python
    ap.add_argument("--dir-a", required=True, help="reference renders / ground truth")
    ap.add_argument("--dir-b", required=True, help="candidate renders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-lpips", action="store_true")
```

The help text fixes the convention: `dir-a` is the reference, `dir-b` the candidate.
The metrics themselves are symmetric, but the labelling matters for interpreting a
saved file — and both directory paths are recorded in the output, so the comparison
is unambiguous after the fact.

`--no-lpips` allows skipping the perceptual metric, which was used during the main
evaluation run because LPIPS on Colab's CPU would have added substantial time. It was
computed later in a dedicated GPU pass.

### Lines 22–27 — Execution and reporting

```python
    result = compare_dirs(args.dir_a, args.dir_b, use_lpips=not args.no_lpips)
    save_metrics(result, args.out)
    print(json.dumps(result["summary"], indent=2))
    if result["failed_views"]:
        print(f"WARNING: {len(result['failed_views'])} failed views", file=sys.stderr)
```

Line 22 does the work. Line 23 writes the complete result — every per-view row, the
failures, the directory paths — while line 24 prints only the summary means, keeping
console output readable.

**Lines 25–26 surface failures on stderr.** Since only the summary is printed, a
failed view would otherwise be invisible on the console even though it is recorded in
the file. Routing the warning to stderr means it still appears when stdout is piped
elsewhere, which is how it was invoked from the notebook.

## Where it was used

This script produced the Stage 2 verification measurement — comparing the round-trip
renders against the baseline renders:

```bash
python scripts/evaluate_model.py \
  --dir-a .../test/ours_30000/renders \
  --dir-b /content/work/renders/roundtrip \
  --out /content/work/metrics_roundtrip.json --no-lpips
```

which returned `mean_ssim: 1.0` and, on inspecting the saved per-view rows, PSNR of
`inf` on all 38 views — the pixel-identical result.

That saved JSON is the artefact the static audit noted was never committed: it lived
in the Colab runtime, which has since been deleted. The finding is recorded in the
stage evidence, but the machine-readable file is not in the repository. Either
re-running the comparison or rewording the claim as a recorded run outcome would
close that gap.

## Why it matters

Having evaluation available as a standalone command — not only inside the big driver
— is what made the Stage 2 check cheap to run at the moment it was needed. It also
means anyone can re-verify a comparison against committed renders without invoking
the full pipeline.
