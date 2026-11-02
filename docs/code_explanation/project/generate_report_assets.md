# `scripts/generate_report_assets.py`

## Purpose

Produces every figure and the results table for the report, reading **only** from
committed result files. This script exists to satisfy one of the brief's strictest
rules:

> Generate graphs from machine-readable result files. Do not manually enter
> experimental values into charts.

## Line-by-line

### Lines 1–12 — Header and imports

```python
"""Generate rate-distortion and storage figures from machine-readable result files."""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

`matplotlib.use("Agg")` before importing `pyplot` — the non-interactive backend,
required on headless machines.

### Lines 15–27 — Loading results

```python
def load_rows(ablation_dir: str) -> list[dict]:
    with open(os.path.join(ablation_dir, "all_results.json")) as f:
        return [r for r in json.load(f)["rows"] if r.get("status") == "ok"]
```

Reads the aggregate JSON and keeps only successful runs. Failed runs are excluded
from *plots* (they have no data to plot) but remain in the source file — the record
is complete even though the figure is not cluttered.

```python
def attach_image_metrics(rows: list[dict], renders_metrics_dir: str) -> None:
    """Merge per-experiment image metrics (metrics_<name>.json) when present."""
    for r in rows:
        p = os.path.join(renders_metrics_dir, f"metrics_{r['name']}.json")
        if os.path.exists(p):
            with open(p) as f:
                r.update(json.load(f).get("summary", {}))
```

**This function joins the two halves of the experiment.** Storage results (all 22
configurations) come from the ablation JSON; image quality (12 configurations) comes
from separate per-experiment files. Matching on `name` merges them.

The `os.path.exists` check means configurations without renders simply lack quality
fields, and the plotting code below skips them — which is why the rate-distortion
figures show 12 points while the coefficient-error figure shows 22.

### Lines 38–60 — The `scatter` helper

```python
    def scatter(xkey, ykey, fname, xlabel, ylabel, logx=False):
        pts = [(r[xkey], r[ykey], r["name"], r.get("mode", "?")) for r in rows
               if xkey in r and ykey in r]
        if not pts:
            print(f"skip {fname}: no data for {ykey}")
            return
```

A closure over `rows` and `args`, producing every line plot.

Lines 39–40 select only rows possessing both keys. Lines 41–43 **report the skip
explicitly** rather than producing an empty figure — this is how the LPIPS figure was
correctly skipped before the LPIPS pass ran, and how its later appearance was
confirmed.

```python
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        by_mode = {}
        for x, y, name, mode in pts:
            by_mode.setdefault(mode, []).append((x, y, name))
        for mode, items in sorted(by_mode.items()):
            items.sort()
            ax.plot([i[0] for i in items], [i[1] for i in items],
                    marker="o", ms=4, label=mode)
```

**Points are grouped by mode** (lowrank / scalar / vq) into separate labelled series.
This is what makes the rate-distortion figures readable: the three methods form
visibly distinct curves, and the VQ cluster's separation from the others is
immediately apparent rather than buried in a single undifferentiated scatter.

`items.sort()` orders by x so lines connect left to right rather than zigzagging.

```python
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.figures_dir, fname), dpi=150)
        plt.close(fig)
        print(f"wrote {fname} ({len(pts)} points)")
```

**`logx` is essential for the rate-distortion plots.** Asset sizes span 778 KB to
194 MB — a factor of 250. On a linear axis every VQ configuration would collapse
against the origin, hiding the entire high-compression regime the project cares
about.

Line 58 prints the filename and point count, giving an auditable record of what each
run produced.

### Lines 61–62, 86–95 — The figure set

```python
    scatter("rank", "coeff_rel_frobenius_error", "coeff_error_vs_rank.png", ...)
    ...
    scatter("asset_file_bytes", "gt_psnr", "rate_distortion_psnr.png", ...)
    scatter("asset_file_bytes", "gt_ssim", "rate_distortion_ssim.png", ...)
    scatter("asset_file_bytes", "gt_lpips", "rate_distortion_lpips.png", ...)
    scatter("asset_file_bytes", "vsbase_psnr", "rate_distortion_vsbase_psnr.png", ...)
    scatter("rank", "gt_psnr", "psnr_vs_rank.png", ...)
```

Six figures. Note both reference frames appear: `gt_*` against ground truth and
`vsbase_psnr` against baseline renders — the two comparisons discussed in
[run_render_eval.md](run_render_eval.md), each getting its own plot rather than being
conflated.

### Lines 64–85 — The results table

```python
    tbl_keys = ["name", "mode", "rank", "codebook_size", "factor_dtype",
                "asset_file_bytes", "sh_compression_ratio", "whole_model_ratio",
                "coeff_rel_frobenius_error", "gt_psnr", "gt_ssim", "gt_lpips",
                "vsbase_psnr", "vsbase_ssim", "vsbase_lpips"]
    present = [k for k in tbl_keys if any(k in r for r in rows)]
```

Emits a markdown table for direct inclusion in the report. Line 69 keeps only columns
that have data somewhere, so the table does not carry empty columns before the LPIPS
pass has run.

```python
    for r in sorted(rows, key=lambda x: (x.get("mode", ""), x.get("rank", 0))):
        cells = []
        for k in present:
            v = r.get(k, "")
            if isinstance(v, float):
                v = f"{v:.4g}"
            elif isinstance(v, int) and k == "asset_file_bytes":
                v = f"{v:,}"
            cells.append(str(v))
```

Sorted by mode then rank, so related configurations sit together. Floats get four
significant figures; byte counts get thousands separators.

**This table is the direct answer to "no hand-entered values in the report."** The
numbers in the results section are copied from a file this script generated, and
regenerating it after any change updates them automatically.

### Lines 97–120 — Storage breakdown

```python
    comp_rows = []
    for r in rows:
        apath = os.path.join(args.ablation_dir, r["name"], "asset.npz")
        if os.path.exists(apath):
            from sh_compression.serialization import CompressedSH
            comp = CompressedSH.load(apath).component_bytes()
            comp_rows.append((r["name"], comp))
```

Builds the stacked-bar storage breakdown by **loading each asset** and querying its
components.

Two caveats, both raised by the static audit:

1. **This reads gitignored `.npz` files.** On a fresh clone the assets do not exist,
   `comp_rows` is empty, and the figure is silently skipped. So this particular
   figure is *not* reproducible from the committed repository alone — it requires
   re-running the ablation first.
2. **`component_bytes()` returns raw array sizes**, while the reported asset size is
   zip-compressed. The bars therefore sum to more than the file size and must be
   labelled as a raw-payload breakdown, not a partition of the compressed file.

```python
        keys = sorted({k for _, c in comp_rows for k in c})
        bottoms = [0] * len(comp_rows)
        for k in keys:
            vals = [c.get(k, 0) / 1024 for _, c in comp_rows]
            ax.bar([n for n, _ in comp_rows], vals, bottom=bottoms, label=k)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
```

Manual stacking: each component is drawn starting at the running total of those
already placed. Values are divided by 1024 to display KiB.

## Why it matters

This script is the mechanism behind the report's traceability claim. No figure in the
dissertation was drawn by hand or populated by copying numbers out of a notebook —
each is regenerated by one command from files under `results/`.

The audit's fair qualification: two figures fall short of the strongest reading of
that claim. The rank figures are generated by `analyse_rank.py` while re-analysing
the model rather than by reading the committed CSVs, and the storage breakdown
depends on uncommitted assets. Both are noted in the outstanding corrections; neither
affects any reported number.
