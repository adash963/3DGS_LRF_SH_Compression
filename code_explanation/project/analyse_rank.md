# `scripts/analyse_rank.py`

## Purpose

Runs the singular-value decomposition sweep over a model's SH matrix and emits three
CSVs plus three figures. **This script produced the dissertation's Stage 3 result:
effective rank 12 of 48.**

## Line-by-line

### Lines 1–14 — Header, imports, default ranks

```python
"""SVD rank analysis of a model's SH matrix -> CSVs + figures."""
import argparse, csv, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sh_compression.extraction import extract_flat_sh
from sh_compression.factorization import analyse_ranks, effective_rank

DEFAULT_RANKS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]
```

The default rank list is roughly logarithmic — dense at the low end where quality
changes fastest, sparser approaching full rank. It ends at 48, the maximum possible
for a 48-column matrix, which provides the exactness check (full rank must
reconstruct the input).

### Lines 17–25 — Arguments

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out-dir", default="results/experiments/rank_analysis")
    ap.add_argument("--ranks", type=int, nargs="*", default=DEFAULT_RANKS)
    ap.add_argument("--figures-dir", default="results/figures")
    args = ap.parse_args()
```

Defaults point at the committed results directories, so the command that regenerates
the dissertation's artefacts is just `python scripts/analyse_rank.py --ply <model>`.
`nargs="*"` allows overriding the sweep on the command line.

### Lines 27–29 — The analysis

```python
    _, flat = extract_flat_sh(args.ply)
    res = analyse_ranks(flat, args.ranks)
    os.makedirs(args.out_dir, exist_ok=True)
```

Two lines do all the work; everything after this is output formatting. `analyse_ranks`
computes one SVD and evaluates every requested rank against it (see
[factorization.md](factorization.md)).

### Lines 31–43 — CSV output

```python
    with open(os.path.join(args.out_dir, "singular_values.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["index", "singular_value"])
        for i, s in enumerate(res.singular_values):
            w.writerow([i, s])
```

The raw spectrum — all 48 singular values with their index. This is the most
fundamental artefact: every other rank-related number in the dissertation is
derivable from it.

```python
    rows = res.as_rows()
    with open(os.path.join(args.out_dir, "error_vs_rank.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)
```

The per-rank error table, using `DictWriter` so column headers derive from the data's
own keys rather than a hand-maintained list — if `RankAnalysis` gains a field, the
CSV gains a column automatically.

```python
    with open(os.path.join(args.out_dir, "energy_vs_rank.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rank", "explained_energy"])
        for r in rows:
            w.writerow([r["rank"], r["explained_energy"]])
```

Explained energy is duplicated into its own file even though it is a column of
`error_vs_rank.csv` — a small convenience for plotting that costs a few hundred bytes.

`newline=""` on each `open` is the documented way to stop Python's csv module
emitting blank lines between rows on Windows.

### Lines 45–50 — Console summary

```python
    eff = effective_rank(res.singular_values)
    print(f"points: {flat.shape[0]:,}; effective rank (99% energy): {eff}")
    for r in rows:
        print(f"rank {r['rank']:>3}: rel_err={r['rel_frobenius_error']:.5f} "
              f"energy={r['explained_energy']:.5f}")
```

**Line 45 computes the headline number.** On the real scene this prints
`effective rank (99% energy): 12` — the figure that appears throughout the report as
evidence that the SH matrix is genuinely low-rank.

The loop prints the full sweep, which is how the results table in
`stage_3_synthetic_tests.md` was populated.

### Lines 52–75 — Figure generation

```python
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
```

Wrapped in `try` so the script still produces its CSVs on a machine without
matplotlib — the data is the primary artefact, figures are derived.

`matplotlib.use("Agg")` selects the non-interactive backend **before** importing
`pyplot`. Required on headless machines (Colab, CI); without it matplotlib may try to
open a display and fail.

```python
        for name, xs, ys, xlab, ylab, logy in [
            ("singular_values", range(len(res.singular_values)), res.singular_values,
             "index", "singular value", True),
            ("energy_vs_rank", res.ranks, res.explained_energy,
             "rank", "cumulative explained energy", False),
            ("error_vs_rank", res.ranks, res.rel_frobenius_error,
             "rank", "relative Frobenius error", False),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(list(xs), list(ys), marker="o", ms=3)
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.figures_dir, f"{name}.png"), dpi=150)
            plt.close(fig)
```

Three figures from one table-driven loop rather than three near-identical blocks.

The `logy` flag applies a **log scale to the singular-value plot specifically**. This
is essential rather than cosmetic: the singular values span several orders of
magnitude, and on a linear axis everything after the first two would be flat against
zero, hiding exactly the decay structure the plot exists to show.

`plt.close(fig)` releases each figure — without it, matplotlib accumulates open
figures and warns.

### Lines 76–77 — Fallback

```python
    except ImportError:
        print("matplotlib unavailable; skipped figures")
```

Reports the omission rather than failing silently — consistent with the project's
practice of never hiding a skipped step.

## A known limitation

The static audit flagged that these figures are generated **while re-analysing the
PLY**, not by reading the committed CSVs. So the rank figures do not strictly satisfy
"all figures generated from committed machine-readable files" — they are generated
from the same in-memory analysis that writes those files.

The distinction is subtle (the numbers are identical, from one computation) but real:
regenerating the figures requires the 265 MB model, not just the committed CSVs. The
recommended fix is to split figure generation into a separate step that consumes
`singular_values.csv` and `error_vs_rank.csv`.

## Why it matters

This script produced the empirical foundation of the project. Before it ran, "SH
coefficients might be compressible" was a hypothesis. After it ran — rank 1 capturing
90% of the energy, 99% reached by rank 12 — it was a measured property of the data,
and the rest of the pipeline had a justified reason to exist.
