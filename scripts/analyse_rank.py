"""SVD rank analysis of a model's SH matrix -> CSVs + figures."""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sh_compression.extraction import extract_flat_sh
from sh_compression.factorization import analyse_ranks, effective_rank

DEFAULT_RANKS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out-dir", default="results/experiments/rank_analysis")
    ap.add_argument("--ranks", type=int, nargs="*", default=DEFAULT_RANKS)
    ap.add_argument("--figures-dir", default="results/figures")
    args = ap.parse_args()

    _, flat = extract_flat_sh(args.ply)
    res = analyse_ranks(flat, args.ranks)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "singular_values.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["index", "singular_value"])
        for i, s in enumerate(res.singular_values):
            w.writerow([i, s])
    rows = res.as_rows()
    with open(os.path.join(args.out_dir, "error_vs_rank.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "energy_vs_rank.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rank", "explained_energy"])
        for r in rows:
            w.writerow([r["rank"], r["explained_energy"]])

    eff = effective_rank(res.singular_values)
    print(f"points: {flat.shape[0]:,}; effective rank (99% energy): {eff}")
    for r in rows:
        print(f"rank {r['rank']:>3}: rel_err={r['rel_frobenius_error']:.5f} "
              f"energy={r['explained_energy']:.5f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(args.figures_dir, exist_ok=True)
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
        print(f"figures written to {args.figures_dir}")
    except ImportError:
        print("matplotlib unavailable; skipped figures")


if __name__ == "__main__":
    main()
