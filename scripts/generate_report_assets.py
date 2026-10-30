"""Generate rate-distortion and storage figures from machine-readable result files."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(ablation_dir: str) -> list[dict]:
    with open(os.path.join(ablation_dir, "all_results.json")) as f:
        return [r for r in json.load(f)["rows"] if r.get("status") == "ok"]


def attach_image_metrics(rows: list[dict], renders_metrics_dir: str) -> None:
    """Merge per-experiment image metrics (metrics_<name>.json) when present."""
    for r in rows:
        p = os.path.join(renders_metrics_dir, f"metrics_{r['name']}.json")
        if os.path.exists(p):
            with open(p) as f:
                r.update(json.load(f).get("summary", {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", default="results/experiments/ablation")
    ap.add_argument("--image-metrics-dir", default="results/experiments/image_metrics")
    ap.add_argument("--figures-dir", default="results/figures")
    args = ap.parse_args()
    rows = load_rows(args.ablation_dir)
    attach_image_metrics(rows, args.image_metrics_dir)
    os.makedirs(args.figures_dir, exist_ok=True)

    def scatter(xkey, ykey, fname, xlabel, ylabel, logx=False):
        pts = [(r[xkey], r[ykey], r["name"], r.get("mode", "?")) for r in rows
               if xkey in r and ykey in r]
        if not pts:
            print(f"skip {fname}: no data for {ykey}")
            return
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        by_mode = {}
        for x, y, name, mode in pts:
            by_mode.setdefault(mode, []).append((x, y, name))
        for mode, items in sorted(by_mode.items()):
            items.sort()
            ax.plot([i[0] for i in items], [i[1] for i in items],
                    marker="o", ms=4, label=mode)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.figures_dir, fname), dpi=150)
        plt.close(fig)
        print(f"wrote {fname} ({len(pts)} points)")

    scatter("rank", "coeff_rel_frobenius_error", "coeff_error_vs_rank.png",
            "rank", "relative Frobenius error")

    # markdown results table for the report (traceable to all_results.json)
    tbl_keys = ["name", "mode", "rank", "codebook_size", "factor_dtype",
                "asset_file_bytes", "sh_compression_ratio", "whole_model_ratio",
                "coeff_rel_frobenius_error", "gt_psnr", "gt_ssim", "gt_lpips",
                "vsbase_psnr", "vsbase_ssim", "vsbase_lpips"]
    present = [k for k in tbl_keys if any(k in r for r in rows)]
    lines = ["| " + " | ".join(present) + " |",
             "|" + "---|" * len(present)]
    for r in sorted(rows, key=lambda x: (x.get("mode", ""), x.get("rank", 0))):
        cells = []
        for k in present:
            v = r.get(k, "")
            if isinstance(v, float):
                v = f"{v:.4g}"
            elif isinstance(v, int) and k == "asset_file_bytes":
                v = f"{v:,}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    os.makedirs(args.figures_dir, exist_ok=True)
    with open(os.path.join(args.figures_dir, "results_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote results_table.md ({len(rows)} rows)")
    scatter("asset_file_bytes", "gt_psnr", "rate_distortion_psnr.png",
            "compressed SH bytes", "PSNR vs ground truth (dB)", logx=True)
    scatter("asset_file_bytes", "gt_ssim", "rate_distortion_ssim.png",
            "compressed SH bytes", "SSIM vs ground truth", logx=True)
    scatter("asset_file_bytes", "gt_lpips", "rate_distortion_lpips.png",
            "compressed SH bytes", "LPIPS vs ground truth (lower better)", logx=True)
    scatter("asset_file_bytes", "vsbase_psnr", "rate_distortion_vsbase_psnr.png",
            "compressed SH bytes", "PSNR vs baseline renders (dB)", logx=True)
    scatter("rank", "gt_psnr", "psnr_vs_rank.png",
            "rank", "PSNR vs ground truth (dB)")

    # storage breakdown (stacked bars) from per-experiment component accounting
    comp_rows = []
    for r in rows:
        mpath = os.path.join(args.ablation_dir, r["name"], "metrics.json")
        apath = os.path.join(args.ablation_dir, r["name"], "asset.npz")
        if os.path.exists(apath):
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
            from sh_compression.serialization import CompressedSH
            comp = CompressedSH.load(apath).component_bytes()
            comp_rows.append((r["name"], comp))
    if comp_rows:
        fig, ax = plt.subplots(figsize=(max(6.5, 0.6 * len(comp_rows)), 4.5))
        keys = sorted({k for _, c in comp_rows for k in c})
        bottoms = [0] * len(comp_rows)
        for k in keys:
            vals = [c.get(k, 0) / 1024 for _, c in comp_rows]
            ax.bar([n for n, _ in comp_rows], vals, bottom=bottoms, label=k)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_ylabel("KiB"); ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(args.figures_dir, "storage_breakdown.png"), dpi=150)
        plt.close(fig)
        print("wrote storage_breakdown.png")


if __name__ == "__main__":
    main()
