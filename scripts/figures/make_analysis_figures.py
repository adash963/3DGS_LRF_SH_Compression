"""Thesis analysis figures: 4.1 (singular spectrum and explained energy), 4.2 (predicted
SH-payload ratio against rank), 4.3 (predicted whole-model ratio and its 4.43x bound) and
5.1 (rate-distortion of the twelve rendered configurations).

Run from the repository root:  python scripts/figures/make_analysis_figures.py
Writes 300-dpi PNG + SVG into results/figures/thesis/.

Inputs: results/experiments/rank_analysis/singular_values.csv (Figure 4.1) and
results/experiments/image_metrics/metrics_<name>.json (Figure 5.1). Figures 4.2 and 4.3
are analytical (thesis Section 3.3.2, Equations 3.10-3.11, Table 4.6) and need no data.
"""
import csv
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sh_compression.factorization import effective_rank
from sh_compression.sh_layout import FLAT_DIM

RES = os.path.join(REPO, "results", "experiments")
OUT = os.path.join(REPO, "results", "figures", "thesis")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5, "axes.edgecolor": "#b9b8b2", "axes.linewidth": 0.8,
    "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.7, "svg.fonttype": "none",
})
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"

# Storage model of the reference PLY (thesis Table 1.1): 62 float32 fields per Gaussian.
BYTES_PER_GAUSSIAN = 62 * 4                     # 248
SH_BYTES = FLAT_DIM * 4                         # 192 (f_dc + f_rest)
NON_SH_BYTES = BYTES_PER_GAUSSIAN - SH_BYTES    # 56 (incl. the three zero-filled normals)
OPERATING_RANK = 16                             # smallest rank within the 1 dB tolerance (5.3)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, facecolor="white",
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", name)


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=11, fontweight="bold", color=INK)


# ---------------------------------------------------------------------------
# Figure 4.1  Singular value spectrum and cumulative explained energy
# ---------------------------------------------------------------------------
def singular_spectrum():
    with open(os.path.join(RES, "rank_analysis", "singular_values.csv")) as f:
        s = np.array([float(r["singular_value"]) for r in csv.DictReader(f)])
    ranks = np.arange(1, len(s) + 1)
    energy = 100 * np.cumsum(s**2) / np.sum(s**2)
    eff = effective_rank(s)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    a.plot(ranks, s, color=BLUE, lw=2, marker="o", ms=4, zorder=3)
    a.set_yscale("log")
    a.axvline(eff, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    a.annotate(f"effective rank {eff}", (eff, s[eff - 1]), xytext=(18, 1050),
               textcoords="data", fontsize=9, color=INK2,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    a.set_xlabel("Component index")
    a.set_ylabel("Singular value (log scale)")
    a.grid(True)
    panel_title(a, "(a) Singular values of X")

    b.plot(ranks, energy, color=BLUE, lw=2, marker="o", ms=4, zorder=2)
    b.axhline(99, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    for r, xytext in ((1, (4, 90.1)), (eff, (14.5, 96.2)), (OPERATING_RANK, (23, 97.6))):
        b.scatter([r], [energy[r - 1]], s=60, color=ORANGE, zorder=3)
        b.annotate(f"r = {r}: {energy[r - 1]:.1f} %", (r, energy[r - 1]), xytext=xytext,
                   textcoords="data", fontsize=9, color=INK2, va="center",
                   arrowprops=None if r == 1 else dict(arrowstyle="-", color=MUTED, lw=1))
    b.set_ylim(88, 100.6)
    b.set_xlabel("Number of components r")
    b.set_ylabel("Cumulative explained energy (%)")
    b.grid(True)
    panel_title(b, "(b) Explained energy E(r)")
    fig.tight_layout(w_pad=3)
    save(fig, "fig4_1_singular_spectrum")


# ---------------------------------------------------------------------------
# Figure 4.2  Predicted SH-payload compression ratio against rank (N >> 48)
# ---------------------------------------------------------------------------
def sh_ratio_vs_rank():
    r = np.arange(1, FLAT_DIM + 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(r, SH_BYTES / (4 * r), color=BLUE, lw=2, label="Low-rank, float32 factors")
    ax.plot(r, SH_BYTES / (2 * r), color=ORANGE, lw=2, ls=(0, (5, 3)),
            label="Low-rank, float16 factors  /  scalar 16-bit U")
    ax.plot(r, SH_BYTES / r, color=AQUA, lw=2, label="Scalar 8-bit U (float32 V)")
    ax.axhline(SH_BYTES / 1, color=INK2, lw=1.8, ls=":", label="K-means VQ, K ≤ 256 (1 B index)")
    ax.axhline(SH_BYTES / 2, color=VIOLET, lw=1.8, ls=":",
               label="K-means VQ, 256 < K ≤ 65,536 (2 B index)")
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlim(1, FLAT_DIM)
    ax.set_ylim(0.8, 320)
    ax.set_xlabel("SVD rank r")
    ax.set_ylabel("SH payload compression ratio (×)")
    ax.grid(True, which="both")
    # sits in the empty band between the scalar curve and the VQ lines
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.78), fontsize=8, frameon=False)
    panel_title(ax, "Predicted SH-payload compression against rank")
    ax.text(0.0, -0.16, "Analytical, N ≫ 48: fixed-size arrays (V, codebook, scale, offset) "
            "and zlib are ignored (Equation 3.10).",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
    save(fig, "fig4_2_sh_ratio_vs_rank")


# ---------------------------------------------------------------------------
# Figure 4.3  Predicted whole-model ratio and its upper bound (56 B non-SH)
# ---------------------------------------------------------------------------
WHOLE_MODEL_CONFIGS = [          # (label, SH bytes per Gaussian), Table 4.6 order
    ("Uncompressed\nPLY", SH_BYTES),
    ("Low-rank\nfp32, r = 16", 4 * 16), ("Low-rank\nfp32, r = 8", 4 * 8),
    ("Low-rank\nfp16, r = 8", 2 * 8),
    ("Scalar 8-bit\nr = 8", 8), ("Scalar 8-bit\nr = 4", 4),
    ("VQ\nK = 4096", 2), ("VQ\nK = 256", 1),
]


def whole_model_ratio():
    labels = [c[0] for c in WHOLE_MODEL_CONFIGS]
    ratios = [BYTES_PER_GAUSSIAN / (NON_SH_BYTES + b) for _, b in WHOLE_MODEL_CONFIGS]
    bound = BYTES_PER_GAUSSIAN / NON_SH_BYTES
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, ratios, width=0.7, color=BLUE, zorder=2)
    for xi, v in zip(x, ratios):
        ax.text(xi, v - 0.1, f"{v:.2f}×", ha="center", va="top", fontsize=8.5,
                color="white", fontweight="bold", zorder=3)
    ax.axhline(bound, color=ORANGE, lw=1.4, ls=(0, (5, 3)), zorder=3)
    ax.text(-0.35, bound + 0.12, f"upper bound {BYTES_PER_GAUSSIAN} / {NON_SH_BYTES} = "
            f"{bound:.2f}× (SH removed entirely)", fontsize=8.5, color=ORANGE, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("Whole-model compression (×)")
    ax.grid(True, axis="y")
    panel_title(ax, "Predicted whole-model compression (56 B of non-SH data per Gaussian)")
    save(fig, "fig4_3_whole_model_ratio")


# ---------------------------------------------------------------------------
# Figure 5.1  Rate-distortion of the rendered configurations
# ---------------------------------------------------------------------------
def rate_distortion():
    runs = {}
    for path in glob.glob(os.path.join(RES, "image_metrics", "metrics_*.json")):
        with open(path) as f:
            m = json.load(f)
        cfg = m["config"]
        runs[cfg["name"]] = dict(mb=m["asset_file_bytes"] / 1e6, mode=cfg.get("mode", "lowrank"),
                                 fp16=cfg.get("factor_dtype") == "float16", **m["summary"])
    # rank 48 float32 is the control: it reproduces the uncompressed model (Table 5.1)
    base = runs["lr_r48_fp32"]

    def family(pred):
        return sorted((r for r in runs.values() if pred(r)), key=lambda r: r["mb"])

    lowrank = family(lambda r: r["mode"] == "lowrank" and not r["fp16"])
    fp16 = family(lambda r: r["mode"] == "lowrank" and r["fp16"])
    scalar = family(lambda r: r["mode"] == "scalar")
    vq = family(lambda r: r["mode"] == "vq")

    panels = [("gt_psnr", "PSNR vs ground truth (dB)", "higher is better"),
              ("gt_ssim", "SSIM vs ground truth", "higher is better"),
              ("gt_lpips", "LPIPS vs ground truth", "lower is better")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.0))
    for ax, (key, ylabel, note) in zip(axes, panels):
        for rows, color, marker in ((lowrank, BLUE, "o"), (scalar, ORANGE, "s"), (vq, AQUA, "D")):
            ax.plot([r["mb"] for r in rows], [r[key] for r in rows], color=color, lw=1.6,
                    alpha=0.55, zorder=2)
            ax.scatter([r["mb"] for r in rows], [r[key] for r in rows], s=46, marker=marker,
                       color=color, zorder=3)
        ax.scatter([r["mb"] for r in fp16], [r[key] for r in fp16], s=46, marker="o",
                   facecolor="white", edgecolor=BLUE, linewidth=1.6, zorder=3)
        ax.axhline(base[key], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 320)
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1", "10", "100"])
        ax.set_xlabel("Compressed SH size (MB)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.set_title(note, loc="left", fontsize=9, color=INK2)
    op = runs[f"sq_r{OPERATING_RANK}_b8"]
    axes[0].annotate(f"rank {OPERATING_RANK}, 8-bit", (op["mb"], op["gt_psnr"]),
                     textcoords="offset points", xytext=(-9, 0), ha="right", va="center",
                     fontsize=8, color=INK2)
    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", lw=1.6, label="Low-rank (filled float32, "
                   "hollow float16)"),
        plt.Line2D([], [], color=ORANGE, marker="s", lw=1.6, label="Scalar 8-bit"),
        plt.Line2D([], [], color=AQUA, marker="D", lw=1.6, label="K-means VQ"),
        plt.Line2D([], [], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="Uncompressed model"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.5)
    save(fig, "fig5_1_rate_distortion")


if __name__ == "__main__":
    singular_spectrum()
    sh_ratio_vs_rank()
    whole_model_ratio()
    rate_distortion()
