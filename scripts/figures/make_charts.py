"""Thesis comparison charts: Figures 5.2 (same-checkpoint comparison with peer-style
SH treatments) and 5.3 (position among published methods on Tanks and Temples).

Run from the repository root:  python scripts/figures/make_charts.py
Writes 300-dpi PNG + SVG into results/figures/thesis/.

Inputs: results/experiments/image_metrics/ and results/experiments/sh_baselines/
(written by scripts/compare_sh_baselines.py), plus the published Tanks & Temples
values below (author-reported, as collected by the 3DGS compression survey of
Bagdasarian et al. (2025), accessed 29 Oct 2026; thesis Table 5.10).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# Tanks & Temples (train + truck average), author-reported
TT_3DGS = dict(psnr=23.14, ssim=0.841, lpips=0.183, mb=411.0)
TT = [
    ("HAC (low)", 24.04, 0.846, 0.187, 8.5), ("HAC (high)", 24.40, 0.853, 0.177, 11.8),
    ("HAC++ (low)", 24.22, 0.849, 0.190, 5.4), ("HAC++ (high)", 24.33, 0.853, 0.181, 7.3),
    ("LightGaussian", 23.11, 0.817, 0.231, 22.0),
    ("Compact 3DGS", 23.32, 0.831, 0.201, 39.4), ("Compact 3DGS + PP", 23.32, 0.831, 0.202, 20.9),
    ("Compressed 3DGS", 23.32, 0.832, 0.194, 17.3), ("EAGLES", 23.37, 0.84, 0.20, 29.0),
    ("Reduced 3DGS", 23.57, 0.840, 0.188, 14.0),
    ("CompGS 16K (Navaneet)", 23.39, 0.836, 0.200, 12.0),
    ("CompGS 32K (Navaneet)", 23.44, 0.838, 0.198, 13.0),
    ("CompGS (Liu)", 23.70, 0.837, 0.208, 10.1), ("Scaffold-GS", 23.96, 0.853, 0.177, 87.0),
    ("ContextGS (low)", 24.12, 0.849, 0.186, 9.9), ("ContextGS (high)", 24.29, 0.855, 0.176, 12.4),
]
BASE_GT_PSNR = 24.9298      # thesis baseline vs GT (lr_r48 control = baseline)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, facecolor="white",
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", name)


def same_checkpoint():
    d = json.load(open(os.path.join(RES, "sh_baselines", "sh_baselines.json")))
    rows = {r["name"]: r for r in d["rows"]}
    mb = lambda n: rows[n]["file_bytes"] / 1e6
    ps = lambda n: rows[n]["colour_proxy_psnr"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    sq = ["sq_r4_b8", "sq_r8_b8", "sq_r12_b8", "sq_r16_b8", "sq_r24_b8", "sq_r32_b8", "sq_r48_b8"]
    ax.plot([mb(n) for n in sq], [ps(n) for n in sq], color=BLUE, lw=2, zorder=2)
    for n in sq:
        rendered = rows[n]["rendered_vsbase_psnr"] is not None
        ax.scatter([mb(n)], [ps(n)], s=46, color=BLUE, facecolor=BLUE if rendered else "white",
                   linewidth=1.6, zorder=3)
    ax.plot([], [], color=BLUE, lw=2, marker="o", ms=6, label="Thesis: SVD + 8-bit scalar, r = 4 … 48")
    vq = ["vq_r8_k64", "vq_r8_k256"]
    ax.scatter([mb(n) for n in vq], [ps(n) for n in vq], s=60, marker="D", color=BLUE, zorder=4,
               label="Thesis: SVD (r = 8) + K-means VQ, K = 64 / 256")
    base = ["q8_raw", "deg2", "deg1", "deg1_q8", "deg0", "deg0_q8", "rawvq_k64", "rawvq_k256"]
    ax.scatter([mb(n) for n in base], [ps(n) for n in base], s=46, marker="s", color=ORANGE,
               zorder=3, label="Peer-style SH treatments (no SVD), same checkpoint")
    lab = [
        ("sq_r4_b8", "r4", (-4, 8)), ("sq_r8_b8", "r8", (-16, 6)), ("sq_r12_b8", "r12", (-22, 4)),
        ("sq_r16_b8", "r16 (13.7 MB)", (7, -9)), ("sq_r24_b8", "r24", (7, -8)),
        ("sq_r32_b8", "r32", (7, -8)), ("sq_r48_b8", "r48", (7, -10)),
        ("q8_raw", "raw SH, 8-bit", (-70, 4)), ("deg2", "SH degree ≤ 2", (-30, 8)),
        ("deg1", "degree ≤ 1", (7, -4)), ("deg1_q8", "degree ≤ 1, 8-bit", (7, -4)),
        ("deg0", "DC only (no view dependence)", (7, -4)), ("deg0_q8", "DC only, 8-bit", (6, -12)),
        ("rawvq_k64", "VQ K64: thesis ≈ raw-SH", (-8, -14)),
        ("rawvq_k256", "VQ K256: thesis ≈ raw-SH", (-16, 9)),
    ]
    for n, text, off in lab:
        ax.annotate(text, (mb(n), ps(n)), textcoords="offset points", xytext=off, fontsize=7.9,
                    color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 200)
    ax.set_ylim(19, 52)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
    ax.set_xlabel("Compressed SH size on disk (MB, log scale)   ·   uncompressed SH = 205.7 MB")
    ax.set_ylabel("SH-colour PSNR vs uncompressed (dB)")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    v = d["proxy_validation"]
    ax.text(0.0, -0.17, "Filled markers: also rendered on GPU; hollow: proxy only. "
            f"proxy vs rendered PSNR: Pearson {v['pearson']:.3f} ({v['n_configs']} configs). "
            "Off scale: fp16 SH (91 MB, 86 dB).",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=MUTED)
    ax.set_title("Same checkpoint (T&T train): thesis vs peer-style SH compression",
                 loc="left", fontsize=11, fontweight="bold", color=INK)
    save(fig, "fig5_2_rd_same_checkpoint")


def published():
    abl = {}
    for f in os.listdir(os.path.join(RES, "image_metrics")):
        m = json.load(open(os.path.join(RES, "image_metrics", f)))
        abl[m["config"]["name"]] = m
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    xs = [TT_3DGS["mb"] / mb for _, _, _, _, mb in TT]
    ys = [p - TT_3DGS["psnr"] for _, p, _, _, _ in TT]
    ax.scatter(xs, ys, s=40, marker="s", color=ORANGE, label="Published methods (T&T average, "
               "author-reported)", zorder=3)
    for (name, p, *_rest, mb), x, y in zip(TT, xs, ys):
        off = {"HAC (high)": (5, 4), "HAC++ (low)": (-30, 8), "LightGaussian": (-20, -13),
               "Compressed 3DGS": (-62, 9), "Scaffold-GS": (6, 3), "Compact 3DGS": (-22, -13)}
        if name in off:
            ax.annotate(name, (x, y), textcoords="offset points", xytext=off[name], fontsize=7.8,
                        color=INK2)
    ours = [("sq_r16_b8", "scalar r16 8-bit"), ("lr_r8_fp16", "low-rank r8 fp16"),
            ("vq_r8_k256", "VQ r8 K256"), ("lr_r16_fp32", "low-rank r16 fp32")]
    for name, text in ours:
        m = abl[name]
        x, y = m["whole_model_ratio"], m["summary"]["gt_psnr"] - BASE_GT_PSNR
        ax.scatter([x], [y], s=52, color=BLUE, zorder=4,
                   label="This thesis (T&T train, same-checkpoint Δ)" if name == "sq_r16_b8" else None)
        ax.annotate(text, (x, y), textcoords="offset points",
                    xytext=(-12, 9) if name == "lr_r16_fp32" else (7, -3), fontsize=7.8, color=INK2)
    ax.axvline(4.43, color=BLUE, lw=1, ls=(0, (4, 3)))
    ax.text(4.6, -1.15, "SH-only ceiling\n248 / 56 = 4.43×", fontsize=7.8, color=BLUE, va="top")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["2×", "3×", "5×", "10×", "20×", "50×", "100×"])
    ax.set_xlim(1.8, 100)
    ax.set_ylim(-4.2, 1.6)
    ax.set_xlabel("Whole-model compression vs uncompressed 3DGS (log scale)")
    ax.set_ylabel("ΔPSNR vs own uncompressed baseline (dB)")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Position among published 3DGS compression methods",
                 loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.text(0.99, 0.36, "unlabelled: EAGLES, Reduced 3DGS, CompGS ×3,\nContextGS ×2, HAC (low), "
            "HAC++ (high)", transform=ax.transAxes, fontsize=7.4, color=MUTED, ha="right")
    ax.text(0.0, -0.2, "Published Δ: held-out test views, each paper's own 3DGS baseline (23.14 dB). "
            "Thesis Δ: same checkpoint, views seen in training.\nPublished methods retrain, prune "
            "and compress geometry; the thesis compresses SH only, post-training.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
    save(fig, "fig5_3_published_comparison")


if __name__ == "__main__":
    published()
    if os.path.exists(os.path.join(RES, "sh_baselines", "sh_baselines.json")):
        same_checkpoint()
