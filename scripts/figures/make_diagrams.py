"""Thesis diagrams: Figures 3.1 (pipeline workflow), 3.2 (model architecture) and
3.3 (model process workflow).

Run from the repository root:  python scripts/figures/make_diagrams.py
Writes 300-dpi PNG + SVG into results/figures/thesis/.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "results", "figures", "thesis")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
})

INK = "#0b0b0b"
INK2 = "#52514e"
# role -> (stroke, fill); strokes from the reference categorical palette
ROLE = {
    "thesis":   ("#2a78d6", "#eaf2fc"),   # new code (blue)
    "upstream": ("#7a7974", "#f2f1ee"),   # unchanged reference 3DGS (grey)
    "artifact": ("#eb6834", "#fdefe8"),   # files on disk (orange)
    "eval":     ("#1baf7a", "#e6f6ef"),   # evaluation (aqua)
    "decision": ("#6250d6", "#efedfb"),   # decisions (violet)
    "plain":    ("#52514e", "#ffffff"),
}


def new(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, role="thesis", size=9.5, bold_first=True, lw=1.4):
    stroke, fill = ROLE[role]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fill, ec=stroke, lw=lw, zorder=2))
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        lh = size / 72 * 1.32
        top = cy + (len(lines) - 1) * lh / 2
        for i, line in enumerate(lines):
            ax.text(cx, top - i * lh, line, ha="center", va="center", fontsize=size,
                    color=INK if i == 0 else INK2,
                    fontweight="bold" if i == 0 else "normal", zorder=3)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=INK,
                fontweight="bold" if bold_first else "normal", linespacing=1.3, zorder=3)
    return (cx, cy, w, h)


def diamond(ax, cx, cy, w, h, text, size=9):
    stroke, fill = ROLE["decision"]
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, fc=fill, ec=stroke, lw=1.4, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=INK,
            linespacing=1.25, zorder=3)


def group(ax, x0, y0, x1, y1, label, role="thesis", size=10):
    stroke, _ = ROLE[role]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                fc="none", ec=stroke, lw=1.2, ls=(0, (5, 3)), zorder=1))
    ax.text(x0 + 0.12, y1 - 0.08, label, ha="left", va="top", fontsize=size,
            color=stroke, fontweight="bold", zorder=3)


def arrow(ax, p0, p1, color=INK2, lw=1.3, ls="-", label=None, lsize=8.5, loff=(0.08, 0),
          rad=0.0, head=11):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head,
                                 color=color, lw=lw, ls=ls, zorder=1.5,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        mx, my = (p0[0] + p1[0]) / 2 + loff[0], (p0[1] + p1[1]) / 2 + loff[1]
        ax.text(mx, my, label, fontsize=lsize, color=INK2, ha="left", va="center",
                style="italic", zorder=3)


def poly_arrow(ax, pts, color=INK2, lw=1.3, ls="-"):
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=1.5, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls)


def legend(ax, x, y, items, size=8.5):
    for i, (role, label) in enumerate(items):
        stroke, fill = ROLE[role]
        ax.add_patch(FancyBboxPatch((x, y - i * 0.28 - 0.09), 0.28, 0.18,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=fill, ec=stroke, lw=1.2))
        ax.text(x + 0.38, y - i * 0.28, label, fontsize=size, va="center", color=INK2)


def hlegend(ax, cx, y, items, size=8.5, step=1.75):
    x = cx - step * len(items) / 2
    for i, (role, label) in enumerate(items):
        stroke, fill = ROLE[role]
        ax.add_patch(FancyBboxPatch((x + i * step, y - 0.09), 0.28, 0.18,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=fill, ec=stroke, lw=1.2))
        ax.text(x + i * step + 0.38, y, label, fontsize=size, va="center", color=INK2)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Figure 3.2  Model architecture (tensor level)
# ---------------------------------------------------------------------------
def model_architecture():
    W, H = 8.6, 11.45
    fig, ax = new(W, H)
    ax.text(W / 2, H - 0.3, "Architecture of the compressed SH representation",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(W / 2, H - 0.6, "tensor shapes, bytes per Gaussian; example values for T&T train "
            "(N = 1,071,462)", ha="center", fontsize=9, color=INK2)

    L, R = 3.05, 7.25            # lane centres: SH lane, non-SH lane
    box(ax, W / 2, 9.8, 7.6, 0.75,
        "Trained 3DGS checkpoint  point_cloud.ply  (iteration 30,000)\n"
        "N Gaussians × 62 float32 fields = 248 B per Gaussian   ·   265.7 MB",
        role="upstream")
    box(ax, L, 8.75, 4.6, 0.72,
        "SH colour coefficients  (48 floats, 192 B)\n"
        "f_dc [N,3]  +  f_rest [N,45]   ·   205.7 MB  (77.4 %)", role="upstream")
    box(ax, R, 8.75, 2.3, 0.72,
        "Non-SH fields (14 floats, 56 B)\n"
        "xyz, normals, opacity,\nscale ×3, rotation ×4", role="upstream", size=8.6)
    arrow(ax, (L, 9.42), (L, 9.11))
    arrow(ax, (R, 9.42), (R, 9.11))

    box(ax, L, 7.85, 4.6, 0.6,
        "Canonical layout  (sh_layout.py)\n"
        "S [N,3,16]  →  X [N,48],   X[i, 16c+j] = S[i,c,j]", role="thesis")
    arrow(ax, (L, 8.39), (L, 8.15))

    # encoder
    group(ax, 0.35, 3.95, 6.05, 7.42, "ENCODER  (offline, CPU)")
    box(ax, L, 6.85, 4.6, 0.6,
        "Truncated SVD, float64  (factorization.py)\n"
        r"$X \approx U\,V$,   $U = U_s[:, :r]\,\Sigma_r$,   $V = V_s^{\top}[:r, :]$",
        role="thesis")
    arrow(ax, (L, 7.55), (L, 7.15))
    box(ax, 1.45, 5.85, 1.75, 0.82,
        "Shared basis V\n[r, 48]  fp32 / fp16\n192r B in total\n(independent of N)",
        role="thesis", size=8.6)
    box(ax, 4.0, 5.85, 2.95, 0.82,
        "Per-Gaussian weights U\n[N, r],  singular values folded in\n"
        "quantised by the selected mode", role="thesis", size=8.6)
    arrow(ax, (2.2, 6.55), (1.6, 6.26))
    arrow(ax, (3.6, 6.55), (4.0, 6.26))
    # three quantiser modes
    mx = [2.55, 3.95, 5.3]
    box(ax, mx[0], 4.6, 1.3, 1.05,
        "lowrank\nU as fp32 / fp16\n4r or 2r B\nper Gaussian", role="thesis", size=8.2)
    box(ax, mx[1], 4.6, 1.3, 1.05,
        "scalar (b bit)\ncodes [N,r] u8/u16\n+ scale, offset [r]\nr·b/8 B / Gaussian",
        role="thesis", size=8.2)
    box(ax, mx[2], 4.6, 1.3, 1.05,
        "K-means VQ\nindex [N] u8/u16\n+ codebook [K,r]\n1 B / Gaussian\n(K ≤ 256)",
        role="thesis", size=8.2)
    for x in mx:
        arrow(ax, (4.0, 5.44), (x, 5.13))

    # container
    box(ax, L, 3.25, 4.6, 0.75,
        "Compressed asset  asset.npz  (serialization.py)\n"
        "V + quantised U + meta_json,  zlib (savez_compressed)\n"
        "rate = measured file size,  e.g. 13.7 MB for scalar r = 16, b = 8",
        role="artifact", size=8.8)
    for x in mx:
        poly_arrow(ax, [(x, 4.07), (x, 3.85), (min(max(x, 1.4), 4.9), 3.63)])
    poly_arrow(ax, [(1.45, 5.44), (1.45, 3.63)])

    # decoder
    group(ax, 0.35, 1.35, 6.05, 2.68, "DECODER  (extraction.py)", role="eval")
    box(ax, L, 2.02, 4.6, 0.62,
        "Dequantise and reconstruct\n"
        r"$\hat U$ from codes / indices,   $\hat X = \hat U\,V$  [N,48]  →  $\hat S$ [N,3,16]",
        role="eval")
    arrow(ax, (L, 2.87), (L, 2.34))

    # merge into PLY and render
    box(ax, W / 2, 0.45, 7.6, 0.62,
        "Temporary PLY (62 fields)  →  unmodified CUDA rasteriser  →  rendered views\n"
        "SH evaluated per view inside the kernel, exactly as for the baseline",
        role="upstream", size=9)
    poly_arrow(ax, [(L, 1.71), (L, 0.77)])
    poly_arrow(ax, [(R, 8.39), (R, 0.77)], ls=(0, (4, 3)))
    ax.text(R + 0.12, 5.2, "preserved\nlosslessly\n(bypasses the\ncompressor)", fontsize=8.5,
            color=INK2, style="italic", va="center")

    # bytes summary panel
    box(ax, R, 3.0, 2.3, 1.7,
        "Measured, scalar r = 16\nSH:  205.7 → 13.7 MB\n= 15.0× smaller\n\n"
        "Whole model:\n265.7 → 73.7 MB = 3.6×\n(bound: 248 / 56 = 4.43×)",
        role="plain", size=8.6)

    hlegend(ax, W / 2, H - 0.95, [("upstream", "unchanged 3DGS"), ("thesis", "thesis code"),
                                  ("artifact", "file on disk"), ("eval", "decoder")])
    save(fig, "fig3_2_model_architecture")


# ---------------------------------------------------------------------------
# Figure 3.1  End-to-end experimental pipeline (swim lanes)
# ---------------------------------------------------------------------------
def pipeline_workflow():
    W, H = 13.0, 7.6
    fig, ax = new(W, H)
    ax.text(0.3, H - 0.32, "End-to-end experimental pipeline", fontsize=13,
            fontweight="bold", color=INK)
    ax.text(0.3, H - 0.62, "step numbers match Table 3.7; scripts in scripts/, "
            "algorithms in src/sh_compression/", fontsize=9, color=INK2)

    cx = [1.85, 3.9, 5.95, 8.05, 10.15, 12.0]     # column centres
    bw = 1.85
    phases = [(0.85, 2.8, "1  TRAINING (unchanged)"),
              (2.9, 9.1, "2  BASELINE RENDER + OFFLINE COMPRESSION"),
              (9.2, 12.95, "3  EVALUATION & REPORTING")]
    for x0, x1, label in phases:
        ax.add_patch(FancyBboxPatch((x0, 6.38), x1 - x0, 0.4,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="#f2f1ee", ec="none"))
        ax.text((x0 + x1) / 2, 6.58, label, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=INK)

    G, C, A = 5.28, 3.3, 1.3
    lanes = [(4.4, 6.18, "GPU\n(Colab\nT4)"), (2.32, 4.28, "CPU\n(any\nmachine)"),
             (0.62, 2.0, "Files\nwritten")]
    for y0, y1, label in lanes:
        ax.add_patch(FancyBboxPatch((0.85, y0), 12.1, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="none", ec="#d9d8d3", lw=1))
        ax.text(0.45, (y0 + y1) / 2, label, ha="center", va="center",
                fontsize=8.5, color=INK2, fontweight="bold", linespacing=1.2)

    box(ax, cx[0], G, bw, 1.25,
        "Trained checkpoint\ntrain.py, 30k iterations\n(here: GaussianViewTest\n"
        "+ tandt_db images)", role="upstream", size=8.4)
    box(ax, cx[1], G, bw, 1.25,
        "1–5  Baseline render\nfix_cfg_args.py\nrender.py --skip_train\n"
        "38 test views, round trip", role="upstream", size=8.4)
    box(ax, cx[2], C, bw, 1.35,
        "4, 7  Inspect + rank\ninspect_model.py\nanalyse_rank.py\n"
        "N, SH share, E(r),\neffective rank", role="thesis", size=8.4)
    box(ax, cx[3], C, bw, 1.35,
        "8  Compression grid\nrun_ablation.py\n22 configurations\n"
        "SVD + quantise\n+ coefficient error", role="thesis", size=8.4)
    box(ax, cx[4], G, bw, 1.25,
        "6, 9  Render & score\nrun_render_eval.py\n12 configurations:\n"
        "decode → render → metrics", role="eval", size=8.4)
    box(ax, cx[5], C + 0.42, 1.6, 0.7,
        "10  Figures, tables\ngenerate_report_assets", role="eval", size=8.1)
    box(ax, cx[5], C - 0.45, 1.6, 0.7,
        "11  Peer baselines\ncompare_sh_baselines", role="eval", size=8.1)

    arts = ["point_cloud.ply\n265.7 MB", "GT + baseline\nrenders (2 × 38 PNG)",
            "rank_analysis/*.csv\nsingular values, E(r)",
            "ablation/<name>/asset.npz\n+ metrics.json (× 22)",
            "image_metrics/\nmetrics_<name>.json (× 12)",
            "figures/*.png,\nsh_baselines.csv"]
    for x, t in zip(cx, arts):
        box(ax, x, A, bw if x != cx[5] else 1.6, 0.72, t, role="artifact", size=7.9,
            bold_first=False)

    # control flow
    arrow(ax, (cx[0] + bw / 2, G), (cx[1] - bw / 2, G))
    poly_arrow(ax, [(cx[1] + bw / 2, G), (cx[2], G), (cx[2], C + 0.68)])
    ax.text(cx[2] + 0.08, G + 0.12, "PLY", fontsize=8.2, color=INK2, style="italic")
    arrow(ax, (cx[2] + bw / 2, C), (cx[3] - bw / 2, C))
    poly_arrow(ax, [(cx[3], C + 0.68), (cx[3], G), (cx[4] - bw / 2, G)])
    ax.text(cx[3] + 0.08, G + 0.12, "12 selected", fontsize=8.2, color=INK2, style="italic")
    poly_arrow(ax, [(cx[4], G - 0.63), (cx[4], C + 0.42), (cx[5] - 0.8, C + 0.42)])
    poly_arrow(ax, [(cx[3] + bw / 2, C - 0.45), (cx[5] - 0.8, C - 0.45)])
    # files written (dotted)
    dots = dict(ls=(0, (2, 2)), lw=1.0)
    poly_arrow(ax, [(cx[0], G - 0.63), (cx[0], A + 0.36)], **dots)
    poly_arrow(ax, [(cx[1], G - 0.63), (cx[1], A + 0.36)], **dots)
    for x in (cx[2], cx[3]):
        arrow(ax, (x, C - 0.68), (x, A + 0.36), **dots)
    poly_arrow(ax, [(cx[4] - 0.35, G - 0.63), (cx[4] - 0.35, A + 0.36)], **dots)
    arrow(ax, (cx[5], C - 0.8), (cx[5], A + 0.36), **dots)
    ax.text(0.85, 0.3, "solid: control flow;  dotted: file written.  The rate is always the "
            "measured .npz size; reconstructed PLYs are temporary and never reported as size.",
            fontsize=8.5, color=INK2, ha="left")
    save(fig, "fig3_1_pipeline_workflow")


# ---------------------------------------------------------------------------
# Figure 3.3  Model process workflow (one model, with decisions)
# ---------------------------------------------------------------------------
def process_workflow():
    W, H = 9.0, 11.6
    fig, ax = new(W, H)
    ax.text(W / 2, H - 0.3, "Model process workflow: compressing one trained scene",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(W / 2, H - 0.6, "decisions in violet; the loop selects the operating point",
            ha="center", fontsize=9, color=INK2)
    X, bw = 3.75, 4.2
    box(ax, X, 10.35, bw, 0.6, "Input: trained checkpoint\npoint_cloud.ply (N Gaussians, 62 fields)",
        role="upstream")
    diamond(ax, X, 9.25, 3.4, 1.05, "Valid 3DGS model?\n45 f_rest fields (degree 3),\nall values finite")
    box(ax, 7.55, 9.25, 2.4, 0.6, "Reject with an error\n(ply_io / sh_layout)", role="plain", size=8.6)
    box(ax, X, 8.1, bw, 0.6, "Extract SH and preserve the rest\nX [N,48]; 14 non-SH fields kept",
        role="thesis")
    box(ax, X, 7.15, bw, 0.6, "Truncated SVD of X  (float64)\n"
        r"singular values $\sigma_1 \geq \dots \geq \sigma_{48}$", role="thesis")
    box(ax, X, 6.2, bw, 0.6, "Choose rank r\nfrom E(r) and the grid (effective rank = 12)",
        role="decision")
    diamond(ax, X, 5.1, 2.6, 0.95, "Quantisation\nmode?")
    mx = [X - 2.3, X, X + 2.3]
    box(ax, mx[0], 3.85, 1.95, 0.8, "lowrank\ncast U, V to\nfp32 or fp16", role="thesis", size=8.6)
    box(ax, mx[1], 3.85, 1.95, 0.8, "scalar\nper-column min/max\n→ 8/16-bit codes", role="thesis",
        size=8.6)
    box(ax, mx[2], 3.85, 1.95, 0.8, "vq\nK-means++ (seed),\n50 Lloyd iterations", role="thesis",
        size=8.6)
    box(ax, X, 2.8, bw, 0.6, "Save asset.npz and measure bytes  →  RATE\nSH ratio = 192N / bytes",
        role="artifact")
    box(ax, X, 1.85, bw, 0.6, "Decode, render 38 views, score  →  DISTORTION\n"
        "PSNR / SSIM / LPIPS vs GT and vs baseline", role="eval")
    diamond(ax, X, 0.72, 3.9, 1.0, "Within tolerance?\nΔPSNR(GT) ≤ 1 dB and SSIM(base) ≥ 0.99")
    box(ax, 7.55, 2.3, 2.5, 1.15,
        "Operating point\nsmallest passing, Pareto-\noptimal configuration\n"
        "scalar r = 16, b = 8:\n13.7 MB, −0.62 dB, SSIM 0.992",
        role="eval", size=8.3)

    arrow(ax, (X, 10.05), (X, 9.78))
    arrow(ax, (X + 1.7, 9.25), (6.35, 9.25), label="no", loff=(-0.25, 0.14))
    arrow(ax, (X, 8.72), (X, 8.4), label="yes", loff=(0.08, 0.02))
    arrow(ax, (X, 7.8), (X, 7.45))
    arrow(ax, (X, 6.85), (X, 6.5))
    arrow(ax, (X, 5.9), (X, 5.58))
    poly_arrow(ax, [(X - 1.3, 5.1), (mx[0], 5.1), (mx[0], 4.25)])
    poly_arrow(ax, [(X + 1.3, 5.1), (mx[2], 5.1), (mx[2], 4.25)])
    arrow(ax, (X, 4.62), (X, 4.25))
    for x in mx:
        poly_arrow(ax, [(x, 3.45), (x, 3.25), (X + (x - X) * 0.35, 3.1)])
    arrow(ax, (X, 2.5), (X, 2.15))
    arrow(ax, (X, 1.55), (X, 1.22))
    poly_arrow(ax, [(X + 1.95, 0.72), (7.55, 0.72), (7.55, 1.72)])
    ax.text(6.0, 0.86, "yes", fontsize=8.5, color=INK2, style="italic")
    poly_arrow(ax, [(X - 1.95, 0.72), (0.3, 0.72), (0.3, 6.2), (X - bw / 2, 6.2)])
    ax.text(0.42, 0.95, "no: increase r, bits or K", fontsize=8.5,
            color=INK2, style="italic", rotation=90, va="center", ha="left")
    save(fig, "fig3_3_process_workflow")


if __name__ == "__main__":
    model_architecture()
    pipeline_workflow()
    process_workflow()
