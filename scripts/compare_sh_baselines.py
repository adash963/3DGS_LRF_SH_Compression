"""Same-checkpoint comparison of the thesis SH compressors against simpler SH
treatments used in published 3DGS compression work.

Every method is applied to the SAME baseline PLY and stored in the same zlib .npz
container, so bytes are directly comparable. Quality is measured in two ways that
need no GPU:

  * coefficient error (relative Frobenius; DC and higher-order bands separately);
  * a view-dependent colour proxy: each Gaussian's RGB is evaluated from its SH for
    every test camera (same polynomial and +0.5 / clamp as the CUDA rasteriser),
    and the opacity-weighted PSNR between original and compressed colours is
    reported. This is NOT a rendered-image PSNR; it is validated against the
    committed rendered metrics (results/experiments/image_metrics) and printed.

Baselines (no fine-tuning, so they are lower bounds on the published methods):
  fp16_raw       half-precision SH                 (EAGLES-style storage)
  q8_raw         per-coefficient 8-bit uniform     (8-bit post-processing)
  deg{2,1,0}     drop SH bands above degree L      (LightGaussian / Reduced-3DGS style)
  deg{1,0}_q8    band drop + 8-bit
  rawvq_k{K}     K-means VQ of the raw 48-D vectors, same kmeans_vq as the thesis
                 (CompGS / Compressed-3DGS style, without sensitivity weighting)
"""
import argparse
import csv
import glob
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from sh_compression.extraction import compress, extract_flat_sh
from sh_compression.quantization import kmeans_vq, scalar_quantize
from sh_compression.serialization import CompressedSH
from sh_compression.storage import ratios

# SH constants from utils/sh_utils.py (identical to the CUDA rasteriser)
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435]

DC_COLS = [0, 16, 32]                      # coefficient 0 of each channel
REST_COLS = [c for c in range(48) if c not in DC_COLS]


def sh_basis(d: np.ndarray) -> np.ndarray:
    """[n, 3] unit directions -> [n, 16] degree-3 basis (rasteriser sign convention)."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    return np.stack([
        np.full_like(x, C0),
        -C1 * y, C1 * z, -C1 * x,
        C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy),
        C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy),
        C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
        C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy),
    ], axis=1).astype(np.float32)


def test_views(cameras_json: str, llffhold: int = 8) -> list[dict]:
    """Every llffhold-th camera of the name-sorted list (scene/dataset_readers.py)."""
    with open(cameras_json) as f:
        cams = sorted(json.load(f), key=lambda c: c["img_name"])
    return [c for i, c in enumerate(cams) if i % llffhold == 0]


class ColourProxy:
    """Per-view visible Gaussians, SH basis and opacity weights, computed once."""

    def __init__(self, xyz, opacity_logit, views):
        w_all = 1.0 / (1.0 + np.exp(-opacity_logit))
        self.views = []
        for cam in views:
            pos = np.asarray(cam["position"], dtype=np.float64)
            rot = np.asarray(cam["rotation"], dtype=np.float64)   # camera-to-world
            p = (xyz - pos) @ rot                                 # camera frame
            z = p[:, 2]
            vis = (z > 0.2) & \
                (np.abs(p[:, 0] / np.maximum(z, 1e-6)) < cam["width"] / (2 * cam["fx"])) & \
                (np.abs(p[:, 1] / np.maximum(z, 1e-6)) < cam["height"] / (2 * cam["fy"]))
            idx = np.flatnonzero(vis)
            d = xyz[idx] - pos
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            self.views.append((idx, sh_basis(d), w_all[idx].astype(np.float64)))

    @staticmethod
    def _rgb(basis, sh_rows):
        # sh_rows [n, 48] channel-major -> [n, 3, 16]; colour = basis . coeffs + 0.5
        c = np.einsum("nk,nck->nc", basis, sh_rows.reshape(-1, 3, 16)) + 0.5
        return np.clip(c, 0.0, 1.0)

    def psnr(self, x: np.ndarray, x_hat: np.ndarray) -> float:
        num, den = 0.0, 0.0
        for idx, basis, w in self.views:
            err = self._rgb(basis, x[idx]) - self._rgb(basis, x_hat[idx])
            num += float(np.sum(w[:, None] * err.astype(np.float64) ** 2))
            den += float(np.sum(w)) * 3
        mse = num / den
        return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def npz_bytes(**arrays) -> int:
    """Size of the arrays in the same zlib .npz container the thesis uses."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.npz")
        np.savez_compressed(path, **arrays)
        return os.path.getsize(path)


def band_mask(max_degree: int) -> np.ndarray:
    keep = (max_degree + 1) ** 2
    return np.array([(c % 16) < keep for c in range(48)])


def baseline_methods(x: np.ndarray, vq_sizes: list[int], seed: int):
    """Yield (name, family, description, x_hat, bytes)."""
    h = x.astype(np.float16)
    yield "fp16_raw", "baseline", "SH stored as float16", h.astype(np.float32), npz_bytes(sh=h)

    q = scalar_quantize(x, 8)
    yield ("q8_raw", "baseline", "per-coefficient 8-bit uniform quantisation",
           q.decode(), npz_bytes(codes=q.codes, scale=q.scale, offset=q.offset))

    for deg in (2, 1, 0):
        m = band_mask(deg)
        kept = x[:, m]
        x_hat = np.zeros_like(x)
        x_hat[:, m] = kept
        yield (f"deg{deg}", "baseline", f"keep SH degree <= {deg} (float32), drop the rest",
               x_hat, npz_bytes(sh=kept))
        if deg < 2:
            qd = scalar_quantize(kept, 8)
            x_hat = np.zeros_like(x)
            x_hat[:, m] = qd.decode()
            yield (f"deg{deg}_q8", "baseline", f"keep SH degree <= {deg}, 8-bit uniform",
                   x_hat, npz_bytes(codes=qd.codes, scale=qd.scale, offset=qd.offset))

    for k in vq_sizes:
        t0 = time.time()
        vq = kmeans_vq(x, k, seed=seed)
        print(f"  rawvq_k{k}: K-means {time.time() - t0:.0f} s", flush=True)
        yield (f"rawvq_k{k}", "baseline", f"K-means VQ of raw 48-D SH vectors, K={k}",
               vq.decode(), npz_bytes(codebook=vq.codebook, indices=vq.indices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True, help="model cameras.json")
    ap.add_argument("--ablation-dir", default="results/experiments/ablation")
    ap.add_argument("--image-metrics-dir", default="results/experiments/image_metrics")
    ap.add_argument("--out-dir", default="results/experiments/sh_baselines")
    ap.add_argument("--vq-sizes", type=int, nargs="*", default=[64, 256])
    ap.add_argument("--extra-scalar", nargs="*", default=["4:8", "12:8", "24:8", "32:8", "48:8", "16:16"],
                    help="extra thesis scalar configs rank:bits (proxy only, not rendered)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, x = extract_flat_sh(args.ply)
    n = x.shape[0]
    ply_bytes = os.path.getsize(args.ply)
    xyz = np.stack([model.fields[k] for k in ("x", "y", "z")], axis=1).astype(np.float64)
    views = test_views(args.cameras)
    print(f"N={n:,}; test views={len(views)}; building colour proxy...", flush=True)
    proxy = ColourProxy(xyz, model.fields["opacity"].astype(np.float64), views)

    rendered = {}
    for f in glob.glob(os.path.join(args.image_metrics_dir, "metrics_*.json")):
        m = json.load(open(f))
        rendered[m["config"]["name"]] = m["summary"]

    def row(name, family, desc, x_hat, nbytes):
        diff = x_hat - x
        r = {
            "name": name, "family": family, "description": desc,
            "file_bytes": nbytes, "bytes_per_gaussian": nbytes / n,
            **ratios(n, nbytes, baseline_ply_bytes=ply_bytes),
            "coeff_rel_err": float(np.linalg.norm(diff) / np.linalg.norm(x)),
            "dc_rel_err": float(np.linalg.norm(diff[:, DC_COLS]) / np.linalg.norm(x[:, DC_COLS])),
            "rest_rel_err": float(np.linalg.norm(diff[:, REST_COLS]) / np.linalg.norm(x[:, REST_COLS])),
            "colour_proxy_psnr": proxy.psnr(x, x_hat),
        }
        s = rendered.get(name, {})
        for k in ("vsbase_psnr", "vsbase_ssim", "vsbase_lpips", "gt_psnr"):
            r[f"rendered_{k}"] = s.get(k)
        print(f"{name:14s} {nbytes:>12,} B  {r['sh_compression_ratio']:7.1f}x  "
              f"rel={r['coeff_rel_err']:.4f}  proxy={r['colour_proxy_psnr']:.2f} dB", flush=True)
        return r

    rows = []
    for asset_path in sorted(glob.glob(os.path.join(args.ablation_dir, "*", "asset.npz"))):
        name = os.path.basename(os.path.dirname(asset_path))
        asset = CompressedSH.load(asset_path)
        rows.append(row(name, "thesis", f"{asset.mode}, rank {asset.rank}",
                        asset.reconstruct_flat(), os.path.getsize(asset_path)))
    for spec in args.extra_scalar:
        r, b = (int(v) for v in spec.split(":"))
        asset = compress(x, rank=r, mode="scalar", scalar_bits=b)
        with tempfile.TemporaryDirectory() as d:
            nbytes = asset.save(os.path.join(d, "a.npz"))
        rows.append(row(f"sq_r{r}_b{b}", "thesis_extra", f"scalar {b}-bit, rank {r} (proxy only)",
                        asset.reconstruct_flat(), nbytes))
    for item in baseline_methods(x, args.vq_sizes, args.seed):
        rows.append(row(*item))

    # proxy validity: agreement with rendered PSNR vs baseline (finite values only)
    pairs = [(r["colour_proxy_psnr"], r["rendered_vsbase_psnr"]) for r in rows
             if r["rendered_vsbase_psnr"] is not None and r["rendered_vsbase_psnr"] < 99]
    p, q = np.array(pairs).T
    rank = lambda v: np.argsort(np.argsort(v))
    validation = {
        "n_configs": len(pairs),
        "pearson": float(np.corrcoef(p, q)[0, 1]),
        "spearman": float(np.corrcoef(rank(p), rank(q))[0, 1]),
        "pairs_proxy_vs_rendered": pairs,
    }
    print(f"proxy vs rendered PSNR: pearson={validation['pearson']:.3f} "
          f"spearman={validation['spearman']:.3f} over {len(pairs)} configs")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "sh_baselines.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out_dir, "sh_baselines.json"), "w") as f:
        json.dump({"n_points": n, "baseline_ply_bytes": ply_bytes,
                   "n_test_views": len(views), "proxy_validation": validation,
                   "rows": rows}, f, indent=2)
    print(f"wrote {len(rows)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
