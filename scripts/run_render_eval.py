"""Stage 4-7 driver: compress → serialise → reconstruct → render → evaluate,
for a named subset of the ablation matrix, on a GPU machine.

Per config, produces under --out-root:
  assets/<name>.npz                     compressed asset (measured bytes)
  image_metrics/metrics_<name>.json     PSNR/SSIM(/LPIPS) vs GT + vs baseline renders
  renders/<name>/                       rendered eval views
Baseline renders/GT must already exist (render.py --skip_train on the baseline).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import yaml

from sh_compression.extraction import compress, decompress_to_ply, extract_flat_sh
from sh_compression.metrics import compare_dirs, save_metrics
from sh_compression.serialization import CompressedSH
from sh_compression.storage import ratios

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="quantization_ablation.yaml")
    ap.add_argument("--names", nargs="+", required=True, help="experiment names to run")
    ap.add_argument("--ply", required=True, help="baseline point_cloud.ply")
    ap.add_argument("--source", required=True, help="COLMAP scene dir")
    ap.add_argument("--baseline-renders", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = {e["name"]: e for e in yaml.safe_load(f)["experiments"]}
    unknown = [n for n in args.names if n not in cfg]
    if unknown:
        raise SystemExit(f"unknown experiment names: {unknown}")

    print("extracting SH once...")
    _, flat = extract_flat_sh(args.ply)
    baseline_ply_bytes = os.path.getsize(args.ply)

    os.makedirs(os.path.join(args.out_root, "assets"), exist_ok=True)
    os.makedirs(os.path.join(args.out_root, "image_metrics"), exist_ok=True)
    summary_rows = []
    for name in args.names:
        e = cfg[name]
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        asset = compress(
            flat, rank=e["rank"], mode=e.get("mode", "lowrank"),
            codebook_size=e.get("codebook_size", 256),
            scalar_bits=e.get("scalar_bits", 8), seed=e.get("seed", 0),
            factor_dtype=e.get("factor_dtype", "float32"),
            source_model=os.path.abspath(args.ply),
        )
        t_compress = time.time() - t0
        asset_path = os.path.join(args.out_root, "assets", f"{name}.npz")
        size = asset.save(asset_path)

        t0 = time.time()
        loaded = CompressedSH.load(asset_path)  # clean-process-equivalent load
        variant_ply = os.path.join(args.out_root, "variant.ply")
        decompress_to_ply(loaded, args.ply, variant_ply)
        t_reconstruct = time.time() - t0

        render_dir = os.path.join(args.out_root, "renders", name)
        t0 = time.time()
        subprocess.check_call([
            sys.executable, os.path.join(REPO, "scripts", "render_model.py"),
            "--ply", variant_ply, "--source", args.source,
            "--staging-dir", os.path.join(args.out_root, "staging"),
            "--out-dir", render_dir,
        ])
        t_render = time.time() - t0

        vs_gt = compare_dirs(args.gt_dir, render_dir, use_lpips=not args.no_lpips)
        vs_base = compare_dirs(args.baseline_renders, render_dir,
                               use_lpips=not args.no_lpips)
        rec = loaded.reconstruct_flat()
        diff = rec - flat
        result = {
            "name": name, "config": e,
            "asset_file_bytes": size,
            **ratios(flat.shape[0], size, baseline_ply_bytes=baseline_ply_bytes),
            "coeff_rel_frobenius_error":
                float(np.linalg.norm(diff) / np.linalg.norm(flat)),
            "timing_s": {"compress": round(t_compress, 2),
                         "reconstruct": round(t_reconstruct, 2),
                         "render_total": round(t_render, 2)},
            "summary": {
                **{f"gt_{k[5:]}" if k.startswith("mean_") else k: v
                   for k, v in vs_gt["summary"].items()},
                **{f"vsbase_{k[5:]}" if k.startswith("mean_") else k: v
                   for k, v in vs_base["summary"].items()},
            },
            "n_views": vs_gt["n_views"],
            "failed_views": vs_gt["failed_views"] + vs_base["failed_views"],
            "per_view_vs_gt": vs_gt["per_view"],
            "per_view_vs_baseline": vs_base["per_view"],
        }
        save_metrics(result, os.path.join(args.out_root, "image_metrics",
                                          f"metrics_{name}.json"))
        summary_rows.append({"name": name, "bytes": size, **result["summary"]})
        print(json.dumps({"name": name, "bytes": size, **result["summary"]}, indent=2))

    print("\n==== SUMMARY ====")
    for r in summary_rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
