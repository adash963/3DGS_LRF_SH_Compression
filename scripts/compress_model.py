"""Compress a model's SH: factorise (+ optional quantisation), save asset, report bytes."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from sh_compression.extraction import compress, extract_flat_sh
from sh_compression.storage import ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--mode", choices=["lowrank", "vq", "scalar"], default="lowrank")
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--scalar-bits", type=int, default=8)
    ap.add_argument("--factor-dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output .npz asset path")
    args = ap.parse_args()

    _, flat = extract_flat_sh(args.ply)
    asset = compress(
        flat, rank=args.rank, mode=args.mode, codebook_size=args.codebook_size,
        scalar_bits=args.scalar_bits, seed=args.seed, factor_dtype=args.factor_dtype,
        source_model=os.path.abspath(args.ply),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    size = asset.save(args.out)

    rec = asset.reconstruct_flat()
    diff = rec - flat
    report = {
        "config": vars(args),
        "n_points": int(flat.shape[0]),
        "coeff_mse": float(np.mean(diff**2)),
        "coeff_rel_frobenius_error": float(np.linalg.norm(diff) / np.linalg.norm(flat)),
        "coeff_max_abs_error": float(np.max(np.abs(diff))),
        "asset_file_bytes": size,
        "component_bytes": asset.component_bytes(),
        **ratios(flat.shape[0], size, baseline_ply_bytes=os.path.getsize(args.ply)),
    }
    out_json = os.path.splitext(args.out)[0] + ".metrics.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
