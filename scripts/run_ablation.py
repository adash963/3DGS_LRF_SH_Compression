"""Run the compression ablation matrix from a YAML config.

Produces per-experiment directories (asset + config + metrics + git commit) and an
aggregated all_results.csv/json of coefficient-space + storage metrics. Rendering and
image metrics are added by the Colab stage using reconstruct_model.py / render.py /
evaluate_model.py per configuration.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import yaml

from sh_compression.extraction import compress, extract_flat_sh
from sh_compression.storage import ratios


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out-dir", default="results/experiments/ablation")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _, flat = extract_flat_sh(args.ply)
    baseline_bytes = os.path.getsize(args.ply)
    commit = git_commit()
    rows = []
    for exp in cfg["experiments"]:
        name = exp["name"]
        exp_dir = os.path.join(args.out_dir, name)
        os.makedirs(exp_dir, exist_ok=True)
        try:
            asset = compress(
                flat, rank=exp["rank"], mode=exp.get("mode", "lowrank"),
                codebook_size=exp.get("codebook_size", 256),
                scalar_bits=exp.get("scalar_bits", 8),
                seed=exp.get("seed", 0),
                factor_dtype=exp.get("factor_dtype", "float32"),
                source_model=os.path.abspath(args.ply),
            )
            size = asset.save(os.path.join(exp_dir, "asset.npz"))
            rec = asset.reconstruct_flat()
            diff = rec - flat
            row = {
                "name": name, "status": "ok", **{k: exp.get(k) for k in
                    ("rank", "mode", "codebook_size", "factor_dtype", "scalar_bits")},
                "coeff_mse": float(np.mean(diff**2)),
                "coeff_rel_frobenius_error":
                    float(np.linalg.norm(diff) / np.linalg.norm(flat)),
                "asset_file_bytes": size,
                **ratios(flat.shape[0], size, baseline_ply_bytes=baseline_bytes),
            }
        except Exception as e:
            row = {"name": name, "status": f"failed: {e}"}
        rows.append(row)
        with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(exp, f)
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump(row, f, indent=2)
        with open(os.path.join(exp_dir, "git_commit.txt"), "w") as f:
            f.write(commit + "\n")
        print(f"{name}: {row.get('status')}  bytes={row.get('asset_file_bytes')}")

    os.makedirs(args.out_dir, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(args.out_dir, "all_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
        json.dump({"git_commit": commit, "rows": rows}, f, indent=2)
    print(f"wrote {len(rows)} results to {args.out_dir}")


if __name__ == "__main__":
    main()
