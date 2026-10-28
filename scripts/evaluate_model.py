"""Compute PSNR/SSIM/(LPIPS) between two render directories."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sh_compression.metrics import compare_dirs, save_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True, help="reference renders / ground truth")
    ap.add_argument("--dir-b", required=True, help="candidate renders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()
    result = compare_dirs(args.dir_a, args.dir_b, use_lpips=not args.no_lpips)
    save_metrics(result, args.out)
    print(json.dumps(result["summary"], indent=2))
    if result["failed_views"]:
        print(f"WARNING: {len(result['failed_views'])} failed views", file=sys.stderr)


if __name__ == "__main__":
    main()
