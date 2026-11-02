"""Reconstruct a renderable PLY from a compressed asset + the baseline model."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sh_compression.extraction import decompress_to_ply
from sh_compression.serialization import CompressedSH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--baseline-ply", required=True)
    ap.add_argument("--out-ply", required=True)
    args = ap.parse_args()
    asset = CompressedSH.load(args.asset)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_ply)), exist_ok=True)
    decompress_to_ply(asset, args.baseline_ply, args.out_ply)
    print(f"wrote {args.out_ply} ({os.path.getsize(args.out_ply):,} bytes) — "
          "temporary render artefact, NOT the compressed size")


if __name__ == "__main__":
    main()
