"""Print structure of a 3DGS PLY: point count, fields, SH degree, byte breakdown."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sh_compression.ply_io import read_gaussian_ply
from sh_compression.storage import baseline_sh_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    args = ap.parse_args()
    m = read_gaussian_ply(args.ply)
    n = m.num_points
    print(f"file: {args.ply}")
    print(f"file bytes: {os.path.getsize(args.ply):,}")
    print(f"points: {n:,}")
    print(f"fields ({len(m.field_names)}): {m.field_names}")
    print(f"SH bytes (N*48*4): {baseline_sh_bytes(n):,} "
          f"({baseline_sh_bytes(n)/os.path.getsize(args.ply):.1%} of file)")


if __name__ == "__main__":
    main()
