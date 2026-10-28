"""Extract the flat SH matrix from a model PLY into an .npz for offline analysis."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from sh_compression.extraction import extract_flat_sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    _, flat = extract_flat_sh(args.ply)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, sh_flat=flat)
    print(f"wrote {args.out}: shape {flat.shape}, "
          f"{os.path.getsize(args.out if args.out.endswith('.npz') else args.out + '.npz'):,} bytes")


if __name__ == "__main__":
    main()
