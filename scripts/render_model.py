"""Render an arbitrary model PLY (baseline, round-trip, or reconstructed variant)
with the unmodified render.py, staging a temporary model directory so the baseline
checkpoint directory is never touched.

Outputs land in <staging>/test/ours_<iter>/renders and are copied to --out-dir.
"""
import argparse
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="variant point_cloud.ply to render")
    ap.add_argument("--source", required=True, help="COLMAP scene dir")
    ap.add_argument("--staging-dir", required=True, help="temp model dir to create")
    ap.add_argument("--out-dir", required=True, help="where renders are copied")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--keep-staging", action="store_true")
    args = ap.parse_args()

    pc_dir = os.path.join(args.staging_dir, "point_cloud", f"iteration_{args.iteration}")
    os.makedirs(pc_dir, exist_ok=True)
    shutil.copyfile(args.ply, os.path.join(pc_dir, "point_cloud.ply"))
    subprocess.check_call([
        sys.executable, os.path.join(REPO, "scripts", "fix_cfg_args.py"),
        "--model-dir", args.staging_dir, "--source-path", args.source,
    ])
    subprocess.check_call([
        sys.executable, os.path.join(REPO, "render.py"),
        "-m", args.staging_dir, "-s", args.source, "--skip_train",
    ], cwd=REPO)

    renders = os.path.join(args.staging_dir, "test", f"ours_{args.iteration}", "renders")
    if not os.path.isdir(renders) or not os.listdir(renders):
        raise SystemExit(f"no renders produced in {renders}")
    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(renders)):
        shutil.copyfile(os.path.join(renders, f), os.path.join(args.out_dir, f))
        n += 1
    print(f"copied {n} renders to {args.out_dir}")
    if not args.keep_staging:
        shutil.rmtree(os.path.join(args.staging_dir, "point_cloud"))


if __name__ == "__main__":
    main()
