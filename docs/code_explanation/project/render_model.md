# `scripts/render_model.py`

## Purpose

Renders an arbitrary variant PLY (round-trip, low-rank, quantised) using the
**unmodified** upstream renderer. The technical problem it solves: `render.py`
expects a model *directory* with a specific structure and config file, not a loose
PLY. This script builds a throwaway directory of that shape, renders, and collects
the output — without ever touching the baseline checkpoint.

## Line-by-line

### Lines 1–16 — Header

```python
"""Render an arbitrary model PLY (baseline, round-trip, or reconstructed variant)
with the unmodified render.py, staging a temporary model directory so the baseline
checkpoint directory is never touched.

Outputs land in <staging>/test/ours_<iter>/renders and are copied to --out-dir.
"""
import argparse, os, shutil, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

"So the baseline checkpoint directory is never touched" is the safety property.
Rendering a variant in place would overwrite the baseline's own renders, destroying
the reference every comparison depends on — and it would do so silently.

### Lines 19–29 — Arguments and staging

```python
    ap.add_argument("--ply", required=True, help="variant point_cloud.ply to render")
    ap.add_argument("--source", required=True, help="COLMAP scene dir")
    ap.add_argument("--staging-dir", required=True, help="temp model dir to create")
    ap.add_argument("--out-dir", required=True, help="where renders are copied")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--keep-staging", action="store_true")
```

`--iteration` defaults to 30000 because that is the checkpoint's iteration number,
which upstream encodes in directory names.

```python
    pc_dir = os.path.join(args.staging_dir, "point_cloud", f"iteration_{args.iteration}")
    os.makedirs(pc_dir, exist_ok=True)
    shutil.copyfile(args.ply, os.path.join(pc_dir, "point_cloud.ply"))
```

**Lines 27–29 build the directory layout upstream requires**:

```
<staging>/point_cloud/iteration_30000/point_cloud.ply
```

`render.py` searches for exactly this structure. Rather than modify the renderer to
accept a loose file — which would break the "unmodified baseline" property the whole
comparison rests on — the script constructs what it expects.

### Lines 30–37 — Config and render

```python
    subprocess.check_call([
        sys.executable, os.path.join(REPO, "scripts", "fix_cfg_args.py"),
        "--model-dir", args.staging_dir, "--source-path", args.source,
    ])
    subprocess.check_call([
        sys.executable, os.path.join(REPO, "render.py"),
        "-m", args.staging_dir, "-s", args.source, "--skip_train",
    ], cwd=REPO)
```

Line 30 generates a valid `cfg_args` for the staging directory (see
[fix_cfg_args.md](fix_cfg_args.md)) — the staging directory is new, so it has no
config at all and one must be created.

Line 34 invokes the **unmodified** `render.py`. `--skip_train` renders only the test
split, the 38 evaluation views. `cwd=REPO` matters because upstream code uses
relative imports and paths that assume the repository root.

`check_call` raises on non-zero exit, so a render failure propagates immediately
rather than leaving the caller to discover an empty directory later.

### Lines 39–47 — Collect output

```python
    renders = os.path.join(args.staging_dir, "test", f"ours_{args.iteration}", "renders")
    if not os.path.isdir(renders) or not os.listdir(renders):
        raise SystemExit(f"no renders produced in {renders}")
    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(renders)):
        shutil.copyfile(os.path.join(renders, f), os.path.join(args.out_dir, f))
        n += 1
    print(f"copied {n} renders to {args.out_dir}")
```

Lines 40–41 verify renders actually appeared. `render.py` can exit successfully while
producing nothing — for instance if the test split is empty because `eval=False` in
the config, which is precisely the failure mode the original checkpoint's `cfg_args`
would have caused. Checking output existence converts that silent no-op into an
error.

Lines 44–46 copy each render to the output directory in sorted order, and line 47
reports the count — the visible "copied 38 renders" line in the Colab logs.

### Lines 48–49 — Cleanup

```python
    if not args.keep_staging:
        shutil.rmtree(os.path.join(args.staging_dir, "point_cloud"))
```

Deletes the copied 265 MB PLY, keeping only the renders. Across twelve
configurations this saves ~3 GB — significant on a Colab runtime with limited disk.

Note it removes only the `point_cloud` subdirectory, not the whole staging directory,
so the renders remain in place for inspection and the `cfg_args` is reusable on the
next configuration.

## Why it matters

This script is what makes the phrase "the unmodified baseline renderer" literally
true. Every configuration in the dissertation — baseline, round-trip, low-rank,
scalar, VQ — renders through exactly the same `render.py`, invoked exactly the same
way, with only the PLY contents differing.

That is the experimental control that lets any measured quality difference be
attributed to the compression rather than to a rendering difference. It also means
none of the results depend on renderer modifications that a marker would have to
audit.
