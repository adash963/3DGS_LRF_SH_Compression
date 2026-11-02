# `scripts/run_render_eval.py`

## Purpose

The full GPU evaluation driver, and the most consequential script in the project. For
each named configuration it runs the complete chain — compress → serialise → load →
reconstruct → render → measure — and writes per-view metrics.

**Every quality number in the dissertation came from this script.**

## Line-by-line

### Lines 1–30 — Header and imports

```python
"""Stage 4-7 driver: compress → serialise → reconstruct → render → evaluate,
for a named subset of the ablation matrix, on a GPU machine.

Per config, produces under --out-root:
  assets/<name>.npz                     compressed asset (measured bytes)
  image_metrics/metrics_<name>.json     PSNR/SSIM(/LPIPS) vs GT + vs baseline renders
  renders/<name>/                       rendered eval views
Baseline renders/GT must already exist (render.py --skip_train on the baseline).
"""
```

The docstring states the contract precisely, including the precondition that baseline
renders must already exist — this script measures *against* them, it does not produce
them.

```python
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

The repository root, computed from this file's location (`scripts/` → parent). Used
to invoke sibling scripts by absolute path regardless of working directory.

### Lines 33–48 — Arguments and config loading

```python
    ap.add_argument("--config", required=True, help="quantization_ablation.yaml")
    ap.add_argument("--names", nargs="+", required=True, help="experiment names to run")
    ...
    with open(args.config) as f:
        cfg = {e["name"]: e for e in yaml.safe_load(f)["experiments"]}
    unknown = [n for n in args.names if n not in cfg]
    if unknown:
        raise SystemExit(f"unknown experiment names: {unknown}")
```

**Configurations come from the YAML file, not the command line.** Only *names* are
passed. This is what makes the experiment matrix authoritative: a run cannot use
settings that differ from the committed configuration, so a result can always be
traced back to a versioned definition.

Lines 46–48 validate every name **before** any work begins — a typo in the twelfth
name fails immediately rather than after forty minutes of rendering.

### Lines 50–58 — One-time setup

```python
    print("extracting SH once...")
    _, flat = extract_flat_sh(args.ply)
    baseline_ply_bytes = os.path.getsize(args.ply)
```

The SH matrix is extracted **once** and reused across all configurations. Reading and
parsing a 265 MB PLY takes real time; doing it twelve times would waste most of an
hour. Note this also means every configuration factorises the *identical* input
matrix — an experimental control, not just an optimisation.

### Lines 59–76 — Compress and serialise

```python
    for name in args.names:
        e = cfg[name]
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        asset = compress(flat, rank=e["rank"], mode=e.get("mode", "lowrank"), ...)
        t_compress = time.time() - t0
        asset_path = os.path.join(args.out_root, "assets", f"{name}.npz")
        size = asset.save(asset_path)
```

`flush=True` on line 62 forces the progress marker out immediately — without it,
Python buffers stdout when piped, and a long run appears frozen.

`e.get("mode", "lowrank")` supplies defaults so the YAML only needs to state what
differs from the default, keeping the config file readable.

```python
        t0 = time.time()
        loaded = CompressedSH.load(asset_path)  # clean-process-equivalent load
        variant_ply = os.path.join(args.out_root, "variant.ply")
        decompress_to_ply(loaded, args.ply, variant_ply)
        t_reconstruct = time.time() - t0
```

**Line 71 is a deliberate verification step.** The asset was just built in memory and
could be used directly — instead it is *saved and re-loaded from disk* before
reconstruction. The comment explains why: this exercises the full serialisation
round trip, so what gets rendered is provably what a fresh process would reconstruct
from the file. Without this, a serialisation bug could hide behind correct in-memory
state.

Timings are captured separately for compress and reconstruct, per the brief's
requirement that timing scopes not be conflated.

### Lines 77–86 — Render

```python
        render_dir = os.path.join(args.out_root, "renders", name)
        t0 = time.time()
        subprocess.check_call([
            sys.executable, os.path.join(REPO, "scripts", "render_model.py"),
            "--ply", variant_ply, "--source", args.source,
            "--staging-dir", os.path.join(args.out_root, "staging"),
            "--out-dir", render_dir,
        ])
        t_render = time.time() - t0
```

Rendering runs as a **subprocess** rather than an in-process import. Three reasons:
the upstream renderer expects to run as a script with its own argument parsing; each
render gets a clean CUDA context, avoiding GPU memory accumulation across twelve
configurations; and a crash in one configuration cannot corrupt the driver's state.

`sys.executable` uses the same interpreter, so the subprocess inherits the current
environment rather than whatever `python` resolves to.

### Lines 87–110 — Evaluate

```python
        vs_gt = compare_dirs(args.gt_dir, render_dir, use_lpips=not args.no_lpips)
        vs_base = compare_dirs(args.baseline_renders, render_dir,
                               use_lpips=not args.no_lpips)
```

**Two comparisons per configuration**, and this is a genuinely important design
decision:

- **vs ground truth** — how close the render is to the actual photograph. Bounded by
  the baseline's own fit (24.93 dB), since compression cannot improve on the model.
- **vs baseline renders** — how much the *compression* changed the image. This
  isolates the effect under study, removing the baseline's own reconstruction error
  from the measurement.

Reporting both is what lets the dissertation say "34.63 dB vs baseline, costing
0.62 dB against ground truth" — two different and complementary statements.

```python
        rec = loaded.reconstruct_flat()
        diff = rec - flat
        result = {
            "name": name, "config": e,
            "asset_file_bytes": size,
            **ratios(flat.shape[0], size, baseline_ply_bytes=baseline_ply_bytes),
            "coeff_rel_frobenius_error": float(np.linalg.norm(diff) / np.linalg.norm(flat)),
            "timing_s": {"compress": ..., "reconstruct": ..., "render_total": ...},
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
```

The result record combines storage, coefficient error, timing, and both image
comparisons in one file.

The dict comprehensions on lines 99–102 rename keys: `mean_psnr` becomes `gt_psnr`
from one comparison and `vsbase_psnr` from the other (`k[5:]` strips the `mean_`
prefix). Without this the two would collide.

**Line 105 concatenates failed views from both comparisons** — failures are never
dropped, satisfying the requirement to record them.

**Lines 106–107 retain every per-view row.** This is what allows the audit to verify
that stored summaries are genuine averages of the stored per-view data — which it
did, confirming they reproduce after rounding.

### Lines 111–118 — Output

```python
        save_metrics(result, os.path.join(args.out_root, "image_metrics",
                                          f"metrics_{name}.json"))
        summary_rows.append({"name": name, "bytes": size, **result["summary"]})
        print(json.dumps({"name": name, "bytes": size, **result["summary"]}, indent=2))

    print("\n==== SUMMARY ====")
    for r in summary_rows:
        print(json.dumps(r))
```

Each configuration's metrics are written **as it completes**, not batched at the end.
If the Colab session died at configuration nine, the first eight results would
survive — which matters on a free-tier runtime with disconnection risk.

The final summary block is what was read out of the notebook to populate the results
tables.

## A documented gap

The static audit noted the image-metric JSONs lack a `git_commit` field and
environment details, which the experiment plan had promised. The configuration is
embedded, but the code version is not. Adding it would be a small change and is on
the outstanding-corrections list.

## Why it matters

This script *is* the experiment. Its structure encodes the methodology:

- one extraction shared by all configurations (controlled input),
- save-and-reload before rendering (verified serialisation),
- identical renderer, cameras and metric code for every configuration (fair
  comparison),
- both reference comparisons (separating compression loss from model error),
- per-view retention and failure recording (no cherry-picking),
- incremental writes (survivable on unreliable infrastructure).

Every one of those is a methodological commitment from the brief, implemented as
code rather than promised as process.
