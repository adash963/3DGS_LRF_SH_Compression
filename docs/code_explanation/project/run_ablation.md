# `scripts/run_ablation.py`

## Purpose

Runs the full storage/coefficient experiment matrix — all 22 configurations — from a
YAML file, producing per-experiment artefact directories plus aggregated CSV and
JSON. This is the CPU-only half of the ablation; `run_render_eval.py` adds rendered
quality for a chosen subset.

## Line-by-line

### Lines 1–29 — Header and `git_commit`

```python
"""Run the compression ablation matrix from a YAML config.

Produces per-experiment directories (asset + config + metrics + git commit) and an
aggregated all_results.csv/json of coefficient-space + storage metrics. Rendering and
image metrics are added by the Colab stage ...
"""
```

```python
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:
        return "unknown"
```

Captures the current commit so every experiment records the code version that
produced it — a direct requirement of the brief.

`cwd=os.path.dirname(__file__)` runs git relative to this script's location, so it
finds the right repository regardless of the caller's working directory. The broad
`except` returning `"unknown"` is a deliberate choice: an experiment should not fail
because git is unavailable (as in an exported tarball), but the record must show that
provenance is missing rather than fabricate it.

### Lines 32–44 — Setup

```python
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _, flat = extract_flat_sh(args.ply)
    baseline_bytes = os.path.getsize(args.ply)
    commit = git_commit()
    rows = []
```

`yaml.safe_load` rather than `load` — safe_load cannot instantiate arbitrary Python
objects, which is the correct default for reading a config file.

As in the render driver, the SH matrix is extracted **once** and shared across all 22
configurations: both an optimisation and an experimental control.

### Lines 45–72 — The experiment loop

```python
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
```

**The `try`/`except` around each configuration is a methodological decision, not
defensive habit.** A failure in one configuration must not abort the other 21, and —
more importantly — it must be *recorded* rather than silently skipped. The failure
becomes a row with `status: "failed: <reason>"`, so the aggregate CSV shows exactly
what was attempted and what succeeded.

This is what makes "zero failed runs" a verifiable claim: every configuration in the
YAML has a row, and all 22 say `ok`. Had the loop simply crashed or skipped, the
absence of a row would be ambiguous.

The `status` field carries the exception text, so a failure is diagnosable from the
committed record without re-running.

### Lines 73–81 — Per-experiment artefacts

```python
        rows.append(row)
        with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(exp, f)
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump(row, f, indent=2)
        with open(os.path.join(exp_dir, "git_commit.txt"), "w") as f:
            f.write(commit + "\n")
        print(f"{name}: {row.get('status')}  bytes={row.get('asset_file_bytes')}")
```

Each experiment directory gets the configuration that produced it, its metrics, and
the code commit. Written **inside the loop**, so results survive an interruption.

Note these three files are written even on failure — the config and commit are still
recorded, and `metrics.json` holds the failure status. The evidence trail does not
depend on success.

The experiment plan also promised `environment.json`, stdout/stderr logs, and sample
renders per directory. Those are absent, which the static audit correctly flagged as
a gap between the plan and the implementation.

### Lines 83–92 — Aggregation

```python
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(args.out_dir, "all_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
        json.dump({"git_commit": commit, "rows": rows}, f, indent=2)
    print(f"wrote {len(rows)} results to {args.out_dir}")
```

Line 83 collects the union of all keys across all rows — necessary because failed
rows have fewer fields than successful ones, and `DictWriter` requires a complete
field list up front. Sorting makes column order deterministic across runs, so the CSV
diffs cleanly in git.

Both formats are written: CSV for spreadsheet inspection, JSON for figure generation
(`generate_report_assets.py` reads the JSON).

## A note on the assets

Each experiment directory contains `asset.npz` — but these are **gitignored**. The
full-rank asset alone is 185 MB, exceeding GitHub's 100 MB hard limit, and the whole
matrix approaches 700 MB.

What is committed is the metrics, configs, and commit hashes. The assets are
regenerable by re-running this script, which is deterministic given the same input
model and seeds. The trade-off is documented rather than hidden.

## Why it matters

This script produced the storage half of the ablation — all 22 configurations with
measured file sizes, forming the dissertation's rate axis. Its structure encodes
three commitments:

1. **Configurations come from a committed YAML**, so results trace to a versioned
   definition.
2. **Failures are recorded, not skipped**, so "zero failures" means something.
3. **Every experiment carries its own config and commit**, so no result depends on
   remembering how it was produced.
