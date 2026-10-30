# `utils/system_utils.py`

## Purpose

Two filesystem helpers. Small, but `searchForMaxIteration` determines which checkpoint
gets loaded, which makes it quietly relevant to our staging approach.

## Line-by-line

### Lines 16–24 — `mkdir_p`

```python
def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise
```

Idempotent directory creation. Catches the "already exists" error and ignores it,
**but only if the path is genuinely a directory** (line 21) — if a *file* exists at
that path, the error re-raises rather than being swallowed.

Modern Python offers `os.makedirs(path, exist_ok=True)`, which our code uses; this
implementation predates widespread use of that flag. Behaviour is equivalent.

Called by `gaussian_model.save_ply` to create the output directory.

### Lines 26–28 — `searchForMaxIteration`

```python
def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
```

Finds the highest-numbered checkpoint in a `point_cloud/` directory containing entries
like `iteration_7000`, `iteration_30000`. It splits each name on underscores, takes the
last part, converts to `int`, and returns the maximum.

**This is what resolves `--iteration -1`** (render.py's default) to 30000 for our
checkpoint — the "Loading trained model at iteration 30000" line in every Colab log.

Two fragilities worth noting, since they affect our staging directories:

- **No filtering.** Any file or directory in `point_cloud/` is parsed. A stray
  `.DS_Store` would raise `ValueError` on `int(...)`. Our staging directories contain
  only `iteration_30000`, so this never arose — but it is why the staging layout is
  built precisely rather than casually.
- **`max` on an empty list raises.** An empty `point_cloud/` gives a bare
  `ValueError: max() arg is an empty sequence` rather than a helpful message.

Our `scripts/render_model.py` sidesteps both by creating exactly one correctly-named
directory and passing `--iteration 30000` explicitly rather than relying on the
search.

## Why it matters

Modest, but relevant in one respect: `searchForMaxIteration` defines the
`iteration_<N>` naming convention that our staging directory must satisfy for the
unmodified renderer to find a variant PLY.

Understanding it is part of understanding why `render_model.py` builds

```
<staging>/point_cloud/iteration_30000/point_cloud.ply
```

rather than simply pointing the renderer at a file — and why that approach let us
render 12 compressed configurations without touching a line of upstream code.
