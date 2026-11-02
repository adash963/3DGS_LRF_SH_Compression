# `scripts/fix_cfg_args.py`

## Purpose

Repairs the pretrained checkpoint's configuration file. This script exists because of
**two real bugs** encountered during Stage 1, both documented in
`DEVELOPMENT_LOG.md`. Without it, nothing in the project renders at all.

## The problem

The distributed `GaussianViewTest` checkpoint ships a `cfg_args` file containing:

```python
Namespace(sh_degree=3, source_path='C:\Users\PC\Desktop\cloud\train', ...)
```

Upstream `arguments/__init__.py` loads this by calling **`eval()`** on the file's
text. Two failures follow.

**Failure 1 — the Windows path crashes the parser.** In a Python string literal,
`\U` begins a unicode escape sequence expecting eight hex digits. `\Users` is not
valid, so `eval()` raises before any rendering begins:

```
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes
in position 2-3: truncated \UXXXXXXXX escape
```

**Failure 2 — missing fields.** `render.py` constructs its parser with
`sentinel=True`, which sets every command-line default to `None`. The merge logic
then keeps only values that are not `None`, meaning **all** configuration must come
from `cfg_args`. The 2023-era checkpoint predates fields the current code requires:

```
AttributeError: 'GroupParams' object has no attribute 'depths'
```

A third, quieter problem: the file says `eval=False`, meaning no test/train split.
`render.py --skip_train` would then render zero views and exit successfully — a
silent no-op that would have been easy to mistake for success.

## Line-by-line

### Lines 1–12 — Docstring

```python
"""Rewrite a pretrained model's cfg_args for the current machine.

The distributed GaussianViewTest checkpoint carries a Windows-authored cfg_args
(`source_path='C:\\Users\\...'`) which crashes render.py's eval() with a unicode
escape error, and sets eval=False (no test split). This writes a valid POSIX
cfg_args pointing at the given paths, with eval=True so --skip_train renders the
standard every-8th-camera test split.

Caveat (documented for the report): the checkpoint was trained with eval=False,
i.e. on ALL cameras. Our "test" views were seen in training, so metrics measure
compression degradation relative to the baseline, not novel-view generalisation.
"""
```

The docstring records the diagnosis **and** a methodological caveat that matters more
than the bug itself.

Setting `eval=True` creates a test split of every 8th camera. But the model was
*trained* with `eval=False`, meaning it saw all 301 cameras during optimisation. So
the 38 "test" views were part of training.

This is a genuine limitation, and it is stated in the report rather than glossed
over. Its consequence is precise: the metrics measure **compression degradation
relative to the baseline**, not novel-view generalisation. That happens to be exactly
what this project studies — the comparison is compressed-vs-uncompressed, both
rendering the same views — so the limitation does not invalidate the results. But
quoting the ground-truth PSNR as evidence of view synthesis quality would be wrong,
and the caveat prevents that misreading.

### Lines 15–24 — Arguments

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--source-path", required=True)
    ap.add_argument("--eval", action="store_true", default=True)
    args = ap.parse_args()
```

`--eval` defaults to `True`, which is the whole point — the checkpoint's `False` is
what would have produced an empty test set.

### Lines 25–32 — Building the config

```python
    # includes every ModelParams field: render.py builds its parser with
    # sentinel=True (all CLI defaults None), so cfg_args must be complete
    cfg = (
        "Namespace(sh_degree=3, "
        f"source_path='{os.path.abspath(args.source_path)}', "
        f"model_path='{os.path.abspath(args.model_dir)}', "
        "images='images', depths='', resolution=-1, white_background=False, "
        f"train_test_exp=False, data_device='cuda', eval={args.eval})"
    )
```

The complete `ModelParams` field set, in `eval()`-able form. The comment explains
*why* completeness is required rather than merely noting that it is — the sentinel
parser is the reason, and without that context the field list looks arbitrary.

`os.path.abspath` produces POSIX paths with forward slashes, which contain no
backslash escape sequences and therefore parse cleanly.

`depths=''` and `train_test_exp=False` are the two fields whose absence caused
Failure 2. Both are set to the values that disable the features they control, since
neither depth regularisation nor exposure compensation is used here.

### Lines 33–40 — Writing, with backup

```python
    path = os.path.join(args.model_dir, "cfg_args")
    if os.path.exists(path):
        backup = path + ".orig"
        if not os.path.exists(backup):
            os.rename(path, backup)
    with open(path, "w") as f:
        f.write(cfg)
    print(f"wrote {path}: {cfg}")
```

Lines 34–37 preserve the original as `cfg_args.orig` — the distributed file remains
inspectable, which matters for reproducing the diagnosis.

**The nested `if` on line 36 makes the script idempotent.** On a second run the
backup already exists, so it is not overwritten with the already-fixed version. Since
`render_model.py` calls this script before every render, this guard is exercised
constantly; without it the original would be lost on the second invocation.

Line 40 echoes the written config, which is why the exact configuration appears in
the Colab logs for every render — useful evidence when reconstructing what a run
actually did.

## Why it matters

Beyond unblocking the project, this file is good dissertation material. It
demonstrates:

- **Diagnosis rather than workaround.** The fix addresses the sentinel-parser
  mechanism, not just the immediate symptom.
- **Fixes in version-controlled code**, not in notebook cells — satisfying the
  brief's requirement that the notebook stay a thin shell.
- **A methodological caveat surfaced from a technical detail.** Noticing that
  `eval=False` in the checkpoint means the test views were seen during training is
  the kind of subtlety that separates a careful project from a superficial one, and
  it is recorded in the report's limitations rather than buried.
