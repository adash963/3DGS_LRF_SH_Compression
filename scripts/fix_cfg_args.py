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
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--source-path", required=True)
    ap.add_argument("--eval", action="store_true", default=True)
    args = ap.parse_args()
    # includes every ModelParams field: render.py builds its parser with
    # sentinel=True (all CLI defaults None), so cfg_args must be complete
    cfg = (
        "Namespace(sh_degree=3, "
        f"source_path='{os.path.abspath(args.source_path)}', "
        f"model_path='{os.path.abspath(args.model_dir)}', "
        "images='images', depths='', resolution=-1, white_background=False, "
        f"train_test_exp=False, data_device='cuda', eval={args.eval})"
    )
    path = os.path.join(args.model_dir, "cfg_args")
    if os.path.exists(path):
        backup = path + ".orig"
        if not os.path.exists(backup):
            os.rename(path, backup)
    with open(path, "w") as f:
        f.write(cfg)
    print(f"wrote {path}: {cfg}")


if __name__ == "__main__":
    main()
