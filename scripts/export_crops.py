"""Print base64 JPEG crops of rendered views for transfer through notebook output.

Used by the Colab session to move small visual-comparison crops into the repo
without credentials. Markers: IMG:<name>:START / IMG:<name>:END per image.
"""
import argparse
import base64
import io
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--renders-root", required=True, help="dir containing <config>/ render dirs")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--view", default="00012.png")
    ap.add_argument("--box", type=int, nargs=4, default=[550, 450, 950, 800],
                    help="left top right bottom")
    ap.add_argument("--quality", type=int, default=85)
    args = ap.parse_args()

    sources = {"gt": os.path.join(args.gt_dir, args.view),
               "baseline": os.path.join(args.baseline_dir, args.view)}
    for c in args.configs:
        sources[c] = os.path.join(args.renders_root, c, args.view)

    for name, path in sources.items():
        buf = io.BytesIO()
        Image.open(path).convert("RGB").crop(tuple(args.box)).save(
            buf, "JPEG", quality=args.quality)
        b = base64.b64encode(buf.getvalue()).decode()
        print(f"IMG:{name}:START")
        for i in range(0, len(b), 400):
            print(b[i:i + 400])
        print(f"IMG:{name}:END")


if __name__ == "__main__":
    main()
