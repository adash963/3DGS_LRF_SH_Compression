"""Image-quality metrics (PSNR, SSIM, optional LPIPS) over render directories."""

import json
import os

import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image as float32 [H, W, 3] in [0, 1]."""
    import imageio.v3 as iio

    img = iio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0
    return img


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))


_LPIPS_MODEL = None
_LPIPS_DEVICE = None


def lpips_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    """LPIPS (AlexNet), on CUDA when available. Returns None when unavailable."""
    global _LPIPS_MODEL, _LPIPS_DEVICE
    try:
        import lpips
        import torch
    except ImportError:
        return None
    if _LPIPS_MODEL is None:
        _LPIPS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False).to(_LPIPS_DEVICE)
    def to_t(x):
        t = torch.from_numpy(x.transpose(2, 0, 1)[None] * 2.0 - 1.0).float()
        return t.to(_LPIPS_DEVICE)
    with torch.no_grad():
        return float(_LPIPS_MODEL(to_t(a), to_t(b)).item())


def compare_dirs(dir_a: str, dir_b: str, use_lpips: bool = True) -> dict:
    """Pairwise metrics over identically named images in two directories.

    Deterministic sorted order; per-view and mean results; failed views recorded.
    """
    names_a = sorted(f for f in os.listdir(dir_a) if f.lower().endswith((".png", ".jpg")))
    names_b = sorted(f for f in os.listdir(dir_b) if f.lower().endswith((".png", ".jpg")))
    common = sorted(set(names_a) & set(names_b))
    if not common:
        raise ValueError(f"no common images between {dir_a} and {dir_b}")
    per_view, failed = [], []
    for name in common:
        try:
            a = load_image(os.path.join(dir_a, name))
            b = load_image(os.path.join(dir_b, name))
            row = {"view": name, "psnr": psnr(a, b), "ssim": ssim(a, b)}
            if use_lpips:
                d = lpips_distance(a, b)
                if d is not None:
                    row["lpips"] = d
            per_view.append(row)
        except Exception as e:  # record, never silently drop
            failed.append({"view": name, "error": str(e)})
    summary = {}
    for key in ("psnr", "ssim", "lpips"):
        vals = [r[key] for r in per_view if key in r and np.isfinite(r[key])]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
    return {
        "n_views": len(common), "per_view": per_view,
        "failed_views": failed, "summary": summary,
        "dir_a": dir_a, "dir_b": dir_b,
    }


def save_metrics(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
