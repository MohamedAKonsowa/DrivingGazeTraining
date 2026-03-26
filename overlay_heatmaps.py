"""
Overlay reference and user heatmaps by timestamp.

Defaults:
- Reference: scout_heatmaps_5s/scout_heatmap_tXXXXXs.png
- User: eyetrax_heatmaps_first30/eyetrax_user_heatmap_tXXXXXs.png
"""

from pathlib import Path
import argparse
import re

import cv2
import numpy as np


TIME_RE = re.compile(r"_t(\d+)s")


def extract_t_seconds(path: Path):
    m = TIME_RE.search(path.stem)
    return int(m.group(1)) if m else None


def index_by_time(paths):
    out = {}
    for p in paths:
        t = extract_t_seconds(p)
        if t is not None:
            out[t] = p
    return out


def save_grid(images, out_path, cols=3):
    if not images:
        return
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / cols))
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    cv2.imwrite(str(out_path), grid)


def to_norm_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.max() > 0:
        gray /= gray.max()
    return gray


def run(args):
    ref_dir = Path(args.reference_dir).resolve()
    user_dir = Path(args.user_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_paths = sorted(ref_dir.glob(args.reference_glob))
    user_paths = sorted(user_dir.glob(args.user_glob))
    if not ref_paths:
        raise RuntimeError(f"No reference heatmaps found in {ref_dir} with {args.reference_glob}")
    if not user_paths:
        raise RuntimeError(f"No user heatmaps found in {user_dir} with {args.user_glob}")

    ref_by_t = index_by_time(ref_paths)
    user_by_t = index_by_time(user_paths)
    common_times = sorted(set(ref_by_t.keys()) & set(user_by_t.keys()))
    if not common_times:
        raise RuntimeError("No matching timestamps between reference and user heatmaps.")

    report = ["t_sec,reference_file,user_file,overlay_file"]
    overlay_images = []
    for t in common_times:
        ref = cv2.imread(str(ref_by_t[t]))
        usr = cv2.imread(str(user_by_t[t]))
        if ref is None or usr is None:
            continue

        if ref.shape[:2] != usr.shape[:2]:
            usr = cv2.resize(usr, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)

        ref_gray = to_norm_gray(ref)
        usr_gray = to_norm_gray(usr)

        # Use distinct colors:
        # - Reference as BLUE channel
        # - User as RED channel
        # Overlap appears magenta.
        overlay = np.zeros_like(ref, dtype=np.uint8)
        overlay[:, :, 0] = np.clip(ref_gray * 255.0, 0, 255).astype(np.uint8)  # B
        overlay[:, :, 2] = np.clip(usr_gray * 255.0, 0, 255).astype(np.uint8)  # R

        cv2.putText(
            overlay,
            f"t={t:02d}s | BLUE=SCOUT reference  RED=EyeTrax user  MAGENTA=overlap",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_name = f"overlay_t{t:05d}s.png"
        cv2.imwrite(str(out_dir / out_name), overlay)
        overlay_images.append(overlay)
        report.append(f"{t},{ref_by_t[t].name},{user_by_t[t].name},{out_name}")

    save_grid(overlay_images, out_dir / "overlay_preview_grid.png", cols=3)
    (out_dir / "overlay_report.csv").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved {len(overlay_images)} overlays to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Overlay reference and user heatmaps by timestamp.")
    p.add_argument("--reference-dir", default="scout_heatmaps_5s")
    p.add_argument("--user-dir", default="eyetrax_heatmaps_first30")
    p.add_argument("--reference-glob", default="scout_heatmap_t*s.png")
    p.add_argument("--user-glob", default="eyetrax_user_heatmap_t*s.png")
    p.add_argument("--output-dir", default="overlay_heatmaps_first30")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
