"""
Build per-5s heatmaps from WebGazer sample export JSON.

Input JSON is exported by webgazer_demo.html as:
- width, height
- duration_sec, interval_sec
- samples: [{x, y, t}, ...]
"""

from pathlib import Path
import argparse
import json

import cv2
import numpy as np


def build_heatmap(points_xy, width, height, blur_sigma=35.0):
    heat = np.zeros((height, width), dtype=np.float32)
    for x, y in points_xy:
        xi = int(np.clip(x, 0, width - 1))
        yi = int(np.clip(y, 0, height - 1))
        heat[yi, xi] += 1.0

    if heat.max() > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), blur_sigma)
        if heat.max() > 0:
            heat /= heat.max()
    return heat


def save_heatmap_png(heat, out_path):
    img = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), color)


def run(args):
    in_path = Path(args.samples_json).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    width = int(data.get("width", 1280))
    height = int(data.get("height", 720))
    duration_sec = float(data.get("duration_sec", 30))
    interval_sec = float(data.get("interval_sec", args.interval_sec))
    samples = data.get("samples", [])

    if not samples:
        raise RuntimeError("No samples in JSON. Record first in webgazer_demo.html.")

    starts = np.arange(0.0, duration_sec, interval_sec).tolist()
    report = ["t_sec,num_points,output_file"]
    preview = []

    for start in starts:
        end = min(start + interval_sec, duration_sec)
        pts = [
            (float(s["x"]), float(s["y"]))
            for s in samples
            if start <= float(s.get("t", -1.0)) < end
        ]
        heat = build_heatmap(pts, width, height, blur_sigma=args.blur_sigma)
        t_sec = int(round(start))
        out_name = f"webgazer_user_heatmap_t{t_sec:05d}s.png"
        out_path = out_dir / out_name
        save_heatmap_png(heat, out_path)
        preview.append(cv2.imread(str(out_path)))
        report.append(f"{t_sec},{len(pts)},{out_name}")

    preview = [p for p in preview if p is not None]
    if preview:
        cols = 3
        h, w = preview[0].shape[:2]
        rows = int(np.ceil(len(preview) / cols))
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for i, img in enumerate(preview):
            r = i // cols
            c = i % cols
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
        cv2.imwrite(str(out_dir / "webgazer_first30_preview_grid.png"), grid)

    (out_dir / "webgazer_first30_report.csv").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved {len(starts)} WebGazer heatmaps to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Create WebGazer per-5s heatmaps from exported samples JSON.")
    p.add_argument("--samples-json", default="webgazer_first30_samples.json")
    p.add_argument("--output-dir", default="webgazer_heatmaps_first30")
    p.add_argument("--interval-sec", type=float, default=5.0)
    p.add_argument("--blur-sigma", type=float, default=35.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
