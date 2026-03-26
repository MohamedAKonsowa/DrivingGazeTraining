"""
Collect EyeTrax gaze heatmaps for first 30 seconds of a video.

Flow:
1) Optional EyeTrax calibration (5-point by default).
2) Play video (first 30 seconds).
3) Track user gaze via webcam and map to video frame coordinates.
4) Save one heatmap every 5 seconds.
"""

from pathlib import Path
import argparse

import cv2
import numpy as np

from eyetrax.calibration import run_5_point_calibration, run_9_point_calibration
from eyetrax.filters import KalmanEMASmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size


def build_heatmap(points_xy, width, height, sigma=35.0):
    heat = np.zeros((height, width), dtype=np.float32)
    for x, y in points_xy:
        xi = int(np.clip(x, 0, width - 1))
        yi = int(np.clip(y, 0, height - 1))
        heat[yi, xi] += 1.0
    if heat.max() > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), sigma)
        if heat.max() > 0:
            heat /= heat.max()
    return heat


def heat_to_color(heat):
    img = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)


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


def run(args):
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gaze_estimator = GazeEstimator(model_name="ridge")
    if args.calibration == "5p":
        run_5_point_calibration(gaze_estimator, camera_index=args.webcam_id)
    elif args.calibration == "9p":
        run_9_point_calibration(gaze_estimator, camera_index=args.webcam_id)

    kalman = make_kalman()
    smoother = KalmanEMASmoother(kalman, ema_alpha=args.ema_alpha)
    smoother.tune(gaze_estimator, camera_index=args.webcam_id)

    video_cap = cv2.VideoCapture(str(video_path))
    if not video_cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    webcam_cap = cv2.VideoCapture(args.webcam_id)
    if not webcam_cap.isOpened():
        raise RuntimeError(f"Could not open webcam id={args.webcam_id}")

    video_fps = video_cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(args.max_seconds * video_fps)

    interval = args.interval_sec
    segment_starts = np.arange(0.0, args.max_seconds, interval).tolist()
    segment_ends = [min(s + interval, args.max_seconds) for s in segment_starts]
    segment_points = [[] for _ in segment_starts]

    screen_w, screen_h = get_screen_size()
    win = "EyeTrax First30 Capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_idx = 0
    while frame_idx < max_frames:
        ok_video, frame = video_cap.read()
        if not ok_video:
            break
        ok_webcam, cam_frame = webcam_cap.read()
        if not ok_webcam:
            break

        t_sec = frame_idx / video_fps
        seg_idx = min(int(t_sec // interval), len(segment_points) - 1)

        # EyeTrax predicts in screen coordinates.
        features, blink = gaze_estimator.extract_features(cam_frame)
        mapped = None
        if features is not None and not blink:
            gaze_point = gaze_estimator.predict(np.array([features]))[0]
            px, py = map(int, gaze_point)
            sx, sy = smoother.step(px, py)
            mapped = (sx, sy)

        h, w = frame.shape[:2]
        if mapped is not None:
            vx = int(np.clip((mapped[0] / max(screen_w, 1)) * w, 0, w - 1))
            vy = int(np.clip((mapped[1] / max(screen_h, 1)) * h, 0, h - 1))
            segment_points[seg_idx].append((vx, vy))
            cv2.circle(frame, (vx, vy), 10, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"t={t_sec:.1f}s / {args.max_seconds}s  seg {seg_idx + 1}/{len(segment_points)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "ESC: stop early",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, frame)
        if cv2.waitKey(max(1, int(1000 / video_fps))) & 0xFF == 27:
            break

        frame_idx += 1

    video_cap.release()
    webcam_cap.release()
    gaze_estimator.close()
    cv2.destroyAllWindows()

    # Build segment heatmaps.
    ref_cap = cv2.VideoCapture(str(video_path))
    ok0, f0 = ref_cap.read()
    ref_cap.release()
    if not ok0:
        raise RuntimeError("Could not read first frame for output size.")
    h, w = f0.shape[:2]

    out_imgs = []
    report = ["segment,start_s,end_s,num_points,file"]
    for i, pts in enumerate(segment_points):
        heat = build_heatmap(pts, w, h, sigma=args.sigma)
        color = heat_to_color(heat)
        name = f"eyetrax_user_heatmap_t{int(segment_starts[i]):05d}s.png"
        cv2.imwrite(str(out_dir / name), color)
        out_imgs.append(color)
        report.append(f"{i + 1},{segment_starts[i]:.2f},{segment_ends[i]:.2f},{len(pts)},{name}")

    (out_dir / "eyetrax_first30_report.csv").write_text("\n".join(report), encoding="utf-8")
    save_grid(out_imgs, out_dir / "eyetrax_first30_preview_grid.png")
    print(f"Saved heatmaps in: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="EyeTrax heatmaps for first 30s of video.")
    p.add_argument("--video", default="First Person POV_ Driving.mp4")
    p.add_argument("--output-dir", default="eyetrax_heatmaps_first30")
    p.add_argument("--webcam-id", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--interval-sec", type=float, default=5.0)
    p.add_argument("--calibration", choices=["none", "5p", "9p"], default="5p")
    p.add_argument("--ema-alpha", type=float, default=0.5)
    p.add_argument("--sigma", type=float, default=35.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
