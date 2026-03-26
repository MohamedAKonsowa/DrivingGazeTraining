"""
Full-screen driving video player with webcam gaze tracking.

What it does:
1) Plays the video in full-screen.
2) Tracks user's gaze from webcam using MediaPipe iris landmarks.
3) Stores gaze points in 5-second buckets.
4) Saves one heatmap image per bucket in JET color format.
5) Saves a preview grid image of all generated heatmaps.

Controls:
- f: toggle full-screen
- q or ESC: quit playback early
"""

import argparse
from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
import numpy as np


# Left eye landmarks
LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_IRIS_IDS = [468, 469, 470, 471]

# Right eye landmarks
RIGHT_EYE_LEFT_CORNER = 362
RIGHT_EYE_RIGHT_CORNER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_IRIS_IDS = [473, 474, 475, 476]


def to_pixel(landmark, width, height):
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def projected_ratio(point, start, end):
    axis = end - start
    denom = np.dot(axis, axis)
    if denom < 1e-6:
        return 0.5
    t = np.dot(point - start, axis) / denom
    return float(np.clip(t, 0.0, 1.0))


def iris_ratio_2d(landmarks, iris_ids, left_id, right_id, top_id, bottom_id, width, height):
    iris_center = np.mean([to_pixel(landmarks[i], width, height) for i in iris_ids], axis=0)
    left_corner = to_pixel(landmarks[left_id], width, height)
    right_corner = to_pixel(landmarks[right_id], width, height)
    top_pt = to_pixel(landmarks[top_id], width, height)
    bottom_pt = to_pixel(landmarks[bottom_id], width, height)
    x_ratio = projected_ratio(iris_center, left_corner, right_corner)
    y_ratio = projected_ratio(iris_center, top_pt, bottom_pt)
    return x_ratio, y_ratio


def smooth_pair(previous_xy, current_xy, alpha=0.25):
    return (previous_xy * (1.0 - alpha)) + (current_xy * alpha)


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


def save_preview_grid(images, out_path, cols=3):
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


class FaceLandmarkTracker:
    def __init__(self, model_asset_path="face_landmarker.task"):
        self.mode = None
        self.face_mesh = None
        self.landmarker = None

        if hasattr(mp, "solutions"):
            self.mode = "solutions"
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return

        # MediaPipe >= 0.10 on Python 3.14 exposes tasks API.
        self.mode = "tasks"
        model_path = Path(model_asset_path).resolve()
        if not model_path.exists():
            url = (
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task"
            )
            print(f"Downloading face landmarker model to: {model_path}")
            urllib.request.urlretrieve(url, str(model_path))

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, bgr_frame, timestamp_ms):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if self.mode == "solutions":
            result = self.face_mesh.process(rgb)
            if result.multi_face_landmarks:
                return result.multi_face_landmarks[0].landmark
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)
        if result.face_landmarks:
            return result.face_landmarks[0]
        return None

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.landmarker is not None:
            self.landmarker.close()


def run(args):
    video_path = Path(args.video).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Webcam access on macOS can be flaky right after permission prompts.
    # Try a few openings/backends before failing.
    webcam_cap = None
    webcam_candidates = [args.webcam_id, 0, 1]
    webcam_candidates = list(dict.fromkeys(webcam_candidates))
    for cam_id in webcam_candidates:
        cap_try = cv2.VideoCapture(cam_id, cv2.CAP_AVFOUNDATION)
        if cap_try.isOpened():
            ok_read, _ = cap_try.read()
            if ok_read:
                webcam_cap = cap_try
                print(f"Using webcam id={cam_id}")
                break
        cap_try.release()
    if webcam_cap is None:
        raise RuntimeError(
            f"Could not open webcam. Tried ids: {webcam_candidates}. "
            "Close other apps using camera and try again."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    segment_starts = np.arange(0.0, duration_sec, args.interval_sec).tolist()
    segment_ends = [min(s + args.interval_sec, duration_sec) for s in segment_starts]
    segment_points = [[] for _ in segment_starts]

    window_name = "Fullscreen Gaze Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    tracker = FaceLandmarkTracker(model_asset_path=args.landmarker_model)
    smooth_xy = np.array([0.5, 0.5], dtype=np.float32)
    frame_idx = 0
    fullscreen = args.fullscreen

    while True:
        ok_video, frame = cap.read()
        if not ok_video:
            break

        ok_webcam, webcam_frame = webcam_cap.read()
        gaze_xy = None
        if ok_webcam:
            webcam_frame = cv2.flip(webcam_frame, 1)
            wh, ww = webcam_frame.shape[:2]
            timestamp_ms = int((frame_idx / fps) * 1000.0)
            landmarks = tracker.detect(webcam_frame, timestamp_ms=timestamp_ms)
            if landmarks is not None:
                lx, ly = iris_ratio_2d(
                    landmarks,
                    LEFT_IRIS_IDS,
                    LEFT_EYE_LEFT_CORNER,
                    LEFT_EYE_RIGHT_CORNER,
                    LEFT_EYE_TOP,
                    LEFT_EYE_BOTTOM,
                    ww,
                    wh,
                )
                rx, ry = iris_ratio_2d(
                    landmarks,
                    RIGHT_IRIS_IDS,
                    RIGHT_EYE_LEFT_CORNER,
                    RIGHT_EYE_RIGHT_CORNER,
                    RIGHT_EYE_TOP,
                    RIGHT_EYE_BOTTOM,
                    ww,
                    wh,
                )
                cur_xy = np.array([(lx + rx) * 0.5, (ly + ry) * 0.5], dtype=np.float32)
                smooth_xy = smooth_pair(smooth_xy, cur_xy, alpha=0.25)
                gaze_xy = smooth_xy

        h, w = frame.shape[:2]
        t_sec = frame_idx / fps
        seg_idx = min(int(t_sec // args.interval_sec), len(segment_points) - 1)

        if gaze_xy is not None:
            gx = float(gaze_xy[0]) * w
            gy = float(gaze_xy[1]) * h
            segment_points[seg_idx].append((gx, gy))
            cv2.circle(frame, (int(gx), int(gy)), 8, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"Time {t_sec:.1f}s  Segment {seg_idx + 1}/{len(segment_points)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "f=fullscreen  q/esc=quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(max(1, int(1000.0 / fps))) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )

        frame_idx += 1

    cap.release()
    webcam_cap.release()
    tracker.close()
    cv2.destroyAllWindows()

    # Re-open video once for output size.
    cap = cv2.VideoCapture(str(video_path))
    ok_first, first_frame = cap.read()
    cap.release()
    if not ok_first:
        raise RuntimeError("Could not read first frame for heatmap sizing.")
    h, w = first_frame.shape[:2]

    saved_images = []
    csv_lines = ["segment,start_s,end_s,num_points,output_file"]
    for i, points in enumerate(segment_points):
        heat = build_heatmap(points, w, h, sigma=args.sigma)
        color = heat_to_color(heat)
        out_name = f"user_heatmap_t{int(segment_starts[i]):05d}s.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), color)
        saved_images.append(color)
        csv_lines.append(
            f"{i + 1},{segment_starts[i]:.2f},{segment_ends[i]:.2f},{len(points)},{out_name}"
        )

    (output_dir / "user_heatmaps_report.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    save_preview_grid(saved_images, output_dir / "user_heatmaps_preview_grid.png", cols=3)

    print(f"Saved {len(saved_images)} heatmaps to: {output_dir}")
    print(f"Preview grid: {output_dir / 'user_heatmaps_preview_grid.png'}")
    print(f"Report CSV: {output_dir / 'user_heatmaps_report.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Full-screen video gaze tracker and heatmap exporter.")
    parser.add_argument("--video", default="First Person POV_ Driving.mp4", help="Input video path.")
    parser.add_argument("--output-dir", default="user_gaze_heatmaps_5s", help="Output directory.")
    parser.add_argument("--interval-sec", type=float, default=5.0, help="Heatmap interval in seconds.")
    parser.add_argument("--webcam-id", type=int, default=0, help="Webcam camera index.")
    parser.add_argument("--sigma", type=float, default=35.0, help="Gaussian blur sigma for heatmaps.")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode.")
    parser.add_argument(
        "--landmarker-model",
        default="face_landmarker.task",
        help="Path to MediaPipe Face Landmarker .task model file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
