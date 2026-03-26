"""
Compare user gaze heatmaps against DR(eye)VE reference heatmaps.

Pipeline:
1) Provide a driving video.
2) Provide DR(eye)VE reference saliency as either:
   - a saliency video (one heatmap frame per video frame), or
   - a directory of heatmap images.
3) User watches the driving video while webcam gaze is tracked.
4) Every N seconds (default 5), a user heatmap is created and compared
   to the reference heatmap for that same segment.

Example:
python drive_heatmap_compare.py \
  --drive-video data/drive.mp4 \
  --reference-heatmap-video data/dreyeve_saliency.mp4 \
  --interval-sec 5 \
  --output-dir outputs
"""

import argparse
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# Landmarks for 2D gaze ratio estimation.
LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_IRIS_IDS = [468, 469, 470, 471]

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


def iris_ratio_2d(
    landmarks,
    iris_ids,
    left_corner_id,
    right_corner_id,
    top_id,
    bottom_id,
    width,
    height,
):
    iris_points = [to_pixel(landmarks[i], width, height) for i in iris_ids]
    iris_center = np.mean(iris_points, axis=0)

    left_corner = to_pixel(landmarks[left_corner_id], width, height)
    right_corner = to_pixel(landmarks[right_corner_id], width, height)
    top_pt = to_pixel(landmarks[top_id], width, height)
    bottom_pt = to_pixel(landmarks[bottom_id], width, height)

    x_ratio = projected_ratio(iris_center, left_corner, right_corner)
    y_ratio = projected_ratio(iris_center, top_pt, bottom_pt)
    return x_ratio, y_ratio


def smooth_pair(prev_xy, cur_xy, alpha=0.25):
    return (prev_xy * (1.0 - alpha)) + (cur_xy * alpha)


def list_image_files(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in exts]
    return files


def sample_reference_maps_from_video(video_path, segment_centers_sec):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open reference heatmap video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    maps = []
    for t in segment_centers_sec:
        frame_idx = int(max(t, 0.0) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        maps.append(frame if ok else None)
    cap.release()
    return maps


def sample_reference_maps_from_dir(folder, num_segments):
    files = list_image_files(folder)
    if not files:
        raise RuntimeError(f"No images found in reference heatmap dir: {folder}")

    if len(files) == num_segments:
        idxs = list(range(num_segments))
    else:
        # Spread available files across segments.
        idxs = np.linspace(0, len(files) - 1, num_segments).round().astype(int).tolist()

    maps = []
    for idx in idxs:
        frame = cv2.imread(str(files[idx]))
        maps.append(frame)
    return maps


def build_heatmap(points_xy, width, height, blur_ksize=0, blur_sigma=35.0):
    heat = np.zeros((height, width), dtype=np.float32)
    for x, y in points_xy:
        xi = int(np.clip(x, 0, width - 1))
        yi = int(np.clip(y, 0, height - 1))
        heat[yi, xi] += 1.0

    if heat.max() > 0:
        heat = cv2.GaussianBlur(heat, (blur_ksize, blur_ksize), blur_sigma)
        if heat.max() > 0:
            heat /= heat.max()
    return heat


def normalize_map(map_img, width, height):
    if map_img is None:
        return np.zeros((height, width), dtype=np.float32)
    gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    norm = gray.astype(np.float32) / 255.0
    return norm


def heat_to_color(heat):
    img = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)


def corr_score(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def run_compare(args):
    drive_cap = cv2.VideoCapture(args.drive_video)
    if not drive_cap.isOpened():
        raise RuntimeError(f"Could not open driving video: {args.drive_video}")

    fps = drive_cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(drive_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    if duration_sec <= 0:
        raise RuntimeError("Could not read duration from driving video.")

    segment_starts = np.arange(0.0, duration_sec, args.interval_sec).tolist()
    segment_ends = [min(s + args.interval_sec, duration_sec) for s in segment_starts]
    segment_centers = [(s + e) * 0.5 for s, e in zip(segment_starts, segment_ends)]
    num_segments = len(segment_starts)

    if args.reference_heatmap_video:
        reference_maps_raw = sample_reference_maps_from_video(args.reference_heatmap_video, segment_centers)
    elif args.reference_heatmap_dir:
        reference_maps_raw = sample_reference_maps_from_dir(args.reference_heatmap_dir, num_segments)
    else:
        raise RuntimeError("Provide --reference-heatmap-video or --reference-heatmap-dir.")

    os.makedirs(args.output_dir, exist_ok=True)
    maps_dir = Path(args.output_dir) / "segment_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    webcam_cap = cv2.VideoCapture(args.webcam_id)
    if not webcam_cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    segment_points = [[] for _ in range(num_segments)]
    scores = []
    smooth_xy = np.array([0.5, 0.5], dtype=np.float32)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        frame_idx = 0
        while True:
            ok, drive_frame = drive_cap.read()
            if not ok:
                break

            ok_webcam, webcam_frame = webcam_cap.read()
            gaze_xy = None
            if ok_webcam:
                webcam_frame = cv2.flip(webcam_frame, 1)
                wh, ww = webcam_frame.shape[:2]
                rgb = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = face_mesh.process(rgb)
                rgb.flags.writeable = True

                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
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
                    gaze_xy = smooth_xy.copy()

            t_sec = frame_idx / fps
            seg_idx = int(min(t_sec // args.interval_sec, num_segments - 1))

            dh, dw = drive_frame.shape[:2]
            if gaze_xy is not None:
                gx = float(gaze_xy[0]) * dw
                gy = float(gaze_xy[1]) * dh
                segment_points[seg_idx].append((gx, gy))
                cv2.circle(drive_frame, (int(gx), int(gy)), 8, (0, 0, 255), -1)

            ref_small = normalize_map(reference_maps_raw[seg_idx], dw, dh)
            ref_color = heat_to_color(ref_small)
            overlay = cv2.addWeighted(drive_frame, 0.80, ref_color, 0.20, 0.0)

            cv2.putText(
                overlay,
                f"Segment {seg_idx + 1}/{num_segments}  Time {t_sec:.1f}s",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                "Watching mode: ESC to stop",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Drive + Reference Heatmap Overlay", overlay)
            key = cv2.waitKey(max(1, int(1000.0 / fps))) & 0xFF
            if key == 27:
                break
            frame_idx += 1

    drive_cap.release()
    webcam_cap.release()
    cv2.destroyAllWindows()

    # Build and compare per-segment maps.
    drive_cap = cv2.VideoCapture(args.drive_video)
    ok, first_frame = drive_cap.read()
    drive_cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame for output sizing.")

    h, w = first_frame.shape[:2]
    report_lines = ["segment,start_s,end_s,user_points,corr_score"]
    for i in range(num_segments):
        user_heat = build_heatmap(segment_points[i], w, h, blur_sigma=args.user_heat_sigma)
        ref_heat = normalize_map(reference_maps_raw[i], w, h)
        score = corr_score(user_heat, ref_heat)
        scores.append(score)

        user_color = heat_to_color(user_heat)
        ref_color = heat_to_color(ref_heat)
        comp = np.hstack([user_color, ref_color])

        cv2.putText(
            comp,
            f"Segment {i + 1}  score={score:.3f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            comp,
            "Left: User heatmap | Right: DR(eye)VE reference",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_img = maps_dir / f"segment_{i + 1:03d}.png"
        cv2.imwrite(str(out_img), comp)
        report_lines.append(
            f"{i + 1},{segment_starts[i]:.2f},{segment_ends[i]:.2f},{len(segment_points[i])},{score:.6f}"
        )

    mean_score = float(np.mean(scores)) if scores else 0.0
    report_lines.append(f"mean_score,,,,{mean_score:.6f}")
    report_path = Path(args.output_dir) / "comparison_report.csv"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved segment comparison maps to: {maps_dir}")
    print(f"Saved report: {report_path}")
    print(f"Mean correlation score: {mean_score:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare user gaze vs DR(eye)VE heatmaps.")
    parser.add_argument("--drive-video", required=True, help="Path to driving video.")
    parser.add_argument(
        "--reference-heatmap-video",
        default=None,
        help="Path to DR(eye)VE saliency video (optional).",
    )
    parser.add_argument(
        "--reference-heatmap-dir",
        default=None,
        help="Path to folder of DR(eye)VE heatmap images (optional).",
    )
    parser.add_argument("--interval-sec", type=float, default=5.0, help="Segment duration in seconds.")
    parser.add_argument("--webcam-id", type=int, default=0, help="Webcam index.")
    parser.add_argument("--user-heat-sigma", type=float, default=35.0, help="User heatmap blur sigma.")
    parser.add_argument("--output-dir", default="outputs_compare", help="Output folder.")
    args = parser.parse_args()

    if bool(args.reference_heatmap_video) == bool(args.reference_heatmap_dir):
        raise ValueError("Provide exactly one of --reference-heatmap-video or --reference-heatmap-dir.")
    return args


if __name__ == "__main__":
    run_compare(parse_args())
