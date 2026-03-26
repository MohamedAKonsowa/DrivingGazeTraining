"""
Quick gaze debug viewer.

Shows webcam feed with a red dot indicating estimated gaze position.
This is a simple sanity-check tool to verify eye tracking is working.

Controls:
- q or ESC: quit
"""

from pathlib import Path
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np


# Eye landmark indices (MediaPipe FaceMesh / FaceLandmarker 478-point topology)
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


def eye_ratio_2d(landmarks, iris_ids, left_id, right_id, top_id, bottom_id, width, height):
    iris_center = np.mean([to_pixel(landmarks[i], width, height) for i in iris_ids], axis=0)
    left_corner = to_pixel(landmarks[left_id], width, height)
    right_corner = to_pixel(landmarks[right_id], width, height)
    top_pt = to_pixel(landmarks[top_id], width, height)
    bottom_pt = to_pixel(landmarks[bottom_id], width, height)
    x_ratio = projected_ratio(iris_center, left_corner, right_corner)
    y_ratio = projected_ratio(iris_center, top_pt, bottom_pt)
    return x_ratio, y_ratio


def map_axis(value, left, center, right):
    if left is None or center is None or right is None:
        return value
    pts = sorted([left, center, right])
    left, center, right = pts
    if right - left < 1e-6:
        return value
    if value <= center:
        t = (value - left) / max(center - left, 1e-6)
        mapped = 0.5 * t
    else:
        t = (value - center) / max(right - center, 1e-6)
        mapped = 0.5 + 0.5 * t
    return float(np.clip(mapped, 0.0, 1.0))


def axis_value(point, axis_idx):
    if point is None:
        return None
    return float(point[axis_idx])


class Tracker:
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

        self.mode = "tasks"
        model_path = Path(model_asset_path).resolve()
        if not model_path.exists():
            url = (
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task"
            )
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
            out = self.face_mesh.process(rgb)
            if out.multi_face_landmarks:
                return out.multi_face_landmarks[0].landmark
            return None

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        out = self.landmarker.detect_for_video(mp_img, timestamp_ms=timestamp_ms)
        if out.face_landmarks:
            return out.face_landmarks[0]
        return None

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.landmarker is not None:
            self.landmarker.close()


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    tracker = Tracker()
    smooth_xy = np.array([0.5, 0.5], dtype=np.float32)
    calib = {
        "left": None,
        "center": None,
        "right": None,
        "up": None,
        "down": None,
    }
    gain_x = 1.0
    gain_y = 1.0
    auto_calibrating = False
    # Edge-friendly targets to force full-screen usable mapping.
    calib_targets = [
        ("center", 0.50, 0.50),
        ("left", 0.03, 0.50),
        ("right", 0.97, 0.50),
        ("up", 0.50, 0.03),
        ("down", 0.50, 0.97),
    ]
    calib_step_idx = 0
    calib_step_start = 0.0
    calib_duration_sec = 5.0
    calib_samples = []
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts_ms = int((frame_idx / fps) * 1000.0)

        landmarks = tracker.detect(frame, ts_ms)
        current_raw_xy = None
        if landmarks is not None:
            lx, ly = eye_ratio_2d(
                landmarks,
                LEFT_IRIS_IDS,
                LEFT_EYE_LEFT_CORNER,
                LEFT_EYE_RIGHT_CORNER,
                LEFT_EYE_TOP,
                LEFT_EYE_BOTTOM,
                w,
                h,
            )
            rx, ry = eye_ratio_2d(
                landmarks,
                RIGHT_IRIS_IDS,
                RIGHT_EYE_LEFT_CORNER,
                RIGHT_EYE_RIGHT_CORNER,
                RIGHT_EYE_TOP,
                RIGHT_EYE_BOTTOM,
                w,
                h,
            )
            cur_xy = np.array([(lx + rx) * 0.5, (ly + ry) * 0.5], dtype=np.float32)
            smooth_xy = smooth_xy * 0.75 + cur_xy * 0.25
            current_raw_xy = (float(smooth_xy[0]), float(smooth_xy[1]))

            mapped_x = map_axis(
                smooth_xy[0],
                axis_value(calib["left"], 0),
                axis_value(calib["center"], 0),
                axis_value(calib["right"], 0),
            )
            mapped_y = map_axis(
                smooth_xy[1],
                axis_value(calib["up"], 1),
                axis_value(calib["center"], 1),
                axis_value(calib["down"], 1),
            )
            # Gain around center to improve magnitude response.
            mapped_x = np.clip(0.5 + (mapped_x - 0.5) * gain_x, 0.0, 1.0)
            mapped_y = np.clip(0.5 + (mapped_y - 0.5) * gain_y, 0.0, 1.0)

            gx = int(mapped_x * w)
            gy = int(mapped_y * h)
            cv2.circle(frame, (gx, gy), 12, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"raw=({smooth_xy[0]:.2f},{smooth_xy[1]:.2f}) mapped=({mapped_x:.2f},{mapped_y:.2f})",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            left_txt = "-" if calib["left"] is None else f"{calib['left'][0]:.2f}"
            center_txt = "-" if calib["center"] is None else f"{calib['center'][0]:.2f}/{calib['center'][1]:.2f}"
            right_txt = "-" if calib["right"] is None else f"{calib['right'][0]:.2f}"
            up_txt = "-" if calib["up"] is None else f"{calib['up'][1]:.2f}"
            down_txt = "-" if calib["down"] is None else f"{calib['down'][1]:.2f}"
            cv2.putText(
                frame,
                f"Calib X L:{left_txt} C:{center_txt} R:{right_txt} | Y U:{up_txt} D:{down_txt}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (255, 220, 80),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if auto_calibrating:
            target_name, target_x, target_y = calib_targets[calib_step_idx]
            target_px = int(target_x * w)
            target_py = int(target_y * h)
            cv2.circle(frame, (target_px, target_py), 16, (0, 255, 255), -1)
            cv2.putText(
                frame,
                f"Look at {target_name.upper()} dot",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            elapsed = time.time() - calib_step_start
            remaining = max(0.0, calib_duration_sec - elapsed)
            cv2.putText(
                frame,
                f"Capturing in: {remaining:.1f}s",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if current_raw_xy is not None:
                calib_samples.append(current_raw_xy)

            if elapsed >= calib_duration_sec:
                if calib_samples:
                    arr = np.asarray(calib_samples, dtype=np.float32)
                    calib[target_name] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
                calib_samples = []
                calib_step_idx += 1
                if calib_step_idx >= len(calib_targets):
                    auto_calibrating = False
                    # Auto gain from calibrated ranges for better edge reach.
                    if calib["left"] and calib["center"] and calib["right"]:
                        span_l = abs(calib["center"][0] - calib["left"][0])
                        span_r = abs(calib["right"][0] - calib["center"][0])
                        span = max((span_l + span_r) * 0.5, 1e-3)
                        gain_x = float(np.clip(0.5 / span, 0.9, 2.2))
                    if calib["up"] and calib["center"] and calib["down"]:
                        span_u = abs(calib["center"][1] - calib["up"][1])
                        span_d = abs(calib["down"][1] - calib["center"][1])
                        span = max((span_u + span_d) * 0.5, 1e-3)
                        gain_y = float(np.clip(0.5 / span, 0.9, 2.2))
                    cv2.putText(
                        frame,
                        f"Auto-calibration complete (gain x={gain_x:.2f}, y={gain_y:.2f})",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    calib_step_start = time.time()

        cv2.putText(
            frame,
            "c:auto(5 points)  a/s/d:manual-x  w/x:manual-y  r:reset  q/ESC:quit",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Quick Gaze Debug", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            calib = {"left": None, "center": None, "right": None, "up": None, "down": None}
            gain_x = 1.0
            gain_y = 1.0
            auto_calibrating = False
        if key == ord("c"):
            auto_calibrating = True
            calib_step_idx = 0
            calib_step_start = time.time()
            calib_samples = []
            calib = {"left": None, "center": None, "right": None, "up": None, "down": None}
            gain_x = 1.0
            gain_y = 1.0
        if key == ord("a"):
            calib["left"] = (float(smooth_xy[0]), float(smooth_xy[1]))
        elif key == ord("s"):
            calib["center"] = (float(smooth_xy[0]), float(smooth_xy[1]))
        elif key == ord("d"):
            calib["right"] = (float(smooth_xy[0]), float(smooth_xy[1]))
        elif key == ord("w"):
            calib["up"] = (float(smooth_xy[0]), float(smooth_xy[1]))
        elif key == ord("x"):
            calib["down"] = (float(smooth_xy[0]), float(smooth_xy[1]))
        frame_idx += 1

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
