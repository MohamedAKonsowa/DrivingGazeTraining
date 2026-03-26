"""
Simple webcam gaze estimation demo.

This script uses MediaPipe Face Mesh with iris landmarks to estimate where
the user is looking within the webcam frame.

Controls:
- Press 'a' while looking LEFT to calibrate left.
- Press 's' while looking CENTER to calibrate center.
- Press 'd' while looking RIGHT to calibrate right.
- Press 'r' to reset calibration.
- Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np


# Left eye landmarks
LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133
LEFT_IRIS_IDS = [468, 469, 470, 471]

# Right eye landmarks
RIGHT_EYE_LEFT_CORNER = 362
RIGHT_EYE_RIGHT_CORNER = 263
RIGHT_IRIS_IDS = [473, 474, 475, 476]


def to_pixel(landmark, width, height):
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def iris_ratio(landmarks, iris_ids, left_corner_id, right_corner_id, width, height):
    iris_points = [to_pixel(landmarks[i], width, height) for i in iris_ids]
    iris_center = np.mean(iris_points, axis=0)

    left_corner = to_pixel(landmarks[left_corner_id], width, height)
    right_corner = to_pixel(landmarks[right_corner_id], width, height)

    eye_width = np.linalg.norm(right_corner - left_corner)
    if eye_width < 1e-6:
        return 0.5, iris_center

    # Project iris center on the eye corner line to get a normalized horizontal ratio.
    eye_vec = right_corner - left_corner
    proj = np.dot(iris_center - left_corner, eye_vec) / np.dot(eye_vec, eye_vec)
    return float(np.clip(proj, 0.0, 1.0)), iris_center


def smooth_value(previous, current, alpha=0.25):
    return previous * (1.0 - alpha) + current * alpha


def map_with_calibration(raw_ratio, calib):
    """
    Piecewise map raw gaze ratio using 3-point calibration.
    Returns a normalized value in [0, 1].
    """
    left = calib["left"]
    center = calib["center"]
    right = calib["right"]

    if left is None or center is None or right is None:
        return raw_ratio

    # Ensure monotonic ordering in case user calibration is noisy.
    points = sorted([left, center, right])
    left, center, right = points

    if right - left < 1e-6:
        return raw_ratio

    if raw_ratio <= center:
        denom = max(center - left, 1e-6)
        t = (raw_ratio - left) / denom
        mapped = 0.5 * t
    else:
        denom = max(right - center, 1e-6)
        t = (raw_ratio - center) / denom
        mapped = 0.5 + 0.5 * t

    return float(np.clip(mapped, 0.0, 1.0))


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    drawing = mp.solutions.drawing_utils
    draw_spec = drawing.DrawingSpec(color=(120, 220, 120), thickness=1, circle_radius=1)
    calib = {"left": None, "center": None, "right": None}
    smoothed_raw_ratio = 0.5

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = face_mesh.process(rgb)
            rgb.flags.writeable = True

            latest_raw_ratio = None

            if result.multi_face_landmarks:
                face = result.multi_face_landmarks[0]
                landmarks = face.landmark

                left_ratio, left_iris_center = iris_ratio(
                    landmarks,
                    LEFT_IRIS_IDS,
                    LEFT_EYE_LEFT_CORNER,
                    LEFT_EYE_RIGHT_CORNER,
                    w,
                    h,
                )
                right_ratio, right_iris_center = iris_ratio(
                    landmarks,
                    RIGHT_IRIS_IDS,
                    RIGHT_EYE_LEFT_CORNER,
                    RIGHT_EYE_RIGHT_CORNER,
                    w,
                    h,
                )

                # Average both eyes. 0=looking left, 1=looking right.
                raw_gaze_x_ratio = (left_ratio + right_ratio) * 0.5
                smoothed_raw_ratio = smooth_value(smoothed_raw_ratio, raw_gaze_x_ratio, alpha=0.25)
                latest_raw_ratio = smoothed_raw_ratio
                gaze_x_ratio = map_with_calibration(smoothed_raw_ratio, calib)

                # Convert horizontal ratio to a visible point in frame.
                gaze_x = int(gaze_x_ratio * w)
                gaze_y = int(h * 0.5)

                # Draw iris centers and estimated gaze point.
                cv2.circle(frame, tuple(left_iris_center.astype(int)), 3, (255, 255, 0), -1)
                cv2.circle(frame, tuple(right_iris_center.astype(int)), 3, (255, 255, 0), -1)
                cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"Raw: {smoothed_raw_ratio:.2f}  Cal: {gaze_x_ratio:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Calibrate: a=left s=center d=right  r=reset  q=quit",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                left_txt = "-" if calib["left"] is None else f"{calib['left']:.2f}"
                center_txt = "-" if calib["center"] is None else f"{calib['center']:.2f}"
                right_txt = "-" if calib["right"] is None else f"{calib['right']:.2f}"
                status = f"L:{left_txt} C:{center_txt} R:{right_txt}"
                cv2.putText(
                    frame,
                    status,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (50, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

                drawing.draw_landmarks(
                    frame,
                    face,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=draw_spec,
                    connection_drawing_spec=draw_spec,
                )
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Simple Gaze Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                calib = {"left": None, "center": None, "right": None}
            if latest_raw_ratio is not None:
                if key == ord("a"):
                    calib["left"] = latest_raw_ratio
                elif key == ord("s"):
                    calib["center"] = latest_raw_ratio
                elif key == ord("d"):
                    calib["right"] = latest_raw_ratio

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
