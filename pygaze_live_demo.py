"""
Minimal live PyGaze demo.

Opens webcam, runs PyGaze face/gaze prediction, and overlays gaze vectors.
Press q to quit.
"""

import cv2
import numpy as np

from pygaze import PyGaze, PyGazeRenderer


def main():
    # Compatibility shim for older pygaze code using deprecated np.int.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]

    gaze = PyGaze(device="cpu")
    renderer = PyGazeRenderer()

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        faces = gaze.predict(frame)
        for face in faces:
            renderer.render(frame, face, draw_face_bbox=True, draw_gaze_vector=True)
            text = "LOOKING AT CAMERA" if gaze.look_at_camera(face) else "LOOKING AWAY"
            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if "AT" in text else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(frame, "PyGaze demo - q to quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("PyGaze Live Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
