"""
demo_pointer_drawing.py
-----------------------
Standalone demo — runs from the terminal with just a webcam.
No Streamlit or other team modules required.

Controls (keyboard):
  P  →  force POINTER gesture
  D  →  force DRAWING gesture
  C  →  force CLEAR  gesture (then auto-returns to idle)
  Q  →  quit

Usage:
  python demo_pointer_drawing.py
"""

import os
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pointer_drawing import pointer_drawing_overlay, clear_canvas

GESTURE_CYCLE = ["", "POINTER", "DRAWING", "CLEAR"]


def run_demo() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (device 0).")

    forced_gesture = ""                         # override via keyboard

    model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.VIDEO
    )

    with vision.HandLandmarker.create_from_options(options) as detector:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)          # mirror feels more natural
            h, w  = frame.shape[:2]

            timestamp_ms = int(time.time() * 1000)

            # ── MediaPipe detection ──────────────────────────────────────── #
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results   = detector.detect_for_video(mp_image, timestamp_ms)
            
            landmarks = (
                results.hand_landmarks[0]
                if results.hand_landmarks
                else None
            )

            # Draw skeleton (optional — remove for cleaner demo)
            if landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame, landmarks, vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    vision.drawing_utils.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=2),
                    vision.drawing_utils.DrawingSpec(color=(60, 60, 60), thickness=1),
                )

            # ── Apply pointer / drawing overlay ─────────────────────────── #
            # In the real app `forced_gesture` comes from the gesture module.
            gesture = forced_gesture
            if gesture == "CLEAR":
                forced_gesture = ""             # one-shot clear

            frame = pointer_drawing_overlay(frame, landmarks, gesture)

            # ── HUD: keyboard hint ───────────────────────────────────────── #
            hints = "[P] Laser  [D] Draw  [C] Clear  [Q] Quit"
            cv2.putText(frame, hints, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            mode_txt = f"Mode: {forced_gesture if forced_gesture else 'idle'}"
            cv2.putText(frame, mode_txt, (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 50), 2, cv2.LINE_AA)

            cv2.imshow("Pointer & Drawing Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                forced_gesture = "POINTER"
            elif key == ord("d"):
                forced_gesture = "DRAWING"
            elif key == ord("c"):
                forced_gesture = "CLEAR"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
