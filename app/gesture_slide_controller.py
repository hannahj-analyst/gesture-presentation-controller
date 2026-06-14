import cv2
import time
import sys
import os
import json
import mediapipe as mp
import numpy as np
from functools import lru_cache
from pathlib import Path

# Allow importing from parent folder
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from utils.slide_controller import SlideController
from training.preprocess import normalize_landmarks


HAND_LANDMARKER_PATH = Path(__file__).resolve().parent.parent / "training" / "models" / "hand_landmarker.task"
GESTURE_MODEL_PATH = Path(__file__).resolve().parent.parent / "training" / "artifacts" / "keras_model.keras"
LABELS_PATH = Path(__file__).resolve().parent.parent / "training" / "artifacts" / "labels.json"
PREDICTION_COOLDOWN = 0.6
MIN_CONFIDENCE = 0.75

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

if not HAND_LANDMARKER_PATH.exists():
    raise FileNotFoundError(f"Hand Landmarker model not found: {HAND_LANDMARKER_PATH}")

if not GESTURE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Gesture model not found: {GESTURE_MODEL_PATH}")

if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Gesture labels not found: {LABELS_PATH}")


@lru_cache(maxsize=1)
def load_gesture_model():
    import tensorflow as tf

    return tf.keras.models.load_model(GESTURE_MODEL_PATH)


@lru_cache(maxsize=1)
def load_labels() -> list[str]:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def landmarks_to_features(hand_landmarks) -> np.ndarray:
    raw_landmarks = np.array(
        [
            (landmark.x, landmark.y, landmark.z)
            for landmark in hand_landmarks
        ],
        dtype=np.float32,
    )

    normalized = normalize_landmarks(raw_landmarks, scale=False)
    return normalized.reshape(1, -1)


def predict_gesture(hand_landmarks):
    model = load_gesture_model()
    labels = load_labels()
    features = landmarks_to_features(hand_landmarks)

    predictions = model.predict(features, verbose=0)[0]
    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])
    label = labels[class_index]

    return label, confidence

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

hands = HandLandmarker.create_from_options(options)

# Webcam setup
cap = cv2.VideoCapture(0)

# Optional: reduce webcam resolution for better speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Slide controller setup
controller = SlideController()

# Gesture prediction state
x_positions = []
last_swipe_time = 0
last_prediction_time = 0

cooldown = 0.8
threshold = 90
history_length = 6

last_action_text = "No action yet"
last_prediction_text = "No gesture yet"
last_prediction_confidence = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = hands.detect(mp_image)

    # Get current slide
    slide = controller.get_current_slide()
    slide = cv2.resize(slide, (900, 600))

    # Display slide number
    slide_text = (
        f"Slide {controller.get_slide_number()} / "
        f"{controller.get_total_slides()}"
    )

    cv2.putText(
        slide,
        slide_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        slide,
        f"Prediction: {last_prediction_text} ({last_prediction_confidence:.2f})",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        slide,
        f"Action: {last_action_text}",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:

            wrist = hand_landmarks.landmark[0]
            current_x = int(wrist.x * width)
            current_y = int(wrist.y * height)

            cv2.circle(
                frame,
                (current_x, current_y),
                10,
                (0, 255, 0),
                -1
            )

            try:
                predicted_label, confidence = predict_gesture(hand_landmarks)
            except Exception as exc:
                predicted_label = "unknown"
                confidence = 0.0
                last_action_text = f"Model error: {exc}"
            else:
                last_prediction_text = predicted_label
                last_prediction_confidence = confidence
                last_prediction_time = time.time()

                if confidence >= MIN_CONFIDENCE:
                    last_action_text = f"Predicted: {predicted_label}"

            # Store recent x positions
            x_positions.append(current_x)

            if len(x_positions) > history_length:
                x_positions.pop(0)

            # Detect swipe using movement over recent frames when the model is not stable
            if len(x_positions) == history_length and (
                time.time() - last_prediction_time > PREDICTION_COOLDOWN
                or last_prediction_confidence < MIN_CONFIDENCE
            ):
                movement = x_positions[-1] - x_positions[0]
                current_time = time.time()

                if current_time - last_swipe_time > cooldown:

                    if movement > threshold:
                        controller.next_slide()
                        last_action_text = "Swipe Right -> Next Slide"
                        print(last_action_text)
                        last_swipe_time = current_time
                        x_positions.clear()

                    elif movement < -threshold:
                        controller.previous_slide()
                        last_action_text = "Swipe Left -> Previous Slide"
                        print(last_action_text)
                        last_swipe_time = current_time
                        x_positions.clear()

    else:
        # Clear history if hand is not visible
        x_positions.clear()

    cv2.putText(
        frame,
        "Model gesture display | Swipe Right = Next | Swipe Left = Previous | q = Quit",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    webcam_display = cv2.resize(frame, (450, 300))

    cv2.imshow("Gesture Webcam", webcam_display)
    cv2.imshow("Gesture Controlled Slides", slide)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quit")
        break

cap.release()
cv2.destroyAllWindows()
hands.close()