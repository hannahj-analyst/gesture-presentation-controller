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
from pointer_drawing import PointerDrawingState, pointer_drawing_overlay, clear_canvas
from training.preprocess import normalize_landmarks


HAND_LANDMARKER_PATH = Path(__file__).resolve().parent.parent / "training" / "models" / "hand_landmarker.task"
GESTURE_MODEL_PATH = Path(__file__).resolve().parent.parent / "training" / "artifacts" / "keras_model.keras"
LABELS_PATH = Path(__file__).resolve().parent.parent / "training" / "artifacts" / "labels.json"
PREDICTION_COOLDOWN = 0.6
SWIPE_COOLDOWN = 1.0
GESTURE_MIN_CONFIDENCE = 0.55
MODE_HOLD_SECONDS = 3.0
SWIPE_TRIGGER_CONFIDENCE = 0.1
SWIPE_HISTORY_LENGTH = 8
SWIPE_MIN_MOVEMENT = 60
SWIPE_MIN_STEP = 6
SWIPE_DIRECTION_RATIO = 0.1

MODE_BY_LABEL = {
    "one": "POINTER",
    "peace": "DRAWING",
    "stop": "CLEAR",
}

MODE_EXIT_GESTURE_BY_MODE = {
    "POINTER": "one",
    "DRAWING": "peace",
}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Used for drawing the full hand skeleton. This is the Tasks API equivalent
# of the legacy mp.solutions.hands.HAND_CONNECTIONS - a list of Connection
# namedtuples with .start / .end landmark indices.
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

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


def detect_swipe(x_history: list[int]) -> str | None:
    if len(x_history) < SWIPE_HISTORY_LENGTH:
        return None

    movement = x_history[-1] - x_history[0]
    if abs(movement) < SWIPE_MIN_MOVEMENT:
        return None

    deltas = np.diff(np.array(x_history))
    if movement > 0:
        consistent_steps = np.count_nonzero(deltas > SWIPE_MIN_STEP)
        swipe_direction = "right"
    else:
        consistent_steps = np.count_nonzero(deltas < -SWIPE_MIN_STEP)
        swipe_direction = "left"

    if consistent_steps / len(deltas) < SWIPE_DIRECTION_RATIO:
        return None

    return swipe_direction


def draw_hand_landmarks(frame, hand_landmarks, width, height):
    """Draw the full hand skeleton (connections + joints)."""
    for connection in HAND_CONNECTIONS:
        start = hand_landmarks[connection.start]
        end = hand_landmarks[connection.end]
        cv2.line(
            frame,
            (int(start.x * width), int(start.y * height)),
            (int(end.x * width), int(end.y * height)),
            (255, 255, 255),
            2,
        )

    for landmark in hand_landmarks:
        cv2.circle(
            frame,
            (int(landmark.x * width), int(landmark.y * height)),
            4,
            (0, 255, 0),
            -1,
        )


# NOTE: running_mode is VIDEO (not IMAGE). VIDEO mode enables MediaPipe's
# inter-frame tracker, which is both faster and far less jittery for a live
# webcam feed. It also requires a monotonically increasing timestamp to be
# passed to detect_for_video() on every call (handled in the main loop below).
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
    running_mode=RunningMode.VIDEO,
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
slide_annotation_states: dict[int, PointerDrawingState] = {}

# Gesture prediction state
x_positions = []
last_swipe_time = 0
last_prediction_time = 0
active_mode = "NAVIGATION"
held_label = None
held_label_start = 0.0
held_label_fired = False

history_length = SWIPE_HISTORY_LENGTH

last_action_text = "No action yet"
last_prediction_text = "No gesture yet"
last_prediction_confidence = 0.0


def get_slide_state(slide_index: int) -> PointerDrawingState:
    if slide_index not in slide_annotation_states:
        slide_annotation_states[slide_index] = PointerDrawingState()
    return slide_annotation_states[slide_index]

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam")
        break

    current_time = time.time()
    frame_timestamp_ms = int(current_time * 1000)

    # --- Swipe cooldown status (used for on-screen display below) ---
    time_since_swipe = current_time - last_swipe_time
    swipe_remaining = SWIPE_COOLDOWN - time_since_swipe
    swipe_on_cooldown = swipe_remaining > 0

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = hands.detect_for_video(mp_image, frame_timestamp_ms)

    # Get current slide
    slide = controller.get_current_slide()
    slide = cv2.resize(slide, (900, 600))
    current_slide_index = controller.current_slide_index
    slide_state = get_slide_state(current_slide_index)

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

    if active_mode != "NAVIGATION":
        exit_gesture = MODE_EXIT_GESTURE_BY_MODE.get(active_mode, "")
        swipe_status_text = f"Swipe: LOCKED ({active_mode}) | hold {exit_gesture} 3s to exit"
        swipe_status_color = (0, 140, 255)  # orange (BGR)
    elif swipe_on_cooldown:
        swipe_status_text = f"Swipe: COOLDOWN ({swipe_remaining:.1f}s)"
        swipe_status_color = (0, 0, 255)  # red (BGR)
    else:
        swipe_status_text = "Swipe: READY"
        swipe_status_color = (0, 255, 0)  # green (BGR)

    cv2.putText(
        slide,
        swipe_status_text,
        (30, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        swipe_status_color,
        2
    )

    cv2.putText(
        slide,
        f"Mode: {active_mode}",
        (30, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 120, 0),
        2
    )

    hand_landmarks = results.hand_landmarks[0] if results.hand_landmarks else None

    if hand_landmarks:
        wrist = hand_landmarks[0]
        current_x = int(wrist.x * width)
        x_positions.append(current_x)

        if len(x_positions) > history_length:
            x_positions.pop(0)

        # --- Gesture prediction: throttled via PREDICTION_COOLDOWN ---
        # model.predict() has real per-call overhead, so we only run it
        # periodically instead of every single frame. On skipped frames
        # we reuse the last known label/confidence so downstream logic
        # does not flicker.
        if current_time - last_prediction_time > PREDICTION_COOLDOWN:
            try:
                predicted_label, confidence = predict_gesture(hand_landmarks)
            except Exception as exc:
                last_action_text = f"Model error: {exc}"
                predicted_label = last_prediction_text
                confidence = last_prediction_confidence
            else:
                last_prediction_text = predicted_label
                last_prediction_confidence = confidence
            finally:
                last_prediction_time = current_time
        else:
            predicted_label = last_prediction_text
            confidence = last_prediction_confidence

        # --- Swipe detection: only active in NAVIGATION mode ---
        swipe_triggered = False

        if active_mode == "NAVIGATION" and len(x_positions) == history_length and current_time - last_swipe_time > SWIPE_COOLDOWN:
            swipe_direction = detect_swipe(x_positions)

            if swipe_direction == "right":
                controller.next_slide()
                last_action_text = "Swipe Right -> Next Slide"
                print(last_action_text)
                last_swipe_time = current_time
                x_positions.clear()
                swipe_triggered = True

            elif swipe_direction == "left":
                controller.previous_slide()
                last_action_text = "Swipe Left -> Previous Slide"
                print(last_action_text)
                last_swipe_time = current_time
                x_positions.clear()
                swipe_triggered = True

        # --- Mode hold detection ---
        stable_label = None
        if confidence >= GESTURE_MIN_CONFIDENCE and predicted_label in MODE_BY_LABEL:
            stable_label = predicted_label

        mode_action_triggered = False

        if stable_label is None:
            held_label = None
            held_label_start = 0.0
            held_label_fired = False
        else:
            if stable_label != held_label:
                held_label = stable_label
                held_label_start = current_time
                held_label_fired = False
            elif not held_label_fired and current_time - held_label_start >= MODE_HOLD_SECONDS:
                target_mode = MODE_BY_LABEL[stable_label]

                if active_mode == "NAVIGATION":
                    if target_mode == "CLEAR":
                        clear_canvas(state=slide_state)
                        last_action_text = "Cleared current slide annotations"
                    else:
                        active_mode = target_mode
                        last_action_text = f"Mode -> {target_mode}"
                else:
                    exit_gesture = MODE_EXIT_GESTURE_BY_MODE.get(active_mode)

                    if stable_label == exit_gesture:
                        active_mode = "NAVIGATION"
                        last_action_text = "Mode -> NAVIGATION"
                    elif target_mode == "CLEAR":
                        clear_canvas(state=slide_state)
                        last_action_text = "Cleared current slide annotations"
                    else:
                        last_action_text = f"Locked in {active_mode}: hold {exit_gesture} to exit"

                print(last_action_text)
                held_label_fired = True
                mode_action_triggered = True

        # --- Slide overlay: render onto the actual slide frame ---
        if active_mode in {"POINTER", "DRAWING"} and not swipe_triggered:
            slide = pointer_drawing_overlay(slide, hand_landmarks, active_mode, slide_state)

        if not swipe_triggered and not mode_action_triggered and confidence >= GESTURE_MIN_CONFIDENCE:
            last_action_text = f"Predicted: {predicted_label}"

    else:
        # Clear history if hand is not visible
        x_positions.clear()
        held_label = None
        held_label_start = 0.0
        held_label_fired = False

    # Always composite the persistent slide canvas, even when hand is absent.
    if slide_state.canvas is not None and not (active_mode in {"POINTER", "DRAWING"} and hand_landmarks is not None):
        slide = pointer_drawing_overlay(slide, None, "", slide_state)

    cv2.putText(
        frame,
        "NAV: hold one=Pointer, peace=Draw, stop=Clear | LOCKED: hold same gesture 3s to exit | q = Quit",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    webcam_display = cv2.resize(frame, (1280, 720))

    cv2.imshow("Gesture Webcam", webcam_display)
    cv2.imshow("Gesture Controlled Slides", slide)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quit")
        break

cap.release()
cv2.destroyAllWindows()
hands.close()