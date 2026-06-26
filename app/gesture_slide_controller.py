# main.py

import cv2
import time
import sys
import os
import json
import mediapipe as mp
import threading
import numpy as np
from functools import lru_cache
from pathlib import Path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from utils.slide_controller import SlideController
from pointer_drawing import PointerDrawingState, pointer_drawing_overlay, clear_canvas
from training.preprocess import normalize_landmarks


HAND_LANDMARKER_PATH = Path(__file__).resolve().parent.parent / "training" / "models" / "hand_landmarker.task"
GESTURE_MODEL_PATH   = Path(__file__).resolve().parent.parent / "training" / "artifacts" / "keras_model.keras"
LABELS_PATH          = Path(__file__).resolve().parent.parent / "training" / "artifacts" / "labels.json"

PREDICTION_COOLDOWN    = 0.9
SWIPE_COOLDOWN         = 1.0
GESTURE_MIN_CONFIDENCE = 0.55
MODE_HOLD_SECONDS      = 3.0
SWIPE_HISTORY_LENGTH   = 8
SWIPE_MIN_MOVEMENT     = 60
SWIPE_MIN_STEP         = 6
SWIPE_DIRECTION_RATIO  = 0.1

# ── Pinch thresholds (normalised wrist→middle-mcp distance units) ─────────
PINCH_CLOSE_THRESH  = 0.2
PINCH_OPEN_THRESH   = 0.3

# ── Zoom constants ────────────────────────────────────────────────────────
ZOOM_SENSITIVITY    = 1.0
ZOOM_MIN            = 1.0
ZOOM_MAX            = 3.0

# ── Cursor smoothing & sensitivity ───────────────────────────────────────
# CURSOR_ALPHA: EMA weight applied to each new raw sample.
#   Lower  → heavier smoothing, more lag   (try 0.12–0.18 for max stability)
#   Higher → snappier tracking, more jitter (approach 1.0 to disable EMA)
CURSOR_ALPHA = 0.3

# CURSOR_SENSITIVITY: multiplier applied to the cursor's displacement from the
# screen centre.  1.0 = cursor matches hand 1-to-1.  1.8 means a hand movement
# of N px from centre registers as 1.8 × N px of cursor travel, so the cursor
# reaches slide edges with smaller hand movements.
CURSOR_SENSITIVITY = 1.2

# ── Gesture → mode mapping ────────────────────────────────────────────────
MODE_BY_LABEL = {
    "one":   "POINTER",
    "ok":    "DRAWING",
    "peace": "ZOOM",
    "stop":  "CLEAR",
    "fist":  "EXIT",
}
EXIT_GESTURE = "fist"

MODE_EXIT_GESTURE_BY_MODE = {
    "DRAWING": "fist",
    "ZOOM":    "fist",
    "POINTER": "fist",
}

# ─────────────────────────────────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode
HAND_CONNECTIONS      = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

for p in (HAND_LANDMARKER_PATH, GESTURE_MODEL_PATH, LABELS_PATH):
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")


@lru_cache(maxsize=1)
def load_gesture_model():
    import tensorflow as tf
    return tf.keras.models.load_model(GESTURE_MODEL_PATH)


@lru_cache(maxsize=1)
def load_labels() -> list[str]:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def landmarks_to_features(hand_landmarks) -> np.ndarray:
    raw = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks], dtype=np.float32)
    return normalize_landmarks(raw, scale=False).reshape(1, -1)


def predict_gesture(hand_landmarks):
    model  = load_gesture_model()
    labels = load_labels()
    preds  = model.predict(landmarks_to_features(hand_landmarks), verbose=0)[0]
    idx    = int(np.argmax(preds))
    return labels[idx], float(preds[idx])

_prediction_result = {
    "label": "none",
    "confidence": 0.0,
}

_prediction_lock = threading.Lock()

_prediction_running = False


def _prediction_worker(hand_landmarks):
    global _prediction_running

    try:
        label, conf = predict_gesture(hand_landmarks)

        with _prediction_lock:
            _prediction_result["label"] = label
            _prediction_result["confidence"] = conf
    finally:
        _prediction_running = False

SWIPE_MIN_VELOCITY = 18   # px per frame — add near the other constants

def detect_swipe(x_history: list[int]) -> str | None:
    if len(x_history) < SWIPE_HISTORY_LENGTH:
        return None
    movement = x_history[-1] - x_history[0]
    if abs(movement) < SWIPE_MIN_MOVEMENT:
        return None

    deltas = np.diff(np.array(x_history))
    avg_velocity = abs(movement) / len(deltas)   # ← avg px/frame
    if avg_velocity < SWIPE_MIN_VELOCITY:         # ← velocity gate
        return None

    if movement > 0:
        consistent = np.count_nonzero(deltas >  SWIPE_MIN_STEP)
        direction  = "right"
    else:
        consistent = np.count_nonzero(deltas < -SWIPE_MIN_STEP)
        direction  = "left"
    return direction if consistent / len(deltas) >= SWIPE_DIRECTION_RATIO else None


def draw_hand_landmarks(frame, hand_landmarks, width, height):
    for conn in HAND_CONNECTIONS:
        s, e = hand_landmarks[conn.start], hand_landmarks[conn.end]
        cv2.line(frame,
                 (int(s.x * width), int(s.y * height)),
                 (int(e.x * width), int(e.y * height)),
                 (255, 255, 255), 2)
    for lm in hand_landmarks:
        cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 4, (0, 255, 0), -1)


def get_pinch_distance(hand_landmarks) -> float:
    """
    Return thumb-tip ↔ index-tip distance normalised by wrist→middle-MCP span.
    Scale-invariant.  Indices: 4=thumb tip, 8=index tip, 0=wrist, 9=middle MCP.
    """
    thumb = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
    index = np.array([hand_landmarks[8].x, hand_landmarks[8].y])
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    mid   = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
    ref   = np.linalg.norm(mid - wrist) or 1e-6
    return float(np.linalg.norm(index - thumb) / ref)


def get_index_tip_px(hand_landmarks, width: int, height: int) -> tuple[int, int]:
    """Raw (un-smoothed) index fingertip pixel position."""
    lm = hand_landmarks[8]
    return int(lm.x * width), int(lm.y * height)


# ── Zoom rendering ────────────────────────────────────────────────────────
def apply_zoom(slide: np.ndarray,
               zoom_level: float,
               cursor_px: tuple[int, int]) -> np.ndarray:
    """Zoom the slide image around cursor_px by zoom_level."""
    if zoom_level <= 1.0:
        return slide

    h, w = slide.shape[:2]
    cx, cy = cursor_px

    crop_w = int(w / zoom_level)
    crop_h = int(h / zoom_level)

    x1 = max(0, min(cx - crop_w // 2, w - crop_w))
    y1 = max(0, min(cy - crop_h // 2, h - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = slide[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ── Cursor smoother ───────────────────────────────────────────────────────
class CursorSmoother:
    """
    Exponential moving average (EMA) + sensitivity amplification for the
    index-fingertip cursor.

    Call  update()  once per frame when a hand is detected.
    Read  .px       anywhere that needs the smoothed cursor position.
    Call  reset()   when no hand is visible so the smoother doesn't drift
                    toward stale coordinates on the next detection.

    Tuning
    ------
    CURSOR_ALPHA       – EMA weight per frame (module-level constant).
                         0.12–0.18  very smooth, noticeable lag
                         0.25       default: smooth but still responsive
                         0.4+       snappier, some jitter returns
    CURSOR_SENSITIVITY – displacement amplifier from screen centre.
                         1.0  cursor matches hand 1-to-1
                         1.8  default: reach edges with smaller hand movements
                         2.5  very sensitive (good for large slides / small gestures)
    """

    def __init__(self) -> None:
        self._sx: float | None = None
        self._sy: float | None = None
        self._last: tuple[int, int] = (0, 0)

    def update(self, raw_x: int, raw_y: int, width: int, height: int) -> tuple[int, int]:
        """
        Feed a new raw fingertip position.  Returns the smoothed,
        amplified, bounds-clamped pixel coordinate.
        """
        if self._sx is None:
            # First detection this run: seed with raw position to avoid
            # a sudden jump from (0,0) to wherever the hand appears.
            self._sx = float(raw_x)
            self._sy = float(raw_y)
        else:
            a = CURSOR_ALPHA
            self._sx = a * raw_x + (1.0 - a) * self._sx
            self._sy = a * raw_y + (1.0 - a) * self._sy

        # Amplify displacement from screen centre
        cx, cy = width / 2.0, height / 2.0
        sx = cx + (self._sx - cx) * CURSOR_SENSITIVITY
        sy = cy + (self._sy - cy) * CURSOR_SENSITIVITY

        # Clamp to valid frame bounds
        self._last = (
            int(max(0.0, min(sx, width  - 1))),
            int(max(0.0, min(sy, height - 1))),
        )
        return self._last

    @property
    def px(self) -> tuple[int, int]:
        """Last computed smoothed cursor position (or (0,0) before first update)."""
        return self._last

    def reset(self) -> None:
        """Discard smoothing state.  Call when the hand disappears."""
        self._sx   = None
        self._sy   = None
        self._last = (0, 0)


# ── MediaPipe setup ───────────────────────────────────────────────────────
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
hands = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

controller                                      = SlideController()
slide_annotation_states: dict[int, PointerDrawingState] = {}

# Gesture state
x_positions            = []
last_swipe_time        = 0.0
last_prediction_time   = 0.0
active_mode            = "NAVIGATION"
held_label             = None
held_label_start       = 0.0
held_label_fired       = False

last_action_text           = "No action yet"
last_prediction_text       = "No gesture yet"
last_prediction_confidence = 0.0

# Pinch state (hysteresis) + previous-frame value for edge detection
is_pinched   = False
was_pinched  = False   # tracks the previous frame's pinch state

# Zoom state
zoom_level        = 1.0
zoom_ref_distance = None

# Cursor smoother — single shared instance updated once per frame
smoother = CursorSmoother()

fps_last_time = time.time()
fps_value     = 0.0

def get_slide_state(idx: int) -> PointerDrawingState:
    if idx not in slide_annotation_states:
        slide_annotation_states[idx] = PointerDrawingState()
    return slide_annotation_states[idx]


# ── Main loop ─────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read webcam")
        break

    current_time      = time.time()
    fps_raw   = 1.0 / max(current_time - fps_last_time, 1e-6)
    fps_value = 0.1 * fps_raw + 0.9 * fps_value   # EMA, α = 0.1
    fps_last_time = current_time
    frame_ts_ms       = int(current_time * 1000)
    time_since_swipe  = current_time - last_swipe_time
    swipe_remaining   = SWIPE_COOLDOWN - time_since_swipe
    swipe_on_cooldown = swipe_remaining > 0

    frame            = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results   = hands.detect_for_video(mp_image, frame_ts_ms)

    slide             = controller.get_current_slide()
    slide             = cv2.resize(slide, (900, 600))
    current_slide_idx = controller.current_slide_index
    slide_state       = get_slide_state(current_slide_idx)

    # ── HUD overlays ──────────────────────────────────────────────────────
    slide_text = f"Slide {controller.get_slide_number()} / {controller.get_total_slides()}"
    cv2.putText(slide, f"FPS: {fps_value:.1f}", (30, 300 if active_mode != "ZOOM" else 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    cv2.putText(slide, slide_text,
                (30, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 0, 255), 2)
    cv2.putText(slide,
                f"Prediction: {last_prediction_text} ({last_prediction_confidence:.2f})",
                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(slide, f"Action: {last_action_text}",
                (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Swipe status line — context-aware for DRAWING mode
    if active_mode == "DRAWING":
        swipe_txt = "Swipe: ← undo  → redo"
        swipe_clr = (0, 200, 140)
    elif active_mode != "NAVIGATION":
        exit_g    = MODE_EXIT_GESTURE_BY_MODE.get(active_mode, "")
        swipe_txt = f"Swipe: LOCKED ({active_mode}) | hold {exit_g} 3s to exit"
        swipe_clr = (0, 140, 255)
    elif swipe_on_cooldown:
        swipe_txt = f"Swipe: COOLDOWN ({swipe_remaining:.1f}s)"
        swipe_clr = (0, 0, 255)
    else:
        swipe_txt = "Swipe: READY"
        swipe_clr = (0, 255, 0)

    cv2.putText(slide, swipe_txt,
                (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, swipe_clr, 2)
    cv2.putText(slide, f"Mode: {active_mode}",
                (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 120, 0), 2)

    if active_mode == "ZOOM":
        cv2.putText(slide, f"Zoom: {zoom_level:.2f}x",
                    (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

    # ── Hand detection ────────────────────────────────────────────────────
    hand_landmarks = results.hand_landmarks[0] if results.hand_landmarks else None

    if hand_landmarks:
        wrist     = hand_landmarks[0]
        current_x = int(wrist.x * width)
        x_positions.append(current_x)
        if len(x_positions) > SWIPE_HISTORY_LENGTH:
            x_positions.pop(0)

        # ── Pinch detection (hysteresis) ──────────────────────────────────
        # was_pinched captures the state from the end of the previous frame
        # so we can detect the falling edge (pinch release = stroke commit).
        was_pinched = is_pinched
        pinch_dist  = get_pinch_distance(hand_landmarks)
        if is_pinched:
            if pinch_dist > PINCH_OPEN_THRESH:
                is_pinched = False
        else:
            if pinch_dist < PINCH_CLOSE_THRESH:
                is_pinched = True

        # ── Commit stroke on pinch release (DRAWING mode only) ────────────
        # The falling edge (was_pinched True → is_pinched False) marks the
        # end of one continuous stroke.  We snapshot the canvas here so that
        # a subsequent swipe-left can undo exactly that stroke.
        if active_mode == "DRAWING" and was_pinched and not is_pinched:
            slide_state.commit_stroke()

        # ── Smooth & amplify cursor (single update per frame) ─────────────
        raw_x, raw_y = get_index_tip_px(hand_landmarks, width, height)
        cursor_px    = smoother.update(raw_x, raw_y, width, height)

        # ── Gesture prediction (throttled) ────────────────────────────────
        if (
            current_time - last_prediction_time > PREDICTION_COOLDOWN
            and not _prediction_running
        ):
            last_prediction_time = current_time
            _prediction_running = True

            threading.Thread(
                target=_prediction_worker,
                args=(hand_landmarks,),
                daemon=True,
            ).start()

        with _prediction_lock:
            predicted_label = _prediction_result["label"]
            confidence = _prediction_result["confidence"]

        last_prediction_text = predicted_label
        last_prediction_confidence = confidence
        # ── Swipe detection ───────────────────────────────────────────────
        # NAVIGATION mode : left/right → previous/next slide  (unchanged)
        # DRAWING mode    : left → undo last stroke, right → redo
        # All other modes : swipe ignored
        swipe_triggered = False
        if (active_mode in {"NAVIGATION", "DRAWING"}
                and len(x_positions) == SWIPE_HISTORY_LENGTH
                and current_time - last_swipe_time > SWIPE_COOLDOWN):

            swipe_dir = detect_swipe(x_positions)

            if active_mode == "NAVIGATION":
                if swipe_dir == "right":
                    controller.next_slide()
                    last_action_text = "Swipe Right -> Next Slide"
                    last_swipe_time  = current_time
                    x_positions.clear()
                    swipe_triggered  = True
                elif swipe_dir == "left":
                    controller.previous_slide()
                    last_action_text = "Swipe Left -> Previous Slide"
                    last_swipe_time  = current_time
                    x_positions.clear()
                    swipe_triggered  = True

            elif active_mode == "DRAWING":
                if swipe_dir == "left":
                    # Undo: if a stroke is currently in progress (pinch held),
                    # release it cleanly before undoing so we don't leave a
                    # half-drawn stroke that undo can't account for.
                    if is_pinched:
                        slide_state.commit_stroke()
                        is_pinched  = False
                        was_pinched = False
                    if slide_state.undo():
                        last_action_text = f"Undo  (remaining: {len(slide_state.undo_stack)})"
                    else:
                        last_action_text = "Nothing to undo"
                    last_swipe_time = current_time
                    x_positions.clear()
                    swipe_triggered = True
                elif swipe_dir == "right":
                    if slide_state.redo():
                        last_action_text = f"Redo  (remaining: {len(slide_state.redo_stack)})"
                    else:
                        last_action_text = "Nothing to redo"
                    last_swipe_time = current_time
                    x_positions.clear()
                    swipe_triggered = True

        # ── Mode-hold detection ───────────────────────────────────────────
        stable_label = (predicted_label
                        if confidence >= GESTURE_MIN_CONFIDENCE
                           and predicted_label in MODE_BY_LABEL
                        else None)
        mode_action_triggered = False

        if stable_label is None:
            held_label       = None
            held_label_start = 0.0
            held_label_fired = False
        else:
            if stable_label != held_label:
                held_label       = stable_label
                held_label_start = current_time
                held_label_fired = False
            elif (not held_label_fired
                  and current_time - held_label_start >= MODE_HOLD_SECONDS):

                target_mode = MODE_BY_LABEL[stable_label]

                if target_mode == "EXIT":
                    if active_mode != "NAVIGATION":
                        active_mode       = "NAVIGATION"
                        last_action_text  = "Mode -> NAVIGATION"
                        zoom_level        = 1.0
                        zoom_ref_distance = None

                elif active_mode == "NAVIGATION":
                    if target_mode == "CLEAR":
                        clear_canvas(state=slide_state)
                        last_action_text = "Cleared current slide annotations"
                    else:
                        active_mode      = target_mode
                        last_action_text = f"Mode -> {target_mode}"
                        if target_mode == "ZOOM":
                            zoom_level        = 1.0
                            zoom_ref_distance = None

                else:
                    if target_mode == "CLEAR":
                        clear_canvas(state=slide_state)
                        last_action_text = "Cleared current slide annotations"
                    else:
                        last_action_text = f"Locked in {active_mode}: hold fist 3s to exit"

                print(last_action_text)
                held_label_fired      = True
                mode_action_triggered = True

        # ── DRAWING mode: pinch=draw, unpinch=move cursor ─────────────────
        if active_mode == "DRAWING" and not swipe_triggered:
            slide = pointer_drawing_overlay(
                slide, hand_landmarks,
                "DRAWING" if is_pinched else "POINTER",
                slide_state,
                cursor_px=cursor_px,
            )

        # ── ZOOM mode ─────────────────────────────────────────────────────
        elif active_mode == "ZOOM" and not swipe_triggered:
            if is_pinched:
                zoom_ref_distance = None
                cv2.circle(slide, cursor_px, 10, (0, 220, 255), 2)
            else:
                if zoom_ref_distance is None:
                    zoom_ref_distance = pinch_dist or 0.15

                spread_ratio = pinch_dist / (zoom_ref_distance or 1e-6)
                zoom_level   = float(np.clip(
                    spread_ratio * ZOOM_SENSITIVITY,
                    ZOOM_MIN, ZOOM_MAX,
                ))

                slide_h, slide_w = slide.shape[:2]
                slide_cursor = (
                    int(cursor_px[0] / width  * slide_w),
                    int(cursor_px[1] / height * slide_h),
                )
                slide = apply_zoom(slide, zoom_level, slide_cursor)

                cv2.circle(slide, (slide_w // 2, slide_h // 2),
                           12, (0, 220, 255), 2)

        # ── POINTER / NAVIGATION mode ─────────────────────────────────────
        elif active_mode == "POINTER" and not swipe_triggered:
            slide = pointer_drawing_overlay(
                slide, hand_landmarks, "POINTER", slide_state,
                cursor_px=cursor_px,
            )

        if (not swipe_triggered
                and not mode_action_triggered
                and confidence >= GESTURE_MIN_CONFIDENCE):
            last_action_text = f"Predicted: {predicted_label}"

    else:
        # No hand visible — reset all per-hand state
        x_positions.clear()
        held_label        = None
        held_label_start  = 0.0
        held_label_fired  = False
        is_pinched        = False
        was_pinched       = False
        zoom_ref_distance = None
        smoother.reset()

    # ── Persistent canvas composite (drawing mode, no hand) ───────────────
    if (slide_state.canvas is not None
            and not (active_mode in {"DRAWING"} and hand_landmarks is not None)):
        slide = pointer_drawing_overlay(slide, None, "", slide_state)

    # ── Webcam HUD ────────────────────────────────────────────────────────
    cv2.putText(
        frame,
        "NAV: hold ok=Draw, peace=Zoom, stop=Clear | hold fist 3s to exit | q=Quit",
        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
    )

    webcam_display = cv2.resize(frame, (640, 480))
    cv2.imshow("Gesture Webcam", webcam_display)
    cv2.imshow("Gesture Controlled Slides", slide)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quit")
        break

cap.release()
cv2.destroyAllWindows()
hands.close()