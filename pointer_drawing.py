"""
pointer_drawing.py
------------------
Pointer & Drawing Tools module for the Gesture-Based Presentation Controller.

Gesture contract (from shared hand_tracking module):
  - gesture == "POINTER"  →  one  (1 finger)
  - gesture == "DRAWING"  →  peace           (2 fingers)
  - gesture == "CLEAR"    →  stop                      (5 fingers)
  - anything else         →  idle / no action

Public API used by app/main.py / Streamlit UI:
  overlay = pointer_drawing_overlay(frame, landmarks, gesture)
  clear_canvas(width, height)  ← call when CLEAR gesture fires

"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────── Configuration ────────────────────────────────── #

@dataclass
class PointerConfig:
    # Laser pointer dot
    laser_color: tuple = (0, 0, 255)        # BGR red
    laser_radius: int  = 12
    laser_glow_radius: int = 22             # softer outer ring
    laser_glow_alpha: float = 0.35

    # Drawing
    draw_color: tuple  = (0, 255, 100)      # neon green
    draw_thickness: int = 4
    draw_smoothing: int = 5                 # rolling-average window (frames)

    # Canvas fade (seconds before strokes fade; 0 = permanent)
    fade_seconds: float = 0.0

    # Stabilisation: dead-zone radius in px (reduces micro-jitter)
    dead_zone_px: int = 6


CFG = PointerConfig()


# ─────────────────────────── Internal state ───────────────────────────────── #

@dataclass
class PointerDrawingState:
    canvas: Optional[np.ndarray] = None   # BGRA, same HxW as frame
    prev_draw_pt: Optional[tuple] = None
    history: list = field(default_factory=list)  # smoothing ring buffer
    last_gesture: str = ""

    def ensure_canvas(self, h: int, w: int) -> None:
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 4), dtype=np.uint8)   # transparent

    def clear(self) -> None:
        if self.canvas is not None:
            self.canvas[:] = 0
        self.prev_draw_pt = None
        self.history.clear()

    def reset_stroke(self) -> None:
        """Call when leaving DRAWING mode to break the stroke."""
        self.prev_draw_pt = None


_DEFAULT_STATE = PointerDrawingState()


def _resolve_state(state: Optional[PointerDrawingState]) -> PointerDrawingState:
    return state if state is not None else _DEFAULT_STATE


# ─────────────────────────── Helpers ──────────────────────────────────────── #

def _smooth_point(state: PointerDrawingState, pt: tuple, window: int) -> tuple:
    """Rolling average over the last `window` points."""
    state.history.append(pt)
    if len(state.history) > window:
        state.history.pop(0)
    xs = [p[0] for p in state.history]
    ys = [p[1] for p in state.history]
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))


def _outside_dead_zone(pt: tuple, prev: Optional[tuple], radius: int) -> bool:
    if prev is None:
        return True
    return ((pt[0] - prev[0]) ** 2 + (pt[1] - prev[1]) ** 2) ** 0.5 > radius


def _index_fingertip(landmarks, frame_w: int, frame_h: int) -> Optional[tuple]:
    """
    Extract index fingertip (landmark 8) pixel coords.
    Returns None if landmarks is None.
    """
    if landmarks is None:
        return None
    
    # Support both legacy NormalizedLandmarkList and new Tasks API (list)
    if hasattr(landmarks, "landmark"):
        lm = landmarks.landmark[8]                    # index fingertip
    else:
        lm = landmarks[8]

    x = int(lm.x * frame_w)
    y = int(lm.y * frame_h)
    # Clamp to frame boundaries
    x = max(0, min(frame_w - 1, x))
    y = max(0, min(frame_h - 1, y))
    return (x, y)


# ─────────────────────────── Drawing on canvas ────────────────────────────── #

def _draw_stroke(state: PointerDrawingState, pt: tuple) -> None:
    """Add a point to the persistent drawing canvas."""
    if state.canvas is None:
        return

    smooth_pt = _smooth_point(state, pt, CFG.draw_smoothing)

    if _outside_dead_zone(smooth_pt, state.prev_draw_pt, CFG.dead_zone_px):
        if state.prev_draw_pt is not None:
            # Draw anti-aliased line on the BGRA canvas (A=255 → fully opaque)
            cv2.line(
                state.canvas,
                state.prev_draw_pt,
                smooth_pt,
                (*CFG.draw_color, 255),           # BGR + Alpha
                CFG.draw_thickness,
                lineType=cv2.LINE_AA,
            )
        state.prev_draw_pt = smooth_pt


# ─────────────────────────── Laser pointer ────────────────────────────────── #

def _render_laser(frame: np.ndarray, pt: tuple) -> np.ndarray:
    """Composite a glowing laser dot onto `frame` (in-place copy)."""
    out = frame.copy()
    # Soft glow halo
    overlay = out.copy()
    cv2.circle(overlay, pt, CFG.laser_glow_radius, CFG.laser_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, CFG.laser_glow_alpha, out, 1 - CFG.laser_glow_alpha, 0, out)
    # Hard dot
    cv2.circle(out, pt, CFG.laser_radius, CFG.laser_color, -1, cv2.LINE_AA)
    # Bright specular highlight
    cv2.circle(out, (pt[0] - 3, pt[1] - 3), 3, (255, 220, 220), -1, cv2.LINE_AA)
    return out


# ─────────────────────────── Canvas composite ─────────────────────────────── #

def _composite_canvas(state: PointerDrawingState, frame: np.ndarray) -> np.ndarray:
    """Alpha-blend the persistent drawing canvas onto the frame."""
    if state.canvas is None or not np.any(state.canvas[:, :, 3]):
        return frame

    # Split canvas into BGR + alpha mask
    canvas_bgr = state.canvas[:, :, :3]
    alpha = state.canvas[:, :, 3:4].astype(np.float32) / 255.0

    out = frame.astype(np.float32)
    out = out * (1.0 - alpha) + canvas_bgr.astype(np.float32) * alpha
    return out.astype(np.uint8)


# ─────────────────────────── HUD overlays ─────────────────────────────────── #

def _draw_mode_badge(frame: np.ndarray, gesture: str) -> np.ndarray:
    """Small semi-transparent badge in the top-right corner."""
    labels = {
        "POINTER": ("● LASER", (0, 0, 200)),
        "DRAWING": ("✏ DRAW",  (0, 180, 80)),
        "CLEAR":   ("⊘ CLEAR", (0, 160, 220)),
    }
    if gesture not in labels:
        return frame

    text, color = labels[gesture]
    h, w = frame.shape[:2]
    font      = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.65
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 8
    x1, y1 = w - tw - pad * 2 - 10, 10
    x2, y2 = w - 10, 10 + th + pad * 2

    badge = frame.copy()
    cv2.rectangle(badge, (x1, y1), (x2, y2), (20, 20, 20), -1, cv2.LINE_AA)
    cv2.addWeighted(badge, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x1 + pad, y2 - pad),
                font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def _draw_crosshair(frame: np.ndarray, pt: tuple, color: tuple) -> None:
    """Tiny crosshair around the fingertip for drawing mode."""
    r = 10
    x, y = pt
    cv2.line(frame, (x - r, y), (x + r, y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y - r), (x, y + r), color, 1, cv2.LINE_AA)
    cv2.circle(frame, pt, r + 2, color, 1, cv2.LINE_AA)


# ─────────────────────────── Public API ───────────────────────────────────── #

def clear_canvas(width: int = 0, height: int = 0, state: Optional[PointerDrawingState] = None) -> None:
    """
    Wipe all drawing strokes.  Call this when the CLEAR gesture fires.
    width / height are optional; if omitted the existing canvas is zeroed.
    """
    drawing_state = _resolve_state(state)
    if width and height:
        drawing_state.canvas = np.zeros((height, width, 4), dtype=np.uint8)
    drawing_state.clear()


def pointer_drawing_overlay(
    frame: np.ndarray,
    landmarks,          # mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList | None
    gesture: str,       # "POINTER" | "DRAWING" | "CLEAR" | anything else
    state: Optional[PointerDrawingState] = None,
) -> np.ndarray:
    """
    Main entry-point.  Call once per frame from the Streamlit loop.

    Parameters
    ----------
    frame     : BGR frame from OpenCV / MediaPipe already drawn on.
    landmarks : MediaPipe hand landmarks for the detected hand (or None).
    gesture   : String gesture label from the shared gesture recognition module.

    Returns
    -------
    frame     : BGR frame with pointer / drawing annotations composited on top.
    """
    drawing_state = _resolve_state(state)
    h, w = frame.shape[:2]
    drawing_state.ensure_canvas(h, w)

    # Break drawing stroke if gesture changed away from DRAWING
    if gesture != "DRAWING" and drawing_state.last_gesture == "DRAWING":
        drawing_state.reset_stroke()
    drawing_state.last_gesture = gesture

    # ── CLEAR ────────────────────────────────────────────────────────────── #
    if gesture == "CLEAR":
        drawing_state.clear()
        frame = _draw_mode_badge(frame, gesture)
        return frame

    # ── Get fingertip position ────────────────────────────────────────────── #
    pt = _index_fingertip(landmarks, w, h)

    # ── Composite persistent drawing (always visible) ─────────────────────── #
    frame = _composite_canvas(drawing_state, frame)

    if pt is None:
        return frame                        # no hand detected → nothing to do

    # ── LASER POINTER ─────────────────────────────────────────────────────── #
    if gesture == "POINTER":
        drawing_state.history.clear()       # don't let old positions bleed in
        frame = _render_laser(frame, pt)
        frame = _draw_mode_badge(frame, gesture)

    # ── DRAWING ───────────────────────────────────────────────────────────── #
    elif gesture == "DRAWING":
        _draw_stroke(drawing_state, pt)
        frame = _composite_canvas(drawing_state, frame)    # re-composite with new stroke
        _draw_crosshair(frame, pt, CFG.draw_color)
        frame = _draw_mode_badge(frame, gesture)

    return frame
