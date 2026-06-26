"""
pointer_drawing.py
------------------
Pointer & Drawing Tools module for the Gesture-Based Presentation Controller.

Gesture contract (from shared hand_tracking module):
  - gesture == "POINTER"  →  one  (1 finger)
  - gesture == "DRAWING"  →  ok / pinch held
  - gesture == "CLEAR"    →  stop (5 fingers)
  - anything else         →  idle / no action

Public API used by app/main.py:
  overlay = pointer_drawing_overlay(frame, landmarks, gesture, state,
                                    cursor_px=None)
  clear_canvas(width, height, state)  ← call when CLEAR gesture fires

Undo / redo
-----------
Each completed pinch stroke is committed to an undo stack as a full canvas
snapshot.  Swipe-left in DRAWING mode calls state.undo(), swipe-right calls
state.redo().  Clearing the canvas is also undoable.

Smoothing note
--------------
main.py feeds a pre-smoothed cursor_px (via CursorSmoother / EMA).  When
that override is present the internal rolling-average window is set to 1
(identity) and the dead-zone is bypassed so there is no double-smoothing lag.
When cursor_px is None (legacy / standalone use) the old behaviour is
preserved: the raw landmark is extracted and the rolling average + dead-zone
apply as before.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────── Configuration ────────────────────────────────── #

MAX_UNDO_STEPS = 20   # cap memory use; oldest snapshots are dropped first


@dataclass
class PointerConfig:
    # Laser pointer dot
    laser_color: tuple      = (0, 0, 255)
    laser_radius: int       = 12
    laser_glow_radius: int  = 22
    laser_glow_alpha: float = 0.35

    # Drawing
    draw_color: tuple    = (0, 255, 100)
    draw_thickness: int  = 4
    # Rolling-average window used when no external cursor_px is provided.
    # When cursor_px IS provided (pre-smoothed by CursorSmoother) this is
    # overridden to 1 inside pointer_drawing_overlay to avoid double-smoothing.
    draw_smoothing: int  = 5

    # Canvas fade (seconds before strokes fade; 0 = permanent)
    fade_seconds: float = 0.0

    # Dead-zone radius in px — suppresses micro-jitter when using raw landmarks.
    # Bypassed automatically when a pre-smoothed cursor_px is supplied.
    dead_zone_px: int = 6


CFG = PointerConfig()


# ─────────────────────────── Internal state ───────────────────────────────── #

@dataclass
class PointerDrawingState:
    canvas: Optional[np.ndarray] = None        # BGRA, same HxW as frame
    prev_draw_pt: Optional[tuple] = None
    history: list = field(default_factory=list) # smoothing ring buffer
    last_gesture: str = ""

    # Undo / redo stacks — each entry is a full BGRA canvas snapshot.
    # undo_stack[-1] is the state just before the most recent committed stroke.
    # redo_stack[-1] is the state that was undone most recently.
    undo_stack: list = field(default_factory=list)
    redo_stack: list = field(default_factory=list)

    def ensure_canvas(self, h: int, w: int) -> None:
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 4), dtype=np.uint8)

    # ── Undo / redo primitives ─────────────────────────────────────────── #

    def _push_undo(self) -> None:
        """Snapshot current canvas onto the undo stack (bounded)."""
        if self.canvas is not None:
            self.undo_stack.append(self.canvas.copy())
            if len(self.undo_stack) > MAX_UNDO_STEPS:
                self.undo_stack.pop(0)

    def commit_stroke(self) -> None:
        """
        Call on pinch release (drawing stroke finished).
        Saves a pre-stroke snapshot so undo restores to before that stroke.
        Any pending redo history is discarded — a new stroke forks the timeline.
        """
        self._push_undo()
        self.redo_stack.clear()

    def undo(self) -> bool:
        """
        Restore the canvas to the state before the last committed stroke.
        Returns True if an undo was available, False if the stack was empty.
        """
        if not self.undo_stack:
            return False
        if self.canvas is not None:
            self.redo_stack.append(self.canvas.copy())
        self.canvas = self.undo_stack.pop()
        self.prev_draw_pt = None   # break any in-progress stroke
        return True

    def redo(self) -> bool:
        """
        Re-apply the most recently undone stroke.
        Returns True if a redo was available, False if the stack was empty.
        """
        if not self.redo_stack:
            return False
        if self.canvas is not None:
            self.undo_stack.append(self.canvas.copy())
        self.canvas = self.redo_stack.pop()
        self.prev_draw_pt = None
        return True

    # ── Canvas management ──────────────────────────────────────────────── #

    def clear(self) -> None:
        """Wipe canvas.  The pre-clear state is pushed to the undo stack."""
        self._push_undo()
        self.redo_stack.clear()
        if self.canvas is not None:
            self.canvas[:] = 0
        self.prev_draw_pt = None
        self.history.clear()

    def reset_stroke(self) -> None:
        """Call when leaving DRAWING mode to break the stroke without committing."""
        self.prev_draw_pt = None


_DEFAULT_STATE = PointerDrawingState()


def _resolve_state(state: Optional[PointerDrawingState]) -> PointerDrawingState:
    return state if state is not None else _DEFAULT_STATE


# ─────────────────────────── Helpers ──────────────────────────────────────── #

def _smooth_point(state: PointerDrawingState, pt: tuple, window: int) -> tuple:
    """Rolling average over the last `window` points.  window=1 → identity."""
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
    Extract index fingertip (landmark 8) pixel coords from raw MediaPipe output.
    Returns None if landmarks is None.
    Only called when no pre-smoothed cursor_px override is provided.
    """
    if landmarks is None:
        return None

    if hasattr(landmarks, "landmark"):
        lm = landmarks.landmark[8]
    else:
        lm = landmarks[8]

    x = int(lm.x * frame_w)
    y = int(lm.y * frame_h)
    x = max(0, min(frame_w - 1, x))
    y = max(0, min(frame_h - 1, y))
    return (x, y)


# ─────────────────────────── Drawing on canvas ────────────────────────────── #

def _draw_stroke(state: PointerDrawingState, pt: tuple,
                 smoothing_window: int, use_dead_zone: bool) -> None:
    """
    Add a point to the persistent drawing canvas.

    Parameters
    ----------
    smoothing_window : pass 1 to skip internal rolling average (when the
                       cursor is already EMA-smoothed upstream).
    use_dead_zone    : pass False to skip the dead-zone check (same reason).
    """
    if state.canvas is None:
        return

    smooth_pt = _smooth_point(state, pt, smoothing_window)

    if (not use_dead_zone) or _outside_dead_zone(smooth_pt, state.prev_draw_pt, CFG.dead_zone_px):
        if state.prev_draw_pt is not None:
            cv2.line(
                state.canvas,
                state.prev_draw_pt,
                smooth_pt,
                (*CFG.draw_color, 255),
                CFG.draw_thickness,
                lineType=cv2.LINE_AA,
            )
        state.prev_draw_pt = smooth_pt


# ─────────────────────────── Laser pointer ────────────────────────────────── #

def _render_laser(frame: np.ndarray, pt: tuple) -> np.ndarray:
    """Composite a glowing laser dot onto `frame` (returns a new array)."""
    out = frame.copy()
    overlay = out.copy()
    cv2.circle(overlay, pt, CFG.laser_glow_radius, CFG.laser_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, CFG.laser_glow_alpha, out, 1 - CFG.laser_glow_alpha, 0, out)
    cv2.circle(out, pt, CFG.laser_radius, CFG.laser_color, -1, cv2.LINE_AA)
    cv2.circle(out, (pt[0] - 3, pt[1] - 3), 3, (255, 220, 220), -1, cv2.LINE_AA)
    return out


# ─────────────────────────── Canvas composite ─────────────────────────────── #

def _composite_canvas(state: PointerDrawingState, frame: np.ndarray) -> np.ndarray:
    """Alpha-blend the persistent drawing canvas onto the frame."""
    if state.canvas is None or not np.any(state.canvas[:, :, 3]):
        return frame

    canvas_bgr = state.canvas[:, :, :3]
    alpha      = state.canvas[:, :, 3:4].astype(np.float32) / 255.0

    out = frame.astype(np.float32)
    out = out * (1.0 - alpha) + canvas_bgr.astype(np.float32) * alpha
    return out.astype(np.uint8)


# ─────────────────────────── HUD overlays ─────────────────────────────────── #

def _draw_mode_badge(frame: np.ndarray, gesture: str,
                     undo_count: int = 0, redo_count: int = 0) -> np.ndarray:
    """
    Small semi-transparent badge in the top-right corner.
    In DRAWING mode the badge also shows undo/redo depth.
    """
    labels = {
        "POINTER": ("● LASER", (0, 0, 200)),
        "DRAWING": ("✏ DRAW",  (0, 180, 80)),
        "CLEAR":   ("⊘ CLEAR", (0, 160, 220)),
    }
    if gesture not in labels:
        return frame

    base_text, color = labels[gesture]
    if gesture == "DRAWING" and (undo_count or redo_count):
        text = f"{base_text}  undo:{undo_count}  redo:{redo_count}"
    else:
        text = base_text

    h, w = frame.shape[:2]
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.65
    thickness  = 1
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

def clear_canvas(width: int = 0, height: int = 0,
                 state: Optional[PointerDrawingState] = None) -> None:
    """
    Wipe all drawing strokes.  The pre-clear state is pushed to the undo stack
    so the clear itself can be undone with a swipe-left.
    width / height are optional; if omitted the existing canvas is zeroed.
    """
    drawing_state = _resolve_state(state)
    if width and height:
        # Re-create canvas at the requested size — but first snapshot the old
        # one so undo works even across a resize.
        drawing_state._push_undo()
        drawing_state.redo_stack.clear()
        drawing_state.canvas = np.zeros((height, width, 4), dtype=np.uint8)
        drawing_state.prev_draw_pt = None
        drawing_state.history.clear()
    else:
        drawing_state.clear()


def pointer_drawing_overlay(
    frame: np.ndarray,
    landmarks,
    gesture: str,
    state: Optional[PointerDrawingState] = None,
    *,
    cursor_px: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Main entry-point.  Call once per frame.

    Parameters
    ----------
    frame     : BGR frame (OpenCV / MediaPipe already drawn on it).
    landmarks : MediaPipe hand landmarks for the detected hand, or None.
    gesture   : "POINTER" | "DRAWING" | "CLEAR" | anything else.
    state     : Per-slide PointerDrawingState; uses a module-level default
                if None (backward-compatible).
    cursor_px : Optional pre-smoothed fingertip position (x, y) in frame
                pixel coordinates, supplied by main.py's CursorSmoother.
                When provided:
                  • _index_fingertip() is NOT called (raw landmark ignored
                    for cursor position).
                  • Internal rolling-average window is set to 1 (identity)
                    to avoid adding lag on top of the upstream EMA.
                  • Dead-zone check is skipped (EMA already killed micro-jitter).
                When None the original behaviour is preserved: the raw
                landmark is extracted and the rolling average + dead-zone
                apply normally.

    Returns
    -------
    frame : BGR frame with pointer / drawing annotations composited on top.
    """
    drawing_state = _resolve_state(state)
    h, w = frame.shape[:2]
    drawing_state.ensure_canvas(h, w)

    pre_smoothed: bool = cursor_px is not None

    # Break drawing stroke if gesture changed away from DRAWING
    if gesture != "DRAWING" and drawing_state.last_gesture == "DRAWING":
        drawing_state.reset_stroke()
    drawing_state.last_gesture = gesture

    # ── CLEAR ────────────────────────────────────────────────────────────── #
    if gesture == "CLEAR":
        drawing_state.clear()
        frame = _draw_mode_badge(frame, gesture)
        return frame

    # ── Resolve cursor position ───────────────────────────────────────────── #
    if pre_smoothed:
        pt = cursor_px
    else:
        pt = _index_fingertip(landmarks, w, h)

    # ── Composite persistent drawing (always visible) ─────────────────────── #
    frame = _composite_canvas(drawing_state, frame)

    if pt is None:
        return frame

    # ── LASER POINTER ─────────────────────────────────────────────────────── #
    if gesture == "POINTER":
        drawing_state.history.clear()
        frame = _render_laser(frame, pt)
        frame = _draw_mode_badge(frame, gesture)

    # ── DRAWING ───────────────────────────────────────────────────────────── #
    elif gesture == "DRAWING":
        smoothing_window = 1 if pre_smoothed else CFG.draw_smoothing
        _draw_stroke(drawing_state, pt,
                     smoothing_window=smoothing_window,
                     use_dead_zone=not pre_smoothed)
        _draw_crosshair(frame, pt, CFG.draw_color)
        frame = _draw_mode_badge(
            frame, gesture,
            undo_count=len(drawing_state.undo_stack),
            redo_count=len(drawing_state.redo_stack),
        )

    return frame