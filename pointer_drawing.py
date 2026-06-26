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
  state.set_base_slide(slide)         ← call on every slide change

Architecture (baked-slide model)
---------------------------------
Instead of maintaining a separate BGRA canvas and alpha-compositing it onto
the slide every frame, drawings are baked directly onto a BGR working_slide:

  original_slide  – immutable copy of the slide as it was when set_base_slide()
                    was last called.  Never drawn on.
  working_slide   – original_slide with all *committed* strokes permanently
                    rendered in.  This is what gets displayed most frames.
  live_stroke_pts – list of (x, y) points for the stroke currently being drawn
                    (pinch held but not yet released).  Each frame during a
                    pinch this is replayed on a working_slide.copy() so the
                    cursor frame is clean.  On commit (pinch release / undo
                    trigger) the pts are baked into working_slide permanently.

Undo / redo
-----------
Each completed pinch stroke is committed to an undo stack as a full
working_slide snapshot.  Swipe-left in DRAWING mode calls state.undo(),
swipe-right calls state.redo().  Clearing the canvas is also undoable.
The snapshot is taken *before* the stroke, so undo restores pre-stroke state.

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

    # Dead-zone radius in px — suppresses micro-jitter when using raw landmarks.
    # Bypassed automatically when a pre-smoothed cursor_px is supplied.
    dead_zone_px: int = 6


CFG = PointerConfig()


# ─────────────────────────── Internal state ───────────────────────────────── #

@dataclass
class PointerDrawingState:
    # ── Baked-slide storage ────────────────────────────────────────────── #
    # original_slide: BGR copy of the slide at the time of the last
    #   set_base_slide() call.  Never mutated.  Used by clear() to reset.
    original_slide: Optional[np.ndarray] = None

    # working_slide: original_slide with all *committed* strokes baked in.
    #   Displayed as-is on frames where no live stroke is being drawn.
    working_slide: Optional[np.ndarray] = None

    # live_stroke_pts: (x, y) points collected during the current pinch.
    #   Replayed on top of working_slide each frame for real-time feedback.
    #   Cleared and baked into working_slide on commit.
    live_stroke_pts: list = field(default_factory=list)

    # ── Shared helpers ─────────────────────────────────────────────────── #
    history: list = field(default_factory=list)   # smoothing ring buffer
    last_gesture: str = ""

    # ── Undo / redo stacks ─────────────────────────────────────────────── #
    # Each entry is a BGR working_slide snapshot (before the committed stroke).
    undo_stack: list = field(default_factory=list)
    redo_stack: list = field(default_factory=list)

    # ── Slide initialisation ───────────────────────────────────────────── #

    def set_base_slide(self, slide: np.ndarray) -> None:
        """
        Call whenever the displayed slide changes (new slide index, resize, etc.).

        Saves an immutable copy as original_slide and resets working_slide to
        match it.  Existing undo/redo history is cleared because it refers to
        a different slide's pixel data.

        main.py should call this once after:
          • Loading / switching slides
          • Resizing the slide image

        It does NOT need to be called every frame.
        """
        self.original_slide = slide.copy()
        self.working_slide  = slide.copy()
        self.live_stroke_pts.clear()
        self.history.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()

    def ensure_slide(self, h: int, w: int) -> None:
        """
        Fallback initialisation: if set_base_slide() was never called, create
        a black working_slide at the requested size.  Preserves backward
        compatibility for callers that don't call set_base_slide() explicitly.
        """
        if self.working_slide is None or self.working_slide.shape[:2] != (h, w):
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            self.original_slide = blank
            self.working_slide  = blank.copy()
            self.live_stroke_pts.clear()

    # ── Undo / redo primitives ─────────────────────────────────────────── #

    def _push_undo(self) -> None:
        """Snapshot current working_slide onto the undo stack (bounded)."""
        if self.working_slide is not None:
            self.undo_stack.append(self.working_slide.copy())
            if len(self.undo_stack) > MAX_UNDO_STEPS:
                self.undo_stack.pop(0)

    def commit_stroke(self) -> None:
        """
        Call on pinch release (drawing stroke finished).

        1. Saves a pre-stroke snapshot (the current working_slide before the
           new stroke) so undo can restore to exactly before this stroke.
        2. Bakes live_stroke_pts permanently into working_slide.
        3. Clears live_stroke_pts.
        4. Discards redo history — a new stroke forks the timeline.
        """
        if not self.live_stroke_pts:
            return                      # nothing to commit

        self._push_undo()
        self.redo_stack.clear()
        self._bake_live_stroke(self.working_slide)
        self.live_stroke_pts.clear()

    def undo(self) -> bool:
        """
        Restore working_slide to the state before the last committed stroke.
        Returns True if an undo was available, False if the stack was empty.
        """
        if not self.undo_stack:
            return False
        if self.working_slide is not None:
            self.redo_stack.append(self.working_slide.copy())
        self.working_slide     = self.undo_stack.pop()
        self.live_stroke_pts.clear()
        return True

    def redo(self) -> bool:
        """
        Re-apply the most recently undone stroke.
        Returns True if a redo was available, False if the stack was empty.
        """
        if not self.redo_stack:
            return False
        if self.working_slide is not None:
            self.undo_stack.append(self.working_slide.copy())
        self.working_slide     = self.redo_stack.pop()
        self.live_stroke_pts.clear()
        return True

    # ── Canvas management ──────────────────────────────────────────────── #

    def clear(self) -> None:
        """
        Reset working_slide back to original_slide (wipes all annotations).
        The pre-clear state is pushed to the undo stack so this is undoable.
        """
        self._push_undo()
        self.redo_stack.clear()
        if self.original_slide is not None:
            self.working_slide = self.original_slide.copy()
        elif self.working_slide is not None:
            self.working_slide[:] = 0
        self.live_stroke_pts.clear()
        self.history.clear()

    def reset_stroke(self) -> None:
        """
        Call when leaving DRAWING mode to discard any in-progress stroke
        without committing it.  Does NOT touch the undo stack.
        """
        self.live_stroke_pts.clear()

    # ── Live-stroke rendering ──────────────────────────────────────────── #

    def _bake_live_stroke(self, target: np.ndarray) -> None:
        """
        Render self.live_stroke_pts as a polyline onto `target` in-place.
        `target` is typically self.working_slide (commit) or a temp copy
        (per-frame preview).
        """
        if len(self.live_stroke_pts) < 2:
            # A single point: draw a filled circle so even a tap is visible.
            if self.live_stroke_pts:
                cv2.circle(
                    target,
                    self.live_stroke_pts[0],
                    CFG.draw_thickness // 2,
                    CFG.draw_color,
                    -1,
                    cv2.LINE_AA,
                )
            return

        pts = np.array(self.live_stroke_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            target,
            [pts],
            isClosed=False,
            color=CFG.draw_color,
            thickness=CFG.draw_thickness,
            lineType=cv2.LINE_AA,
        )

    def get_display_slide(self) -> Optional[np.ndarray]:
        """
        Return the frame-ready BGR slide:
          • If a live stroke is in progress, overlay it on a copy of
            working_slide (no mutation).
          • Otherwise return working_slide directly (zero-copy fast path).
        """
        if self.working_slide is None:
            return None

        if self.live_stroke_pts:
            display = self.working_slide.copy()
            self._bake_live_stroke(display)
            return display

        return self.working_slide          # no copy needed


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


# ─────────────────────────── Accumulate live stroke ───────────────────────── #

def _accumulate_stroke_point(state: PointerDrawingState, pt: tuple,
                              smoothing_window: int, use_dead_zone: bool) -> None:
    """
    Smooth `pt` and append to state.live_stroke_pts when outside the dead zone.

    Parameters
    ----------
    smoothing_window : pass 1 to skip rolling average (cursor already EMA-
                       smoothed upstream).
    use_dead_zone    : pass False to skip the dead-zone check (same reason).
    """
    smooth_pt = _smooth_point(state, pt, smoothing_window)

    prev = state.live_stroke_pts[-1] if state.live_stroke_pts else None
    if (not use_dead_zone) or _outside_dead_zone(smooth_pt, prev, CFG.dead_zone_px):
        state.live_stroke_pts.append(smooth_pt)


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


# ─────────────────────────── HUD overlays ─────────────────────────────────── #
def draw_exit_progress_bar(frame: np.ndarray, progress: float, label: str) -> np.ndarray:
    """
    Render a semi-transparent 'Exiting...' progress bar at the bottom of the
    frame.  `progress` is a float in [0.0, 1.0] representing hold completion.
    """
    h, w = frame.shape[:2]

    bar_h      = 36
    bar_w      = 320
    x1         = (w - bar_w) // 2
    y1         = h - bar_h - 16
    x2         = x1 + bar_w
    y2         = y1 + bar_h
    fill_x2    = x1 + int(bar_w * progress)
    corner_r   = 8

    # Background pill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Filled progress region (amber → red as it fills)
    r = int(50  + 205 * progress)
    g = int(180 - 180 * progress)
    b = 0
    cv2.rectangle(frame, (x1, y1), (fill_x2, y2), (b, g, r), -1)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1, cv2.LINE_AA)

    # Label
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    thickness  = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = y1 + (bar_h + th) // 2
    cv2.putText(frame, label, (tx, ty), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

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
    Wipe all drawing strokes, restoring working_slide to original_slide.
    The pre-clear state is pushed to the undo stack so the clear is undoable.

    width / height are accepted for backward compatibility but ignored —
    the slide dimensions are already known from set_base_slide().
    """
    drawing_state = _resolve_state(state)
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
    frame     : BGR slide frame that will be annotated and returned.
                In the baked-slide model this is typically the raw slide
                image (before HUD text is composited), but the function
                works regardless — it starts from the slide's working_slide
                or falls back to `frame` if set_base_slide() was not called.
    landmarks : MediaPipe hand landmarks for the detected hand, or None.
    gesture   : "POINTER" | "DRAWING" | "CLEAR" | anything else.
    state     : Per-slide PointerDrawingState; uses a module-level default
                if None (backward-compatible).
    cursor_px : Optional pre-smoothed fingertip position (x, y) in frame
                pixel coordinates, supplied by main.py's CursorSmoother.
                When provided:
                  • _index_fingertip() is NOT called.
                  • Internal rolling-average window is set to 1 (identity).
                  • Dead-zone check is skipped.
                When None the original behaviour is preserved.

    Returns
    -------
    frame : BGR frame with pointer / drawing annotations composited on top.
            In DRAWING mode this is get_display_slide() (working_slide with
            the live stroke overlaid) plus the crosshair and HUD badge.
            In POINTER mode this is the working_slide with a laser dot.
            The HUD text written by main.py is NOT included here and should
            be added after this call returns.

    Notes
    -----
    The caller is responsible for calling state.set_base_slide(slide) once
    whenever the displayed slide changes, and state.commit_stroke() on pinch
    release.  pointer_drawing_overlay() accumulates live_stroke_pts during a
    DRAWING frame but never commits them itself.
    """
    drawing_state = _resolve_state(state)
    h, w = frame.shape[:2]
    drawing_state.ensure_slide(h, w)

    pre_smoothed: bool = cursor_px is not None

    # ── Break drawing stroke if gesture changed away from DRAWING ──────── #
    if gesture != "DRAWING" and drawing_state.last_gesture == "DRAWING":
        drawing_state.commit_stroke()
    drawing_state.last_gesture = gesture

    # ── CLEAR ─────────────────────────────────────────────────────────── #
    if gesture == "CLEAR":
        drawing_state.clear()
        out = drawing_state.get_display_slide() or frame
        return out

    # ── Start from baked working_slide (with live stroke if any) ─────── #
    # This replaces the old _composite_canvas() call.  For most frames the
    # fast path is taken: no copy, no alpha multiply — just use working_slide.
    out = drawing_state.get_display_slide()

    if out is None:
        out = frame.copy()
    else:
        # Always copy while drawing so temporary overlays
        # (crosshair, badge, etc.) aren't baked into working_slide.
        if gesture == "DRAWING":
            out = out.copy()
        else:
            out = out.copy()

    # ── Resolve cursor position ───────────────────────────────────────── #
    if pre_smoothed:
        pt = cursor_px
    else:
        pt = _index_fingertip(landmarks, w, h)

    if pt is None:
        return out

    # ── LASER POINTER ─────────────────────────────────────────────────── #
    if gesture == "POINTER":
        drawing_state.history.clear()
        out = _render_laser(out, pt)

    # ── DRAWING ───────────────────────────────────────────────────────── #
    elif gesture == "DRAWING":
        smoothing_window = 1 if pre_smoothed else CFG.draw_smoothing
        _accumulate_stroke_point(
            drawing_state, pt,
            smoothing_window=smoothing_window,
            use_dead_zone=not pre_smoothed,
        )
        # out already includes the live stroke via get_display_slide().
        # We only need to add the crosshair cursor and HUD badge on top.
        _draw_crosshair(out, pt, CFG.draw_color)

    return out