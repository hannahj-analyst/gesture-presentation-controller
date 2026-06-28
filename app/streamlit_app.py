from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "app" / "gesture_slide_controller.py"
SLIDES_DIR = ROOT / "assets" / "slides"
METRICS_PATH = ROOT / "training" / "artifacts" / "metrics.json"
LABELS_PATH = ROOT / "training" / "artifacts" / "labels.json"
CONFUSION_MATRIX_PATH = ROOT / "training" / "artifacts" / "confusion_matrix.png"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, .stApp {
            background:
                radial-gradient(circle at top left, rgba(251, 191, 36, 0.12), transparent 18%),
                radial-gradient(circle at 86% 14%, rgba(34, 211, 238, 0.14), transparent 24%),
                radial-gradient(circle at 35% 82%, rgba(52, 211, 153, 0.10), transparent 22%),
                linear-gradient(180deg, #07111f 0%, #050911 100%);
            color: #f5f8ff;
            font-family: 'Manrope', sans-serif;
        }

        #MainMenu, footer, header { visibility: hidden; }
        .block-container {
            padding: 1rem 1.2rem 1.1rem;
            max-width: 100%;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #091224 0%, #07101c 100%);
            border-right: 1px solid rgba(34, 211, 238, 0.12);
        }

        .hero {
            padding: 1.1rem 1.15rem;
            border-radius: 22px;
            border: 1px solid rgba(251, 191, 36, 0.18);
            background:
                linear-gradient(135deg, rgba(13, 23, 40, 0.98), rgba(7, 15, 28, 0.98)),
                radial-gradient(circle at 18% 20%, rgba(34, 211, 238, 0.12), transparent 30%),
                radial-gradient(circle at 86% 12%, rgba(251, 191, 36, 0.12), transparent 22%);
            box-shadow: 0 24px 56px rgba(0, 0, 0, 0.28);
            overflow: hidden;
            position: relative;
            margin-bottom: 1rem;
        }
        .hero::before, .hero::after {
            content: "";
            position: absolute;
            border-radius: 999px;
            filter: blur(32px);
            pointer-events: none;
        }
        .hero::before {
            width: 220px;
            height: 220px;
            right: -70px;
            top: -76px;
            background: rgba(251, 191, 36, 0.14);
        }
        .hero::after {
            width: 180px;
            height: 180px;
            left: 28%;
            bottom: -90px;
            background: rgba(34, 211, 238, 0.12);
        }
        .hero-kicker {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #fbbf24;
            margin-bottom: 0.45rem;
            position: relative;
            z-index: 1;
        }
        .hero h1 {
            margin: 0;
            font-family: 'Sora', sans-serif;
            font-size: clamp(2rem, 3vw, 3.1rem);
            line-height: 1.03;
            letter-spacing: -0.05em;
            position: relative;
            z-index: 1;
        }
        .hero p {
            margin: 0.5rem 0 0;
            max-width: 76rem;
            color: #aac0df;
            line-height: 1.6;
            position: relative;
            z-index: 1;
        }

        .chip {
            display: inline-block;
            margin: 0.75rem 0.45rem 0 0;
            padding: 0.36rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(34, 211, 238, 0.18);
            background: rgba(10, 19, 33, 0.78);
            color: #eef4ff;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.16);
            position: relative;
            z-index: 1;
        }

        .panel {
            border-radius: 18px;
            border: 1px solid rgba(34, 211, 238, 0.14);
            background: linear-gradient(180deg, rgba(12, 23, 40, 0.98), rgba(7, 15, 28, 0.98));
            padding: 0.95rem;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(12px);
            overflow: hidden;
            position: relative;
        }
        .panel::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(34, 211, 238, 0.05), transparent 24%, transparent 76%, rgba(251, 191, 36, 0.04));
            pointer-events: none;
        }

        .section-title {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            font-family: 'IBM Plex Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.78rem;
            color: #d9e6f7;
            margin-bottom: 0.65rem;
        }
        .section-title::after {
            content: "";
            height: 1px;
            flex: 1;
            background: linear-gradient(90deg, rgba(251, 191, 36, 0.45), rgba(34, 211, 238, 0.5), transparent);
        }

        .slide-shell {
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(34, 211, 238, 0.14);
            background: #08111a;
            box-shadow: 0 24px 52px rgba(0, 0, 0, 0.32);
        }
        .slide-shell img {
            display: block;
            border-radius: 20px;
        }
        .slide-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            justify-content: space-between;
            margin-top: 0.75rem;
            color: #aac0df;
            font-size: 0.85rem;
        }
        .slide-pill {
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            border: 1px solid rgba(34, 211, 238, 0.16);
            background: rgba(10, 19, 33, 0.78);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            color: #eef4ff;
        }

        .legend-row, .hook-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 0.8rem;
            padding: 0.72rem 0.82rem;
            border-radius: 12px;
            border: 1px solid rgba(34, 211, 238, 0.12);
            background: linear-gradient(180deg, #0a1321, #07101a);
            margin-bottom: 0.48rem;
        }
        .legend-name, .hook-title {
            font-family: 'Sora', sans-serif;
            font-weight: 800;
            color: #f4f8ff;
        }
        .legend-desc, .hook-desc {
            color: #aac0df;
            font-size: 0.88rem;
        }
        .tag, .hook-tag {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .tag { color: #fbbf24; }
        .hook-tag { color: #22c55e; }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.6rem;
        }
        .status-card {
            padding: 0.72rem 0.8rem;
            border-radius: 12px;
            border: 1px solid rgba(34, 211, 238, 0.12);
            background: linear-gradient(180deg, #0a1321, #07101a);
        }
        .status-kicker {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.66rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #9fb1d3;
            margin-bottom: 0.25rem;
        }
        .status-value {
            font-weight: 800;
            color: #f4f8ff;
            font-size: 0.96rem;
        }
        .status-desc {
            color: #aac0df;
            font-size: 0.82rem;
            margin-top: 0.15rem;
            line-height: 1.4;
        }

        .log-item {
            padding: 0.52rem 0.68rem;
            margin-bottom: 0.4rem;
            border-radius: 10px;
            border: 1px solid rgba(34, 211, 238, 0.12);
            background: linear-gradient(180deg, #0a1321, #07101a);
            color: #d9e6f7;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
        }
        .empty-state {
            padding: 1rem;
            border-radius: 14px;
            border: 1px dashed rgba(251, 191, 36, 0.22);
            background: rgba(10, 19, 33, 0.8);
            color: #aac0df;
        }

        .stButton > button {
            border-radius: 12px !important;
            border: 1px solid rgba(34, 211, 238, 0.22) !important;
            background: linear-gradient(135deg, #17314f, #0c1728) !important;
            color: #eef4ff !important;
            font-family: 'IBM Plex Mono', monospace !important;
            text-transform: uppercase !important;
            font-size: 0.72rem !important;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.2);
        }
        .stButton > button:hover {
            border-color: rgba(251, 191, 36, 0.5) !important;
            transform: translateY(-1px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_slides(source: str, uploaded_files) -> list:
    if source == "Uploaded images" and uploaded_files:
        return list(uploaded_files)

    if SLIDES_DIR.exists():
        slides = [
            p for p in sorted(SLIDES_DIR.iterdir())
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        return slides

    return []


def slide_name(item) -> str:
    return getattr(item, "name", Path(str(item)).name)


def open_controller() -> None:
    if APP_PATH.exists():
        subprocess.Popen([sys.executable, str(APP_PATH)], cwd=str(ROOT))
        st.session_state["controller_launched"] = True
    else:
        st.session_state["controller_launched"] = False


def init_state() -> None:
    defaults = {
        "slide_index": 0,
        "zoom": 1.0,
        "camera_on": False,
        "slide_source": "Assets folder",
        "uploaded_slides": [],
        "gesture_label": "Waiting for signal",
        "gesture_log": [],
        "controller_launched": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log_event(message: str) -> None:
    st.session_state.gesture_log = [message] + st.session_state.gesture_log[:4]


def move_slide(delta: int) -> None:
    slides = st.session_state.slides
    if not slides:
        return
    st.session_state.slide_index = max(0, min(st.session_state.slide_index + delta, len(slides) - 1))
    log_event(f"Slide changed to {st.session_state.slide_index + 1}")


def main() -> None:
    st.set_page_config(
        page_title="Gesture-Based Presentation Controller",
        page_icon="🖐",
        layout="wide",
    )
    inject_css()
    init_state()

    with st.sidebar:
        st.markdown("## Controls")
        st.caption("Use this panel to load slides and test the demo.")

        st.session_state.slide_source = st.radio(
            "Slide source",
            ["Assets folder", "Uploaded images"],
            index=0 if st.session_state.slide_source == "Assets folder" else 1,
            label_visibility="collapsed",
        )
        uploaded = st.file_uploader(
            "Upload slide images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        st.session_state.uploaded_slides = uploaded or []
        st.session_state.slides = load_slides(st.session_state.slide_source, st.session_state.uploaded_slides)

        st.divider()
        if st.button("Previous slide", use_container_width=True):
            move_slide(-1)
        if st.button("Next slide", use_container_width=True):
            move_slide(1)

        jump_to = st.number_input(
            "Jump to slide",
            min_value=1,
            max_value=max(len(st.session_state.slides), 1),
            value=min(st.session_state.slide_index + 1, max(len(st.session_state.slides), 1)),
            step=1,
        )
        if st.button("Jump", use_container_width=True):
            if st.session_state.slides:
                st.session_state.slide_index = int(jump_to) - 1
                log_event(f"Jumped to slide {jump_to}")

        st.session_state.zoom = st.slider("Zoom", 1.0, 2.0, float(st.session_state.zoom), 0.05)
        st.session_state.camera_on = st.toggle("Show camera panel", value=st.session_state.camera_on)

        st.divider()
        st.markdown("## Demo actions")
        if st.button("One finger"):
            st.session_state.gesture_label = "One finger"
            log_event("Gesture: one finger")
        if st.button("Two fingers"):
            st.session_state.gesture_label = "Two fingers"
            log_event("Gesture: two fingers")
        if st.button("Open palm"):
            st.session_state.gesture_label = "Open palm"
            log_event("Gesture: open palm")
        if st.button("Reset demo"):
            st.session_state.slide_index = 0
            st.session_state.zoom = 1.0
            st.session_state.gesture_label = "Waiting for signal"
            st.session_state.gesture_log = []
            log_event("Demo reset")

        st.divider()
        if st.button("Launch live OpenCV controller", use_container_width=True):
            open_controller()
            if st.session_state.controller_launched:
                st.success("Controller launched. Check the OpenCV windows.")
            else:
                st.error(f"Could not find {APP_PATH}")

        st.divider()
        st.markdown("## Module hooks")
        st.caption("These are the places your teammates' code plugs in later.")
        st.markdown(
            """
            - `apply_hand_tracking(...)`
            - `apply_swipe(...)`
            - `apply_gesture_prediction(...)`
            - `apply_slide_command(...)`
            - `apply_pointer(...)`
            """
        )

    slides = st.session_state.slides
    current_slide = slides[st.session_state.slide_index] if slides else None
    slide_total = max(len(slides), 1)
    progress = 0.0 if not slides else (st.session_state.slide_index + 1) / len(slides)
    labels = read_json(LABELS_PATH) or []
    metrics = read_json(METRICS_PATH) or {}

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">CSCI435 computer vision project</div>
            <h1>Gesture-Based Presentation Controller</h1>
            <p>
                A live dashboard for slide control, gesture handling, and presentation demos.
                It is built to match the rest of the project and stay ready for the final handoff.
            </p>
            <div>
                <span class="chip">Deck: {st.session_state.slide_source}</span>
                <span class="chip">Slides: {len(slides)}</span>
                <span class="chip">Gesture: {st.session_state.gesture_label}</span>
                <span class="chip">UI ready for live inputs</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.65, 0.95], gap="medium")

    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-title">Presentation stage - slide {st.session_state.slide_index + 1} of {slide_total}</div>',
            unsafe_allow_html=True,
        )
        st.progress(progress)

        if current_slide is not None:
            st.markdown('<div class="slide-shell">', unsafe_allow_html=True)
            st.image(current_slide, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            current_name = slide_name(current_slide)
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <strong>No slides loaded.</strong><br>
                    Put slide images in <code>assets/slides</code> or upload images from the sidebar.
                </div>
                """,
                unsafe_allow_html=True,
            )
            current_name = "No file loaded"

        st.markdown(
            f"""
            <div class="slide-meta">
                <span class="slide-pill">{current_name}</span>
                <span class="slide-pill">Zoom {st.session_state.zoom:.2f}x</span>
                <span class="slide-pill">Manual demo mode</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 0.9])
        with btn1:
            if st.button("Previous", use_container_width=True, disabled=not slides):
                move_slide(-1)
                st.rerun()
        with btn2:
            if st.button("Reset", use_container_width=True):
                st.session_state.slide_index = 0
                st.session_state.zoom = 1.0
                st.session_state.gesture_label = "Waiting for signal"
                log_event("Demo reset")
                st.rerun()
        with btn3:
            if st.button("Next", use_container_width=True, disabled=not slides):
                move_slide(1)
                st.rerun()
        with btn4:
            if st.button("Clear log", use_container_width=True):
                st.session_state.gesture_log = []
                log_event("Log cleared")
                st.rerun()

    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Gesture legend</div>', unsafe_allow_html=True)
        for name, desc, tag in [
            ("Swipe right", "next slide", "nav"),
            ("Swipe left", "previous slide", "nav"),
            ("One finger", "laser pointer", "pointer"),
            ("Two fingers", "draw mode", "draw"),
            ("Open palm", "clear annotations", "reset"),
            ("Zoom gesture", "zoom in / out", "zoom"),
        ]:
            st.markdown(
                f"""
                <div class="legend-row">
                    <div>
                        <div class="legend-name">{name}</div>
                        <div class="legend-desc">{desc}</div>
                    </div>
                    <div class="tag">{tag}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height: 0.55rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Module hooks</div>', unsafe_allow_html=True)
        for title, desc, tag in [
            ("Hand tracking", "MediaPipe landmarks into the UI.", "apply_hand_tracking"),
            ("Swipe detector", "Move slides from hand movement.", "apply_swipe"),
            ("Gesture model", "Label + confidence from the classifier.", "apply_gesture_prediction"),
            ("Slide controller", "Jump / next / previous commands.", "apply_slide_command"),
            ("Pointer / draw", "Pointer coordinates and annotation points.", "apply_pointer"),
        ]:
            st.markdown(
                f"""
                <div class="hook-row">
                    <div>
                        <div class="hook-title">{title}</div>
                        <div class="hook-desc">{desc}</div>
                    </div>
                    <div class="hook-tag">{tag}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height: 0.55rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Live status</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="status-grid">
                <div class="status-card">
                    <div class="status-kicker">Camera</div>
                    <div class="status-value">{'On' if st.session_state.camera_on else 'Off'}</div>
                    <div class="status-desc">The camera area is ready for the webcam or MediaPipe feed.</div>
                </div>
                <div class="status-card">
                    <div class="status-kicker">Slides</div>
                    <div class="status-value">{len(slides)}</div>
                    <div class="status-desc">Loaded from assets or from uploaded slide images.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.gesture_log:
            st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)
            for item in st.session_state.gesture_log[:4]:
                st.markdown(f'<div class="log-item">{item}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="empty-state" style="margin-top:0.45rem;">No actions yet. Use the sidebar or demo buttons.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height: 0.55rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model info</div>', unsafe_allow_html=True)
        if metrics:
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Train", metrics.get("train_samples", "N/A"))
            with metric_cols[1]:
                st.metric("Val", metrics.get("validation_samples", "N/A"))
            with metric_cols[2]:
                test_accuracy = metrics.get("test_accuracy")
                st.metric("Test", f"{test_accuracy * 100:.1f}%" if isinstance(test_accuracy, (int, float)) else "N/A")
        else:
            st.caption("No metrics.json found.")
        if labels:
            st.caption("Gesture classes: " + ", ".join(labels))
        else:
            st.caption("No labels.json found.")

        if CONFUSION_MATRIX_PATH.exists():
            with st.expander("Confusion matrix", expanded=False):
                st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
