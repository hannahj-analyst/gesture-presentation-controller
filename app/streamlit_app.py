import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "training" / "artifacts" / "metrics.json"
LABELS_PATH = ROOT / "training" / "artifacts" / "labels.json"
CONFUSION_MATRIX_PATH = ROOT / "training" / "artifacts" / "confusion_matrix.png"
OPEN_CV_APP_PATH = ROOT / "app" / "gesture_slide_controller.py"


st.set_page_config(
    page_title="Gesture-Based Presentation Controller",
    page_icon="🖐️",
    layout="wide",
)


st.title("Gesture-Based Presentation Controller")
st.write(
    "A computer vision application that controls presentation slides using hand gestures. "
    "The project uses OpenCV, MediaPipe, and a custom-trained Keras gesture recognition model."
)


st.divider()


col1, col2 = st.columns([1, 1])

with col1:
    st.header("Project Controls")

    st.markdown(
        """
        - Swipe right → next slide
        - Swipe left → previous slide
        - Hold `one` for 2 seconds → pointer mode
        - Hold `ok` for 2 seconds → drawing mode
        - Hold `peace` for 2 seconds → zoom mode
        - Hold `stop` for 2 seconds → clear annotations
        - Hold `fist` for 2 seconds → exit active mode
        - Press `q`, `Q`, or `Esc` → quit the OpenCV app
        """
    )

with col2:
    st.header("Computer Vision Tasks")

    st.markdown(
        """
        This project includes several computer vision tasks:

        1. Real-time webcam video processing
        2. MediaPipe hand landmark detection
        3. Hand tracking across frames
        4. Swipe motion detection
        5. Static gesture classification with a custom-trained model
        6. Fingertip-based pointer and drawing interaction
        7. Pinch/spread-based zoom interaction
        """
    )


st.divider()


st.header("Run the Live Gesture Controller")

st.write(
    "Click the button below to launch the working OpenCV gesture controller. "
    "The webcam and slide windows will open outside the browser."
)

if st.button("Launch OpenCV Gesture Controller"):
    if OPEN_CV_APP_PATH.exists():
        subprocess.Popen([sys.executable, str(OPEN_CV_APP_PATH)])
        st.success("Gesture controller launched. Check for the OpenCV webcam and slide windows.")
    else:
        st.error(f"Could not find app file: {OPEN_CV_APP_PATH}")


st.divider()


st.header("Model Information")

if LABELS_PATH.exists():
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    st.write("Gesture classes:")
    st.write(", ".join(labels))
else:
    st.warning("labels.json was not found.")


if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("Train Samples", metrics.get("train_samples", "N/A"))

    with metric_col2:
        st.metric("Validation Samples", metrics.get("validation_samples", "N/A"))

    with metric_col3:
        test_accuracy = metrics.get("test_accuracy", None)
        if test_accuracy is not None:
            st.metric("Test Accuracy", f"{test_accuracy * 100:.2f}%")
        else:
            st.metric("Test Accuracy", "N/A")

    st.subheader("Classification Report")

    report = metrics.get("classification_report", {})

    rows = []

    for label, values in report.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "Class": label,
                    "Precision": values.get("precision"),
                    "Recall": values.get("recall"),
                    "F1-score": values.get("f1-score"),
                    "Support": values.get("support"),
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No classification report found in metrics.json.")

else:
    st.warning("metrics.json was not found.")


if CONFUSION_MATRIX_PATH.exists():
    st.subheader("Confusion Matrix")
    st.image(str(CONFUSION_MATRIX_PATH), caption="Gesture Classification Confusion Matrix")
else:
    st.warning("confusion_matrix.png was not found.")


st.divider()


st.header("How to Use")

st.markdown(
    """
    1. Start the Streamlit app.
    2. Click `Launch OpenCV Gesture Controller`.
    3. Allow the webcam window to open.
    4. Use hand gestures to control the slides.
    5. Press `q`, `Q`, or `Esc` in the OpenCV window to quit.
    """
)


st.info(
    "Note: The Streamlit page serves as the web interface for the project, while the real-time gesture "
    "controller runs through OpenCV windows for low-latency webcam interaction."
)