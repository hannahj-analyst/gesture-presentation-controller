# Gesture-Based Presentation Controller

This project lets a presenter control slides with hand gestures instead of a mouse or keyboard.

## Project Idea

The system uses:

- OpenCV for the live camera controller
- MediaPipe for hand landmark detection
- a custom Keras model for gesture recognition
- Streamlit for the presentation dashboard

## Gesture Mapping

- Swipe right -> next slide
- Swipe left -> previous slide
- One finger -> pointer mode
- Two fingers -> drawing mode
- Open palm -> clear annotations
- Zoom gesture -> zoom in or out

## Repository Layout

```text
gesture-presentation-controller/
  app/
    gesture_slide_controller.py
    main.py
    streamlit_app.py

  assets/
    slides/

  training/
    artifacts/
    models/

  utils/
    slide_controller.py
    swipe_detection.py

  pointer_drawing.py
  requirements.txt
  README.md
```

## How To Run

### 1. Install the dashboard dependencies

```bash
python -m pip install -r requirements.txt
```

This installs the lightweight Streamlit dashboard only.

### 2. Run the Streamlit dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

The dashboard shows the slides, project info, and a button to launch the live controller.

### 3. Run the live OpenCV controller

For the webcam controller and MediaPipe-based gesture runtime, use Python 3.12 and install the full stack:

```bash
python -m pip install -r requirements-full.txt
```

```bash
python app/gesture_slide_controller.py
```

## Notes

- Put slide images in `assets/slides/`.
- The Streamlit dashboard is the front-end view for the project.
- The OpenCV controller is the part that actually reads the webcam and changes slides.
- If `streamlit` is not recognized, run it with `python -m streamlit ...` instead of `streamlit ...`.
