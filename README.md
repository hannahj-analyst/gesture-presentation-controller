# Gesture-Based Presentation Controller

A computer vision application that controls presentation slides using hand gestures through a webcam. The system uses OpenCV for webcam and slide display, MediaPipe for hand landmark detection, and a custom-trained Keras gesture recognition model for static hand gesture classification.

## Project Overview

The Gesture-Based Presentation Controller allows a presenter to control slides without using a keyboard, mouse, or physical clicker. The webcam captures the presenter's hand movements and gestures in real time. The system then processes the video feed, detects hand landmarks, recognizes gestures, and maps those gestures to presentation actions.

The application supports slide navigation, pointer mode, drawing mode, annotation clearing, zoom mode, and gesture-based mode switching.

## Features

* Swipe right → next slide
* Swipe left → previous slide
* Hand tracking using MediaPipe
* Gesture-controlled slide navigation
* Hold `one` for 2 seconds → pointer mode
* Hold `ok` for 2 seconds → drawing mode
* Hold `peace` for 2 seconds → zoom mode
* Hold `stop` for 2 seconds → clear annotations
* Hold `fist` for 2 seconds → exit active mode
* Drawing mode supports undo and redo using swipe gestures
* OpenCV-based webcam and slide display
* Custom-trained gesture recognition model
* Model evaluation metrics and confusion matrix included

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

## Computer Vision Capabilities

This project integrates multiple computer vision tasks:

1. Real-time video processing using webcam input
2. Hand landmark detection using MediaPipe
3. Hand tracking across frames
4. Motion-based swipe detection
5. Static gesture recognition using a custom-trained Keras model
6. Pointer and drawing interaction using fingertip position
7. Zoom interaction using hand landmark distance

## Current Progress

✅ Webcam input and OpenCV frame processing
✅ MediaPipe hand landmark detection
✅ Swipe-based slide navigation
✅ Slide controller
✅ Trained custom gesture recognition model
✅ Pointer mode
✅ Drawing mode with undo/redo
✅ Clear annotation gesture
✅ Zoom mode
✅ Model evaluation metrics and confusion matrix

## Folder Structure

```text
gesture-presentation-controller/
  app/
    gesture_slide_controller.py
    main.py
    test_slide_controller.py

  assets/
    slides/
      slide1.png
      slide2.jpg
      slide3.jpg

  training/
    artifacts/
      keras_model.keras
      labels.json
      metadata.json
      metrics.json
      confusion_matrix.png
      processed_landmarks.csv
      tfjs_model/

    models/
      hand_landmarker.task

    constants.py
    preprocess.py
    train.py
    README.md

  utils/
    slide_controller.py
    swipe_detection.py

  pointer_drawing.py
  requirements.txt
  README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/joel-cdev/gesture-presentation-controller.git
cd gesture-presentation-controller
```

### 2. Create and activate a virtual environment

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python app\gesture_slide_controller.py
```

Two windows should open:

* Gesture Webcam
* Gesture Controlled Slides

## Controls

* Swipe right → next slide
* Swipe left → previous slide
* Hold `one` for 2 seconds → pointer mode
* Hold `ok` for 2 seconds → drawing mode
* Hold `peace` for 2 seconds → zoom mode
* Hold `stop` for 2 seconds → clear annotations
* Hold `fist` for 2 seconds → exit active mode
* Press `q` → quit

## Gesture Model

The custom gesture recognition model is stored in:

```text
training/artifacts/keras_model.keras
```

The model uses 63 input features from MediaPipe hand landmarks:

```text
21 landmarks × 3 coordinates = 63 features
```

The trained gesture classes are:

* one
* peace
* stop
* ok
* fist

The class labels are stored in:

```text
training/artifacts/labels.json
```

## Model Results

The trained model evaluation results are stored in:

```text
training/artifacts/metrics.json
```

The final model achieved approximately 94.63% test accuracy.

The confusion matrix image is stored in:

```text
training/artifacts/confusion_matrix.png
```

## Training

See the training documentation:

```text
training/README.md
```

The training pipeline extracts MediaPipe hand landmarks from labeled gesture images and trains a Keras neural network classifier.

## Notes

For best performance:

* Use good lighting.
* Keep your hand clearly visible to the webcam.
* Stand about 2–3 feet from the camera.
* Use clear and steady gestures.
* Avoid cluttered backgrounds when possible.

## Repository

GitHub Repository:

```text
https://github.com/joel-cdev/gesture-presentation-controller
```
