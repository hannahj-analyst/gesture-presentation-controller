# Gesture-Based Presentation Controller

A computer vision web application that controls presentation slides using hand gestures through a webcam.

## Features

- Swipe right → next slide
- Swipe left → previous slide
- Hand tracking using MediaPipe
- Gesture-controlled slide navigation
- Hold `one` for 3 seconds → pointer mode on the slides
- Hold `peace` for 3 seconds → draw on the slides
- Hold `stop` for 3 seconds → clear current slide annotations
- OpenCV-based presentation display

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Streamlit

## Current Progress

✅ Hand tracking  
✅ Swipe detection  
✅ Slide controller  
✅ Gesture-controlled slide navigation  
✅ Training pipeline scaffold for HaGRID landmark classification  

## How to Run

Activate virtual environment:

```bash
.venv\Scripts\activate
```

Run application:

```bash
python app/gesture_slide_controller.py
```

## Controls

- Swipe right → next slide
- Swipe left → previous slide
- Hold `one` for 3 seconds → pointer mode
- Hold `peace` for 3 seconds → drawing mode
- Hold `stop` for 3 seconds → clear annotations
- q → quit

## Training

See [training/README.md](training/README.md) for the HaGRID preprocessing and TF.js export workflow.
