# Gesture-Based Presentation Controller

A computer vision web application that controls presentation slides using hand gestures through a webcam.

## Features

- Swipe right → next slide
- Swipe left → previous slide
- Hand tracking using MediaPipe
- Gesture-controlled slide navigation
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
- q → quit

## Training

See [training/README.md](training/README.md) for the HaGRID preprocessing and TF.js export workflow.
