import cv2
import time
import sys
import os
import mediapipe as mp
from pathlib import Path

# Allow importing from parent folder
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from utils.slide_controller import SlideController


MODEL_PATH = Path(__file__).resolve().parent.parent / "training" / "models" / "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Hand Landmarker model not found: {MODEL_PATH}")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

hands = HandLandmarker.create_from_options(options)

# Webcam setup
cap = cv2.VideoCapture(0)

# Optional: reduce webcam resolution for better speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Slide controller setup
controller = SlideController()

# Improved swipe detection variables
x_positions = []
last_swipe_time = 0

cooldown = 0.8
threshold = 90
history_length = 6

last_action_text = "No action yet"

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = hands.detect(mp_image)

    # Get current slide
    slide = controller.get_current_slide()
    slide = cv2.resize(slide, (900, 600))

    # Display slide number
    slide_text = (
        f"Slide {controller.get_slide_number()} / "
        f"{controller.get_total_slides()}"
    )

    cv2.putText(
        slide,
        slide_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        slide,
        f"Last Action: {last_action_text}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:

            wrist = hand_landmarks.landmark[0]
            current_x = int(wrist.x * width)
            current_y = int(wrist.y * height)

            cv2.circle(
                frame,
                (current_x, current_y),
                10,
                (0, 255, 0),
                -1
            )

            # Store recent x positions
            x_positions.append(current_x)

            if len(x_positions) > history_length:
                x_positions.pop(0)

            # Detect swipe using movement over recent frames
            if len(x_positions) == history_length:
                movement = x_positions[-1] - x_positions[0]
                current_time = time.time()

                if current_time - last_swipe_time > cooldown:

                    if movement > threshold:
                        controller.next_slide()
                        last_action_text = "Swipe Right -> Next Slide"
                        print(last_action_text)
                        last_swipe_time = current_time
                        x_positions.clear()

                    elif movement < -threshold:
                        controller.previous_slide()
                        last_action_text = "Swipe Left -> Previous Slide"
                        print(last_action_text)
                        last_swipe_time = current_time
                        x_positions.clear()

    else:
        # Clear history if hand is not visible
        x_positions.clear()

    cv2.putText(
        frame,
        "Swipe Right = Next | Swipe Left = Previous | q = Quit",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    webcam_display = cv2.resize(frame, (450, 300))

    cv2.imshow("Gesture Webcam", webcam_display)
    cv2.imshow("Gesture Controlled Slides", slide)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quit")
        break

cap.release()
cv2.destroyAllWindows()
hands.close()