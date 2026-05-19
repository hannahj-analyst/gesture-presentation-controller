import cv2
import mediapipe as mp
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

# Variables for swipe detection
previous_x = None
last_swipe_time = 0

cooldown = 1.0
threshold = 120

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam")
        break

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    height, width, _ = frame.shape

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand detection
    results = hands.process(rgb_frame)

    swipe_text = ""

    # If hand is detected
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Use wrist landmark
            wrist = hand_landmarks.landmark[0]

            current_x = int(wrist.x * width)
            current_y = int(wrist.y * height)

            # Draw tracking circle
            cv2.circle(
                frame,
                (current_x, current_y),
                10,
                (0, 255, 0),
                -1
            )

            # Swipe detection
            if previous_x is not None:

                movement = current_x - previous_x

                current_time = time.time()

                # Cooldown to prevent multiple swipes
                if current_time - last_swipe_time > cooldown:

                    # Swipe Right
                    if movement > threshold:
                        swipe_text = "Swipe Right"

                        print("Swipe Right")

                        last_swipe_time = current_time

                    # Swipe Left
                    elif movement < -threshold:
                        swipe_text = "Swipe Left"

                        print("Swipe Left")

                        last_swipe_time = current_time

            previous_x = current_x

    # Display swipe text
    cv2.putText(
        frame,
        swipe_text,
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        3
    )

    # Show webcam window
    cv2.imshow(
        "Swipe Detection Prototype",
        frame
    )

    # Quit with q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()