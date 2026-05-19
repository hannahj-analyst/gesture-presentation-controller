import cv2
import sys
import os

# Allow importing from parent folder
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from utils.slide_controller import SlideController


# Create slide controller object
controller = SlideController()


while True:

    # Get current slide
    slide = controller.get_current_slide()

    # Resize slide window
    slide = cv2.resize(slide, (900, 600))

    # Slide number text
    slide_text = (
        f"Slide {controller.get_slide_number()} / "
        f"{controller.get_total_slides()}"
    )

    # Display slide number
    cv2.putText(
        slide,
        slide_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Display controls
    cv2.putText(
        slide,
        "Right arrow / n = next | Left arrow / p = previous | q = quit",
        (30, 580),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # Show slide
    cv2.imshow(
        "Slide Controller Test",
        slide
    )

    # Wait for keyboard input
    key = cv2.waitKeyEx(100)

    # Next slide
    if key == ord("n") or key == 2555904:
        controller.next_slide()
        print(f"Next Slide -> {controller.get_slide_number()}")

    # Previous slide
    elif key == ord("p") or key == 2424832:
        controller.previous_slide()
        print(f"Previous Slide -> {controller.get_slide_number()}")

    # Quit
    elif key == ord("q"):
        print("Quit")
        break


cv2.destroyAllWindows()