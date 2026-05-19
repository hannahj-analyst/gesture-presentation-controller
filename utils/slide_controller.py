import os
import cv2


class SlideController:
    def __init__(self):
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        self.slides_folder = os.path.join(
            base_dir,
            "assets",
            "slides"
        )

        self.slides = self.load_slides()

        self.current_slide_index = 0

    def load_slides(self):

        valid_extensions = (
            ".png",
            ".jpg",
            ".jpeg"
        )

        slides = [
            os.path.join(self.slides_folder, file)

            for file in os.listdir(self.slides_folder)

            if file.lower().endswith(valid_extensions)
        ]

        slides.sort()

        if not slides:
            raise FileNotFoundError(
                "No slide images found in assets/slides"
            )

        return slides

    def get_current_slide(self):

        slide_path = self.slides[self.current_slide_index]

        slide = cv2.imread(slide_path)

        if slide is None:
            raise FileNotFoundError(
                f"Could not load slide: {slide_path}"
            )

        return slide

    def next_slide(self):

        if self.current_slide_index < len(self.slides) - 1:
            self.current_slide_index += 1

    def previous_slide(self):

        if self.current_slide_index > 0:
            self.current_slide_index -= 1

    def get_slide_number(self):

        return self.current_slide_index + 1

    def get_total_slides(self):

        return len(self.slides)