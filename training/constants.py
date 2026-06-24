from __future__ import annotations

DEFAULT_LABELS = ("one", "peace", "stop", "ok")
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 3
FEATURE_DIMENSION = NUM_LANDMARKS * FEATURES_PER_LANDMARK
FEATURE_COLUMNS = tuple(
    f"feature_{index:02d}" for index in range(FEATURE_DIMENSION)
)
DEFAULT_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
)
