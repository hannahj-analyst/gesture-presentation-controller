from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from training.constants import (
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_LABELS,
    FEATURE_COLUMNS,
    NUM_LANDMARKS,
)


def normalize_landmarks(
    raw_landmarks: np.ndarray,
    scale: bool = False,
) -> np.ndarray:
    points = np.asarray(raw_landmarks, dtype=np.float32)

    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(
            "Expected 21 landmarks with x, y, z coordinates each"
        )

    normalized = points - points[0]

    if scale:
        max_distance = float(np.linalg.norm(normalized, axis=1).max())
        if max_distance > 0:
            normalized = normalized / max_distance

    return normalized


def extract_landmark_features(
    image_path: Path,
    hands,
    scale: bool = False,
) -> np.ndarray | None:
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    raw_landmarks = np.array(
        [
            (landmark.x, landmark.y, landmark.z)
            for landmark in hand_landmarks.landmark
        ],
        dtype=np.float32,
    )

    normalized = normalize_landmarks(raw_landmarks, scale=scale)
    return normalized.reshape(-1)


def iter_labeled_images(
    dataset_root: Path,
    labels: Sequence[str] = DEFAULT_LABELS,
    extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> Iterable[tuple[str, Path]]:
    root = Path(dataset_root)

    for label in labels:
        label_dir = root / label

        if not label_dir.exists():
            continue

        for image_path in sorted(label_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in extensions:
                yield label, image_path


def build_landmark_dataframe(
    dataset_root: Path,
    labels: Sequence[str] = DEFAULT_LABELS,
    scale: bool = False,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.7,
) -> pd.DataFrame:
    root = Path(dataset_root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    hands_api = mp.solutions.hands
    hands = hands_api.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    records: list[dict[str, object]] = []

    try:
        for label, image_path in iter_labeled_images(root, labels=labels):
            features = extract_landmark_features(
                image_path=image_path,
                hands=hands,
                scale=scale,
            )

            if features is None:
                continue

            record: dict[str, object] = {
                "label": label,
                "source_path": str(image_path),
            }

            record.update(
                {
                    column: float(value)
                    for column, value in zip(FEATURE_COLUMNS, features)
                }
            )
            records.append(record)
    finally:
        hands.close()

    if not records:
        raise ValueError(
            "No hand landmarks were extracted. Check the dataset layout and labels."
        )

    return pd.DataFrame.from_records(
        records,
        columns=("label", "source_path", *FEATURE_COLUMNS),
    )
