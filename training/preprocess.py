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


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


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
    hand_landmarker,
    scale: bool = False,
) -> np.ndarray | None:
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = hand_landmarker.detect(mp_image)

    if not results.hand_landmarks:
        return None

    hand_landmarks = results.hand_landmarks[0]
    raw_landmarks = np.array(
        [
            (landmark.x, landmark.y, landmark.z)
            for landmark in hand_landmarks
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
    model_path: Path | None = None,
    min_detection_confidence: float = 0.7,
    min_hand_presence_confidence: float = 0.7,
    min_tracking_confidence: float = 0.7,
) -> pd.DataFrame:
    root = Path(dataset_root)
    task_model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if not task_model_path.exists():
        raise FileNotFoundError(
            f"Hand Landmarker model does not exist: {task_model_path}"
        )

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_model_path)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    from tqdm import tqdm

    records: list[dict[str, object]] = []
    labeled_images = list(iter_labeled_images(root, labels=labels))

    with HandLandmarker.create_from_options(options) as hand_landmarker:
        progress = tqdm(labeled_images, desc="Extracting landmarks", unit="img")
        for label, image_path in progress:
            progress.set_postfix(label=label)
            features = extract_landmark_features(
                image_path=image_path,
                hand_landmarker=hand_landmarker,
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

    if not records:
        raise ValueError(
            "No hand landmarks were extracted. Check the dataset layout and labels."
        )

    return pd.DataFrame.from_records(
        records,
        columns=("label", "source_path", *FEATURE_COLUMNS),
    )
