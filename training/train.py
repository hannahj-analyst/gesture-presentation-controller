from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from training.constants import DEFAULT_LABELS, FEATURE_COLUMNS, FEATURE_DIMENSION
from training.preprocess import build_landmark_dataframe


def build_model(input_dim: int, num_classes: int, learning_rate: float):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_tfjs_model(model, output_dir: Path) -> None:
    import tensorflowjs as tfjs

    output_dir.mkdir(parents=True, exist_ok=True)
    tfjs.converters.save_keras_model(model, str(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the gesture classifier from labeled hand images."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Folder containing one directory per label with gesture images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "artifacts",
        help="Where to write the trained Keras model, metrics, and TF.js export.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=list(DEFAULT_LABELS),
        help="Label order to train and export. Defaults to one peace stop ok.",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Scale normalized landmarks by the farthest wrist-relative landmark.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation fraction.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test fraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the dataset split.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.7,
        help="MediaPipe Hands detection confidence for preprocessing.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.7,
        help="MediaPipe Hands tracking confidence for preprocessing.",
    )

    return parser.parse_args()


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
):
    if not (0 < test_size < 1):
        raise ValueError("test-size must be between 0 and 1")

    if not (0 < val_size < 1):
        raise ValueError("val-size must be between 0 and 1")

    if test_size + val_size >= 1:
        raise ValueError("test-size + val-size must be less than 1")

    def can_stratify(targets: np.ndarray) -> bool:
        if targets.size == 0:
            return False

        counts = np.bincount(targets)
        return bool(counts.size) and int(counts.min()) >= 2

    temp_size = test_size + val_size
    first_stratify = labels if can_stratify(labels) else None

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=temp_size,
        random_state=seed,
        stratify=first_stratify,
    )

    relative_test_size = test_size / temp_size
    second_stratify = y_temp if can_stratify(y_temp) else None

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=seed,
        stratify=second_stratify,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def main() -> None:
    args = parse_args()
    label_order = tuple(args.labels) if args.labels else DEFAULT_LABELS

    output_dir = args.output_dir.expanduser().resolve()
    tfjs_dir = output_dir / "tfjs_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_landmark_dataframe(
        dataset_root=args.dataset_root,
        labels=label_order,
        scale=args.scale,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    dataset = dataset[dataset["label"].isin(label_order)].copy()

    if dataset.empty:
        raise ValueError("No samples matched the requested label order.")

    class_counts = dataset["label"].value_counts().reindex(label_order, fill_value=0)
    missing_labels = [label for label, count in class_counts.items() if count == 0]

    if missing_labels:
        raise ValueError(
            "Missing samples for labels: " + ", ".join(missing_labels)
        )

    if int(class_counts.min()) < 3:
        raise ValueError(
            "Need at least three samples per label before train/validation/test splitting."
        )

    dataset_csv = output_dir / "processed_landmarks.csv"
    dataset.to_csv(dataset_csv, index=False)

    label_to_index = {label: index for index, label in enumerate(label_order)}
    features = dataset.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    target = dataset["label"].map(label_to_index).to_numpy(dtype=np.int64)

    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        features=features,
        labels=target,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    import tensorflow as tf

    tf.random.set_seed(args.seed)

    model = build_model(
        input_dim=FEATURE_DIMENSION,
        num_classes=len(label_order),
        learning_rate=args.learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    test_predictions = model.predict(x_test, verbose=0).argmax(axis=1)

    report = classification_report(
        y_test,
        test_predictions,
        labels=list(range(len(label_order))),
        target_names=list(label_order),
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(
        y_test,
        test_predictions,
        labels=list(range(len(label_order))),
    )

    model_path = output_dir / "keras_model.keras"
    model.save(model_path)

    tfjs_dir.mkdir(parents=True, exist_ok=True)
    export_tfjs_model(model, tfjs_dir)

    labels_payload = list(label_order)
    save_json(output_dir / "labels.json", labels_payload)
    save_json(tfjs_dir / "labels.json", labels_payload)

    history_payload = {
        name: [float(value) for value in values]
        for name, values in history.history.items()
    }

    metrics_payload = {
        "labels": list(label_order),
        "feature_dimension": FEATURE_DIMENSION,
        "normalized_relative_to": "wrist",
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "class_counts": {label: int(count) for label, count in class_counts.items()},
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),
        "history": history_payload,
    }

    save_json(output_dir / "metrics.json", metrics_payload)
    save_json(
        output_dir / "metadata.json",
        {
            "labels": list(label_order),
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_dimension": FEATURE_DIMENSION,
            "normalized_relative_to": "wrist",
        },
    )

    confusion_frame = pd.DataFrame(
        confusion,
        index=[f"true_{label}" for label in label_order],
        columns=[f"pred_{label}" for label in label_order],
    )
    confusion_frame.to_csv(output_dir / "confusion_matrix.csv")

    print(f"Saved processed landmarks to {dataset_csv}")
    print(f"Saved Keras model to {model_path}")
    print(f"Saved TF.js model to {tfjs_dir}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
