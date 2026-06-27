# Training Pipeline

This folder contains the preprocessing, training, evaluation, and export workflow for the custom hand gesture classifier used in the Gesture-Based Presentation Controller.

The classifier is trained using MediaPipe hand landmarks extracted from labeled hand gesture images. Each hand is represented by 63 numerical features:

```text
21 hand landmarks × 3 coordinates = 63 features
```

The trained model is used by the main application to classify static hand gestures and map them to presentation control actions.

## Gesture Classes

The final trained gesture classes are:

* one
* peace
* stop
* ok
* fist

These labels must match the order in:

```text
training/artifacts/labels.json
```

## Expected Dataset Layout

Organize images into one folder per label:

```text
training/data/
  one/
  peace/
  stop/
  ok/
  fist/
```

Each folder should contain images of that gesture. The folder names are used as class labels during preprocessing and training.

## Install Training Requirements

From the repository root:

```bash
pip install -r training/requirements.txt
```

## Train and Export

Run this command from the repository root:

```bash
python -m training.train --dataset-root training/data --output-dir training/artifacts
```

The command writes the trained model and evaluation files to:

```text
training/artifacts/
```

Expected outputs:

```text
training/artifacts/processed_landmarks.csv
training/artifacts/keras_model.keras
training/artifacts/labels.json
training/artifacts/metadata.json
training/artifacts/metrics.json
training/artifacts/confusion_matrix.png
training/artifacts/tfjs_model/model.json
training/artifacts/tfjs_model/labels.json
```

## Model Architecture

The gesture classifier is a Keras neural network that uses MediaPipe hand landmark features as input.

Input:

```text
63 features
```

Model structure:

```text
Input layer
Dense layer with 128 units and ReLU activation
Dropout layer
Dense layer with 64 units and ReLU activation
Dense output layer with softmax activation
```

Output:

```text
5 gesture classes
```

## Model Results

The final trained model uses the following classes:

```text
one, peace, stop, ok, fist
```

The saved metrics file reports:

```text
Train samples: 3421
Validation samples: 428
Test samples: 428
Test accuracy: approximately 94.63%
```

The confusion matrix is saved as:

```text
training/artifacts/confusion_matrix.png
```

The full metrics are saved as:

```text
training/artifacts/metrics.json
```

## How the Model Is Used in the App

The main application loads:

```text
training/artifacts/keras_model.keras
training/artifacts/labels.json
```

The app detects hand landmarks using MediaPipe, normalizes the landmark coordinates, passes the 63 features into the trained Keras model, and receives a predicted gesture label.

The gesture label is then mapped to a presentation action:

```text
one   → pointer mode
ok    → drawing mode
peace → zoom mode
stop  → clear annotations
fist  → exit active mode
```

## Important Notes

The order of labels in `labels.json` must match the class order used during model training. If the order is changed, the application may assign the wrong action to a gesture.

The trained model, labels, metadata, metrics, and confusion matrix should be kept in the `training/artifacts/` folder for final submission.
