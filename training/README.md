# Training Pipeline

This folder contains the HaGRID preprocessing and model export workflow for the gesture classifier.

## Expected dataset layout

Organize images into one folder per label:

```text
training/data/
  one/
  peace/
  stop/
  ok/
```

If your HaGRID export uses different names, map or symlink the images into these label folders before training so the exported class order stays stable.

## Install

From the repository root:

```bash
pip install -r training/requirements.txt
```

## Train and export

```bash
python -m training.train --dataset-root training/data --output-dir training/artifacts
```

The command writes:

- `training/artifacts/processed_landmarks.csv`
- `training/artifacts/keras_model.keras`
- `training/artifacts/tfjs_model/model.json`
- `training/artifacts/tfjs_model/labels.json`
- `training/artifacts/metrics.json`

The normalized model input is always 63 values per hand, in wrist-relative landmark order.
