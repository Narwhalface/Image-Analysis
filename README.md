# Image Analysis Coursework

This project extracts object crops from Azure Kinect recordings, lets you label them, trains a classifier, and runs live predictions on the testing video.

The repository is set up to work with these video files in the project root:

- `kinect-training-set.mkv`
- `kinect-testing-set.mkv`

## What This Project Does

The workflow is:

1. Extract object crops from the training video.
2. Show a preview image for each crop so it can be labeled by a human.
3. Train a classifier from the labeled crops.
4. Run the classifier live on the testing video.

The current pipeline uses:

- colour crop
- depth crop
- IR crop
- ORB features
- histogram/statistical features from colour, depth, and IR
- OpenCV linear SVM classifier

## Project Files

- `main.py`: video playback, object detection, crop extraction, live prediction
- `label_training_images.py`: interactive labeling tool
- `train_classifier.py`: training entry point
- `test_classifier.py`: batch prediction / evaluation entry point
- `training_data_massive/`: extracted and labeled training data
- `testing_data/`: optional extracted testing data

## Environment Setup

### Option 1: Conda

Create the environment from `environment.yml`:

```powershell
conda env create -f environment.yml
conda activate image-analysis
```

### Option 2: Virtual Environment

Create and activate a virtual environment, then install the requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If your PowerShell execution policy blocks activation, you can still run the scripts directly with:

```powershell
& ".venv\Scripts\python.exe" <script>
```

## Recommended Workflow

### 1. Extract Training Data

This command extracts training crops from `kinect-training-set.mkv` and saves them into `training_data_massive`.

Current recommended settings:

- minimum save gap: `3`
- crop change threshold: `5.0`

Run:

```powershell
& "c:/Dev/Image Analysis/.venv/Scripts/python.exe" "c:/Dev/Image Analysis/main.py" --dataset training --no-realtime-wait --ir --save-every-detection --output-dir "c:/Dev/Image Analysis/training_data_massive" --min-save-gap 3 --min-crop-change 5.0
```

What this does:

- reads the training MKV
- detects the main object using the depth stream
- saves a multimodal crop for training
- saves a matching preview image into `_label_previews/` for human labeling

### 2. Label the Training Images

Run:

```powershell
& "c:/Dev/Image Analysis/.venv/Scripts/python.exe" "c:/Dev/Image Analysis/label_training_images.py" "c:/Dev/Image Analysis/training_data_massive"
```

Controls:

- `N`: create a new label
- `L`: type an existing label name or number
- `1` to `9`: assign one of the first nine labels
- `R`: move the image to `Rejected`
- `S`: skip the image
- `Q` or `Esc`: quit

Notes:

- The labeler shows the preview image if available, not just the crop.
- Images moved to `Rejected` are ignored during training.
- `_label_previews/` is also ignored during training.

### 3. Train the Classifier

Run:

```powershell
& "c:/Dev/Image Analysis/.venv/Scripts/python.exe" "c:/Dev/Image Analysis/train_classifier.py" "c:/Dev/Image Analysis/training_data_massive" --no-visualise-preview --save-arrays --model-output "c:/Dev/Image Analysis/training_data_massive/classifier.yml" --report-dir "c:/Dev/Image Analysis/training_data_massive/reports"
```

This creates:

- `training_data_massive/classifier.yml`
- `training_data_massive/classifier.json`
- `training_data_massive/reports/confusion_matrix.png`
- `training_data_massive/reports/per_class_accuracy.png`
- saved NumPy arrays for features, labels, and splits

### 4. Run the Live Testing Video

Run:

```powershell
& "c:/Dev/Image Analysis/.venv/Scripts/python.exe" "c:/Dev/Image Analysis/main.py" --dataset testing --ir --classifier-model "c:/Dev/Image Analysis/training_data_massive/classifier.yml"
```

This opens the testing video and overlays:

- the detected bounding box
- the predicted object class

Quit with:

- `Q`
- `Esc`

## Optional: Batch Prediction on Saved Test Images

If you already have extracted test crops in `testing_data`, run:

```powershell
& "c:/Dev/Image Analysis/.venv/Scripts/python.exe" "c:/Dev/Image Analysis/test_classifier.py" "c:/Dev/Image Analysis/training_data_massive/classifier.yml" "c:/Dev/Image Analysis/testing_data" --predictions-output "c:/Dev/Image Analysis/testing_data/predictions.csv"
```

## Data Quality Guidance

For better results:

- keep labels consistent
- reject images that are too ambiguous to identify confidently
- collect enough examples per class
- avoid training on very noisy or partial crops unless they are realistic examples you expect at test time

If an image does not contain enough information to identify the object, move it to `Rejected` instead of forcing a class.

## Troubleshooting

### Bounding Box Looks Wrong

The current code uses the largest connected component from the depth mask. If the box still looks off, adjust the threshold values during playback and make sure the object is well separated in depth from the background.

### The Labeler Shows Only Crops

Make sure the data was extracted with the current version of `main.py`, which creates the `_label_previews/` folder alongside the crop images.

### The Live Viewer Predicts the Same Class Repeatedly

This usually means one of these:

- the training data is too imbalanced
- the model was trained on old data that does not match the current extraction format
- there are too few examples for some classes

If that happens, re-extract, relabel, and retrain using the current multimodal pipeline.

## Quick Start

If you only want the shortest working sequence, use these three commands in order:

```powershell
& "c:/Dev/Image Analysis/.venv/Scripts/python.exe" "c:/Dev/Image Analysis/main.py" --dataset training --no-realtime-wait --ir --save-every-detection --output-dir "c:/Dev/Image Analysis/training_data_massive" --min-save-gap 3 --min-crop-change 5.0
```

```powershell
& "c:/Dev/Image Analysis/.venv/Scripts/python.exe" "c:/Dev/Image Analysis/label_training_images.py" "c:/Dev/Image Analysis/training_data_massive"
```

```powershell
& "c:/Dev/Image Analysis/.venv/Scripts/python.exe" "c:/Dev/Image Analysis/train_classifier.py" "c:/Dev/Image Analysis/training_data_massive" --no-visualise-preview --save-arrays --model-output "c:/Dev/Image Analysis/training_data_massive/classifier.yml" --report-dir "c:/Dev/Image Analysis/training_data_massive/reports"
```