# Human Activity Recognition with WISDM Sensor Data

This project classifies human activities from smartphone accelerometer data using a subject-independent evaluation protocol. The notebook loads WISDM sensor files, builds fixed-length windows, compares a Random Forest baseline with a lightweight 1D CNN, and converts the final model to TensorFlow Lite INT8 for edge deployment.

## Overview

The repository contains a notebook-based machine learning workflow for recognizing daily activities from inertial sensor data. The pipeline includes data loading, preprocessing, window generation, subject-based splitting, model training, performance evaluation, and TensorFlow Lite conversion.

## Objective

The goal is to classify activities such as walking, jogging, sitting, standing, and stairs using a compact sensor model that generalizes across subjects without leaking subject identity into the test set.

## WISDM dataset

The project uses the WISDM smartphone accelerometer dataset, organized as subject-specific CSV files. The notebook works with the WISDM-51 folder structure and evaluates performance with a subject-independent split so the model is tested on unseen individuals rather than on windows from the same subject.

## Data preprocessing and windowing

The notebook reads raw sensor streams from the WISDM files and converts them into fixed-length windows over the acceleration channels. These windows are aligned with activity labels and organized into training and test folds to preserve realistic generalization.

## Subject independent evaluation

Model performance is reported with GroupKFold-based subject splitting. This avoids unrealistic validation by ensuring that data from the same subject does not appear in both training and test folds.

## Random Forest baseline

A Random Forest classifier is trained on window-level features and used as a baseline for comparison. The notebook reports per-fold metrics and an aggregate accuracy and macro-F1 score under the subject-independent evaluation setup.

## Lightweight 1D CNN

A compact 1D CNN is trained on the same windowed time-series data. The architecture uses shallow convolutional blocks, pooling, dropout, and global average pooling to keep the model lightweight while preserving enough temporal structure for activity recognition.

## Verified model results

The notebook records the following subject-independent results from the implemented evaluation:

- Random Forest baseline: accuracy ≈ 0.27, macro-F1 ≈ 0.26
- Lightweight 1D CNN: accuracy ≈ 0.20, macro-F1 ≈ 0.18

These values are reported from the notebook’s actual evaluation and should be interpreted as realistic subject-independent performance rather than a highly tuned benchmark.

## TensorFlow Lite INT8 conversion

The trained lightweight CNN is converted to TensorFlow Lite using INT8 quantization. This step reduces model size and supports on-device or embedded inference while keeping the deployment path aligned with the project goal of mobile-friendly activity recognition.

## Technologies used

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- TensorFlow
- Keras
- TensorFlow Lite

## How to run the notebook

1. Create a Python environment.
2. Install the requirements:

```bash
pip install -r requirements.txt
```

3. Place the WISDM dataset in a local directory and set the `WISDM_DATA_DIR` environment variable if needed.
4. Open `HAR_Sensor_Classification.ipynb` in Jupyter or VS Code and run all cells.

The notebook is designed to work with a local dataset folder and does not require a separate data repository in this project.

## Limitations

The project is intentionally lightweight and focused on a narrow set of smartphone sensor features. The dataset is limited to accelerometer-based human activity recognition, and the reported performance is modest under a strict subject-independent setup. Additional tuning, richer features, or larger datasets may improve results.

## Future work

Potential next steps include more aggressive class balancing, comparison with additional sensor feature sets, evaluation of alternative 1D CNN architectures, and benchmarking the TFLite model on-device for latency and memory usage.


