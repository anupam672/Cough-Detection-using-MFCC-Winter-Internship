# Cough Detection using MFCC

A machine learning-based audio analysis system for automated cough detection using MFCC (Mel-Frequency Cepstral Coefficients) features and audio signal processing.

Built as part of a Winter Internship at the Centre for Indian Knowledge Systems (CIKS), IIT Guwahati, under the guidance of Prof. Uday Shanker Dixit.

---

## Overview

This project classifies short audio clips as **COUGH** or **NON-COUGH** using acoustic signal features. Raw audio is converted into a 205-dimensional MFCC-based feature vector, and two boosted decision-tree classifiers (Gradient Boosting and XGBoost) are trained and compared. The trained model is deployed in two ways: file-based prediction and real-time microphone-based prediction.

**Pipeline:** `Raw Audio → Preprocessing → Feature Extraction (MFCC) → Classification → COUGH / NON-COUGH`

---

## Results

| Metric | Gradient Boosting | XGBoost |
|---|---|---|
| Test Accuracy | 94.46% | **95.42%** |
| Precision (Cough) | 0.92 | 0.94 |
| Recall (Cough) | 0.90 | 0.91 |
| F1-score (Cough) | 0.91 | 0.93 |
| AUC (ROC) | 0.985 | **0.990** |

XGBoost outperforms Gradient Boosting across every metric and is used as the model for real-time prediction.

---

## Datasets Used

This project combines three publicly available audio datasets (**not included in this repository** — see [Dataset Setup](#dataset-setup) below):

| Dataset | Description | Source |
|---|---|---|
| **Coswara** | Clean, labelled cough & speech recordings from respiratory research | [Kaggle](https://www.kaggle.com/datasets/himanshu007121/coswara-dataset) |
| **ESC-50** | 50-class environmental sound dataset, used for non-cough examples | [GitHub](https://github.com/karolpiczak/ESC-50) |
| **CoughVid** | Large-scale, real-world crowdsourced cough recordings | [Zenodo](https://zenodo.org/records/4498364) |

Combined raw dataset: **22,785 samples** (7,096 cough, 15,689 non-cough). After preprocessing and feature extraction: **21,920 labelled feature vectors**.

---

## Project Structure

```
cough-detection-mfcc/
├── notebooks/
│   ├── Untitled.ipynb          # Data preprocessing, feature extraction & Gradient Boosting training
│   ├── Traninig_XGB.ipynb      # XGBoost training
│   ├── pred_gb.ipynb           # File-based prediction (Gradient Boosting)
│   └── prediction.ipynb        # Real-time microphone prediction (XGBoost)
├── models/                     # Trained model artifacts
│   ├── cough_detector.pkl      # Trained Gradient Boosting model
│   ├── xgboost.pkl             # Trained XGBoost model
│   ├── scaler.pkl              # StandardScaler fitted on training features
│   └── label_encoder.pkl       # Label encoder (cough / non_cough)
├── processed/                  # Cached extracted features (optional, regenerable)
│   ├── features.npy
│   └── labels.npy
├── final_dataset/              # Preprocessed audio, organized by class (not included)
│   ├── cough/
│   └── non_cough/
└── README.md
```

> **Note:** Raw and preprocessed audio datasets are **not pushed to this repository** due to size. See [Dataset Setup](#dataset-setup) to reproduce them locally.

---

## Dataset Setup

Since dataset files are not included in this repo, to reproduce the pipeline from scratch:

1. Download the three datasets from the links in the [table above](#datasets-used).
2. Run the preprocessing steps in `Untitled.ipynb` to:
   - Resample all audio to 22,050 Hz
   - Standardize duration to 5 seconds
   - Trim silence (amplitude-based, `top_db=20`)
   - Reduce background noise (`noisereduce`)
   - Convert non-WAV files to WAV
3. Organize the processed audio into:
   ```
   final_dataset/
   ├── cough/
   └── non_cough/
   ```
4. Run feature extraction to generate `processed/features.npy` and `processed/labels.npy`.

If you already have `processed/features.npy` and `processed/labels.npy`, you can skip straight to model training (no need to re-process raw audio).

---

## Feature Extraction

Each audio clip is converted into a **205-dimensional feature vector**:

| Component | Dimensions | Description |
|---|---|---|
| MFCC statistics | 120 | Mean, std-dev, max of 40 MFCC coefficients |
| Delta MFCC | 80 | Mean, std-dev of the rate of change of MFCCs |
| Spectral features | 4 | Centroid, rolloff, zero-crossing rate, RMS energy |
| Band energy | 1 | Normalized energy in the 500–2000 Hz cough band |

---

## Models

Both models are ensemble, gradient-boosted decision tree classifiers trained on an 80:20 stratified train-test split (17,536 train / 4,384 test).

**Gradient Boosting**
```python
GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.8
)
```

**XGBoost**
```python
XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    objective='binary:logistic', eval_metric='logloss'
)
```

---

## Usage

Trained models (`models/cough_detector.pkl`, `models/xgboost.pkl`, `models/scaler.pkl`, `models/label_encoder.pkl`) are included in this repo, so you can jump straight to prediction (steps 2–3) without retraining.

### 1. Train the models (optional — already included)
Run `Untitled.ipynb` (Gradient Boosting) and `Traninig_XGB.ipynb` (XGBoost) in order. This saves the trained models, scaler, and label encoder into `models/`.

### 2. File-based prediction
```python
# In pred_gb.ipynb
file_path = "path/to/your/audio.wav"
predict_audio(file_path)   # returns "COUGH" or "NON-COUGH"
```
A sliding-window variant (`predict_audio_sliding`) is also available for scanning longer recordings.

### 3. Real-time microphone prediction
```python
# In prediction.ipynb
record_and_predict()   # records 2.5s from mic, returns COUGH / NON-COUGH
```
Requires a working microphone and the `sounddevice` package. Run locally (not on a cloud notebook).

---

## Requirements

```
numpy
librosa
noisereduce
scipy
scikit-learn
xgboost
joblib
sounddevice
```

Install with:
```bash
pip install numpy librosa noisereduce scipy scikit-learn xgboost joblib sounddevice
```

---

## Author

**Anupam Kumari**
B.Tech, Electronics and Communication Engineering
National Institute of Technology Mizoram (BT23EC011)

**Guide:** Prof. Uday Shanker Dixit, Centre for Indian Knowledge Systems (CIKS), IIT Guwahati

---

## Future Work

- Cough sub-type classification (dry vs. wet cough)
- CNN-based spectrogram models
- Robustness to overlapping speech and noisy audio
- Lightweight mobile/embedded deployment for continuous monitoring