# Song Identification from Humming and Whistling

Multiclass audio classification: given a 10-second recording of someone humming or whistling, identify which of 8 songs is being performed. Trained and evaluated on 400 recordings collected from students.

## Problem

8-class classification from raw audio. 50 recordings per song, split across hum and whistle performances. No lyrics or rhythm cues, only pitch and spectral patterns.

## Pipeline

**1. Audio standardisation**
All files normalised to exactly 10 seconds at 22,050 Hz (220,500 samples) using truncation or zero-padding.

**2. Feature extraction (56 dimensions)**
- Spectral features (6): power, centroid, bandwidth, flatness, rolloff, zero-crossing rate
- Chroma features (24): mean and std across 12 pitch classes
- MFCC features (26): mean and std of 13 Mel-frequency cepstral coefficients

**3. Classification**
Seven classifiers trained under identical conditions and compared. An ensemble combines diverse models to reduce overfitting.

## Models

Random Forest, Gradient Boosting, K-Nearest Neighbours, XGBoost, Support Vector Machine, Gaussian Naive Bayes, Extra Trees

## Evaluation

- Accuracy per model
- Precision, recall, F1-score per class
- Confusion matrix showing misclassification patterns
- Train/test split validation on held-out samples

## Libraries

`librosa` `scikit-learn` `xgboost` `numpy` `pandas` `matplotlib`

## How to Run

```bash
pip install librosa scikit-learn xgboost numpy pandas matplotlib
jupyter notebook Hums_vs_Whist_ML.ipynb
```

Place audio files in the expected directory before running (see notebook cell 1 for paths).
