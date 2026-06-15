# 🎵 Music Emotion Recognition using CNN and HuBert..

A machine learning project that classifies the emotional content of music audio files (e.g., happy, sad, calm, angry) using features extracted from the audio signals.

## 📌 Overview

Music powerfully conveys and evokes emotions. This project builds an automatic **music emotion recognition (MER)** system using supervised learning. It extracts acoustic features (like MFCCs, chroma, spectral contrast, etc.) from audio tracks and maps them to discrete emotion categories.

The notebook (`music-emotion-recognition (2).ipynb`) contains the end‑to‑end pipeline: data loading, feature extraction, model training, evaluation, and demonstration on new tracks.

## 🧠 Theoretical Background

### Emotion Models
Two common frameworks are often used:
- **Categorical**: discrete labels (e.g., *happy, sad, angry, relaxed, fearful*).
- **Dimensional**: continuous valence‑arousal space (valence = positivity, arousal = intensity).

This project follows a **categorical approach** for simplicity and interpretability.

### Key Audio Features
Music emotion relies on both temporal and spectral properties:

- **MFCCs (Mel‑Frequency Cepstral Coefficients)** – capture timbre and texture, related to perceived emotion.
- **Chroma Features** – represent harmonic content (pitch classes); useful for detecting major/minor tonalities (often linked to happy/sad).
- **Spectral Contrast** – separates peaks from valleys in the spectrum; correlates with brightness and complexity.
- **RMS Energy / Zero Crossing Rate** – indicate loudness and noisiness, linked to arousal.

### Typical Pipeline
1. **Preprocessing** – load audio, resample to common rate, normalize amplitude.
2. **Feature extraction** – slice audio into fixed‑length windows, compute feature vectors per window, then aggregate (mean/std).
3. **Modeling** – use classifiers like SVM, Random Forest, or a small neural network.
4. **Evaluation** – accuracy, F1‑score, confusion matrix.

## 🗂️ Repository Contents

| File | Description |
|------|-------------|
| `music-emotion-recognition (2).ipynb` | Complete Jupyter notebook with code, visualizations, and results. |
| `README.md` | This file – project overview and theoretical background. |

> Note: The audio dataset is **not included** due to size/licensing. See the “Dataset” section below for recommendations.
