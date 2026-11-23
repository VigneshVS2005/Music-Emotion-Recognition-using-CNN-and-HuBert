# 🎵 Music Emotion Recognition using CNN and HuBERT

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Librosa-2496ED?style=for-the-badge&logo=python&logoColor=white" alt="Librosa" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Accuracy-89.8%25-brightgreen?style=for-the-badge" alt="Accuracy" />
</p>

An acoustic feature extraction and deep learning framework that combines fine-tuned **HuBERT (Hidden-Unit BERT)** speech representation embeddings with a **2D Convolutional Neural Network (CNN)** to classify musical audio into 8 categorical emotional states.

---

## 📌 Executive Summary & Key Results

- **Peak Validation Accuracy**: **93.62%**
- **Test Accuracy**: **89.83%** across 3,244 holdout test audio samples.
- **Trained on**: 10,812 audio samples utilizing NVIDIA A100 80GB GPU acceleration.
- **Multi-Class Evaluation**: Robust F1-scores across 8 discrete affective categories (*Bored/Lethargic: 0.94*, *Sad/Depressed: 0.91*, *Calm/Relaxed: 0.91*, *Excited/Alert: 0.90*, *Happy/Pleased: 0.88*).

---

## 🧠 System Architecture

The pipeline uses a multi-branch representation learning approach:
1. **HuBERT Feature Extraction**: Pretrained HuBERT transformer backbone captures high-level acoustic semantics and vocal tone dynamics.
2. **Spectral Audio Feature Extraction**: Librosa extracts Mel-Frequency Cepstral Coefficients (MFCCs), Chroma representations, Spectral Contrast, and RMS Energy to model timbre and pitch.
3. **CNN & Fusion Head**: 2D Convolutional layers process time-frequency spectrogram representations, concatenated with the pooled HuBERT embeddings through dense classification layers with dropout regularization.

```
Raw Audio (.mp3/.wav)
        │
        ├───> [ HuBERT Transformer Backbone ] ───> Acoustic Latent Vector (768-d) ──┐
        │                                                                           │
        └───> [ STFT / Mel-Spectrogram ]     ───> [ 2D CNN Conv Layers ]          ──┼──> [ Concatenation & Dense Layers ] ──> 8 Emotion Classes
```

---

## 🗂️ Repository Structure

```
├── HubertFusionModel.py                     # End-to-end model definition, feature extraction & training pipeline
├── requirements.txt                         # Python dependencies
├── Music Emotion Recognition Results/
│   └── Hubert_results_5/
│       ├── 01_training_history.png          # Epoch-by-epoch loss & accuracy trajectories
│       ├── 02_confusion_matrix_absolute.png # Raw counts confusion matrix
│       ├── 03_confusion_matrix_normalized.png# Normalized confusion matrix
│       ├── 09_model_architecture.png       # Layer connectivity diagram
│       ├── 11_tsne_visualization.png        # Latent space t-SNE projection
│       ├── best_enhanced_song_model.pth     # Checkpointed PyTorch model weights
│       └── results_summary.json             # Exhaustive classification benchmarks & metrics
└── README.md
```

---

## ⚡ Quickstart & Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/Karthickruban/Music-Emotion-Recognition-using-CNN-and-HuBert.git
cd Music-Emotion-Recognition-using-CNN-and-HuBert
pip install -r requirements.txt
```

### 2. Run Training / Inference Pipeline

```bash
python HubertFusionModel.py
```

---

## 👥 Contributors & Research Team

Developed as a collaborative AI research project:
- **Vignesh V S** ([@VigneshVS2005](https://github.com/VigneshVS2005)) — HuBERT transformer integration & model architecture
- **Karthick Ruban** ([@Karthickruban](https://github.com/Karthickruban)) — Audio DSP feature extraction, MFCC/spectrogram pipelines & evaluation

---

## 📄 License
MIT License
