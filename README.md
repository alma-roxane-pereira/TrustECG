
# TrustECG Explainable Spatio-Temporal Graph Neural Network for Multi-Label ECG Arrhythmia Classification

## Overview

ECG-STGNN is a deep learning framework for automatic multi-label arrhythmia classification from 12-lead Electrocardiogram (ECG) recordings.

The model combines:

- ResNet-1D for local feature extraction
- Graph Attention Networks (GAT) for spatial lead relationships
- Bidirectional GRU for temporal dependency modeling
- Attention Pooling for explainability
- Multi-label classification using BCEWithLogitsLoss

The project is built using PyTorch and PyTorch Geometric on the PhysioNet Large Scale 12-lead ECG Dataset.

---

## Features

- Multi-label ECG arrhythmia classification
- Physiological ECG graph construction
- ResNet-based signal encoder
- Graph Attention Network (GAT)
- Bidirectional GRU
- Attention-based pooling
- End-to-end deep learning pipeline
- GPU support
- Modular architecture
- Evaluation with multiple metrics

---

## Project Structure

```
ECG-STGNN
│
├── src
│   ├── dataset
│   ├── preprocessing
│   ├── models
│   ├── training
│   ├── evaluate
│   └── config.py
│
├── data
│   ├── splits
│   └── label_mapping.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Model Architecture

```
12-Lead ECG
      │
      ▼
 ResNet-1D Encoder
      │
      ▼
 Graph Attention Network
      │
      ▼
 Bidirectional GRU
      │
      ▼
 Attention Pooling
      │
      ▼
 Fully Connected Classifier
      │
      ▼
 Multi-label Predictions
```

---

## Dataset

Dataset used:

**PhysioNet Large Scale 12-lead ECG Database**

Each sample contains:

- 12 ECG leads
- Sampling Frequency: 500 Hz
- Approximately 10 seconds per recording
- Multi-label diagnosis

This repository does **not** include the dataset.

Please download it separately from PhysioNet.

---

## Preprocessing Pipeline

The preprocessing pipeline consists of:

- Baseline wander removal
- 50 Hz notch filtering
- Low-pass filtering
- Graph construction using physiological lead connections
- Multi-label encoding

---

## Training

Run

```bash
python -m src.training.train
```

Training uses:

- AdamW Optimizer
- BCEWithLogitsLoss
- ReduceLROnPlateau Scheduler
- Gradient Clipping

---

## Evaluation

Run

```bash
python -m src.evaluate.evaluate
```

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- Macro F1
- Micro F1
- AUROC
- Average Precision (mAP)
- Hamming Loss
- Classification Report
- Confusion Matrix

---

## Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- NumPy
- Scikit-learn
- SciPy
- NetworkX
- tqdm

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/TrustECG.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Results

Example evaluation:

| Metric | Score |
|---------|-------|
| Accuracy | 61.67% |
| Macro F1 | 0.7037 |
| Micro F1 | 0.7997 |
| Macro AUROC | 0.9465 |
| Average Precision | 0.7031 |
| Hamming Loss | 0.0654 |

---

## Future Work

- Explainable AI (Attention Visualization)
- Grad-CAM for ECG signals
- Threshold Optimization
- Improved Class Imbalance Handling
- Clinical Decision Support Integration

---

