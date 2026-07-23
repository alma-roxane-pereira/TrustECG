import torch
import torch.nn as nn

from pathlib import Path

from torch_geometric.loader import DataLoader

from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    hamming_loss,
    classification_report,
    multilabel_confusion_matrix
)

from src.dataset.ecg_dataset import ECGGraphDataset
from src.models.stgnn import STGNN


# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Device :", device)
print("=" * 60)


# =====================================================
# Paths
# =====================================================

GRAPH_ROOT = Path("data/graphs_9Class")

MODEL_PATH = "best_model_9class.pth"


# =====================================================
# Test Dataset
# =====================================================

test_dataset = ECGGraphDataset(
    graph_root=GRAPH_ROOT,
    split_file="data/splits/test_9class.txt"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,      # Windows-safe
    pin_memory=True
)

print(f"Loaded {len(test_dataset)} Test Graphs")

# =====================================================
# Load Model
# =====================================================

model = STGNN(
    num_classes=9
).to(device)

# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# -----------------------------------------------------
# Load model weights
# -----------------------------------------------------

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(f"\nLoaded checkpoint from Epoch {checkpoint['epoch']}")
    print(f"Best Validation mAP : {checkpoint['best_mAP']:.4f}")

else:

    # Backward compatibility
    model.load_state_dict(checkpoint)

    print("\nLoaded legacy model checkpoint.")

model.eval()

print("\nModel Loaded Successfully!")

# =====================================================
# Loss Function
# =====================================================

criterion = nn.BCEWithLogitsLoss()
# =====================================================
# Evaluation
# =====================================================

all_targets = []
all_predictions = []
all_probabilities = []

running_loss = 0.0

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        outputs, attention = model(batch)

        targets = batch.y.view(outputs.size(0), -1).float()

        loss = criterion(outputs, targets)

        running_loss += loss.item()

        probabilities = torch.sigmoid(outputs)

        predictions = (probabilities > 0.5).float()

        all_targets.append(targets.cpu())

        all_predictions.append(predictions.cpu())

        all_probabilities.append(probabilities.cpu())


# =====================================================
# Convert to NumPy
# =====================================================

all_targets = torch.cat(all_targets).numpy()

all_predictions = torch.cat(all_predictions).numpy()

all_probabilities = torch.cat(all_probabilities).numpy()

test_loss = running_loss / len(test_loader)

# =====================================================
# Metrics
# =====================================================

accuracy = accuracy_score(
    all_targets,
    all_predictions
)

precision = precision_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

micro_f1 = f1_score(
    all_targets,
    all_predictions,
    average="micro",
    zero_division=0
)

auc = roc_auc_score(
    all_targets,
    all_probabilities,
    average="macro"
)

ap = average_precision_score(
    all_targets,
    all_probabilities,
    average="macro"
)

hloss = hamming_loss(
    all_targets,
    all_predictions
)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Loss            : {test_loss:.4f}")
print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")
print(f"Micro F1        : {micro_f1:.4f}")
print(f"Macro AUROC     : {auc:.4f}")
print(f"Average Precision : {ap:.4f}")
print(f"Hamming Loss    : {hloss:.4f}")

CLASS_NAMES = [
    "SB",
    "TWC",
    "SR",
    "AFIB",
    "ST",
    "OldMI",
    "STTC",
    "SVT",
    "AF"
]

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)

print(
    classification_report(
        all_targets,
        all_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("\n" + "="*60)
print("CONFUSION MATRICES")
print("="*60)

cms = multilabel_confusion_matrix(
    all_targets,
    all_predictions
)

for i, cm in enumerate(cms):

    print(f"\nClass : {CLASS_NAMES[i]}")

    print(cm)
    
with open("classification_report.txt", "w") as f:

    f.write(
        classification_report(
            all_targets,
            all_predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )
    
import numpy as np

np.save(
    "confusion_matrices_v2.npy",
    cms
)