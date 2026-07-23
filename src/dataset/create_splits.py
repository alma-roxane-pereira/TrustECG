from pathlib import Path

import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# =====================================================
# PATHS
# =====================================================

GRAPH_ROOT = Path("data/graphs_9Class")

PREPROCESSED_ROOT = Path("data/preprocessed_9Class")

MANIFEST = PREPROCESSED_ROOT / "manifest.csv"

SPLIT_ROOT = Path("data/splits")

SPLIT_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

manifest = pd.read_csv(MANIFEST)

print("="*60)
print("Creating Train / Validation / Test Split")
print("="*60)

print(f"Total ECGs : {len(manifest)}")

graph_paths = []

for _, row in manifest.iterrows():

    npy_path = Path(row["npy_path"])

    relative = npy_path.relative_to(PREPROCESSED_ROOT)

    graph_path = GRAPH_ROOT / relative.parent / f"{npy_path.stem}.pt"

    graph_paths.append(graph_path)


# =====================================================
# BUILD LABEL MATRIX
# =====================================================

CLASS_COLUMNS = [
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

Y = manifest[CLASS_COLUMNS].values.astype(int)

print(f"Label Matrix Shape : {Y.shape}")
# =====================================================
# FIRST SPLIT
# Train (70%)
# Temp  (30%)
# =====================================================

msss = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=RANDOM_STATE
)

train_idx, temp_idx = next(
    msss.split(np.zeros(len(Y)), Y)
)

train_paths = [graph_paths[i] for i in train_idx]
train_labels = Y[train_idx]

temp_paths = [graph_paths[i] for i in temp_idx]
temp_labels = Y[temp_idx]

print(f"Train ECGs : {len(train_paths)}")
print(f"Temp ECGs  : {len(temp_paths)}")

# =====================================================
# SECOND SPLIT
# Temp -> Validation + Test
# =====================================================

msss = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=RANDOM_STATE
)

val_idx, test_idx = next(
    msss.split(np.zeros(len(temp_labels)), temp_labels)
)

val_paths = [temp_paths[i] for i in val_idx]
test_paths = [temp_paths[i] for i in test_idx]

print(f"Validation ECGs : {len(val_paths)}")
print(f"Test ECGs       : {len(test_paths)}")

# =====================================================
# SAVE SPLITS
# =====================================================

train_file = SPLIT_ROOT / "train_9class.txt"
val_file = SPLIT_ROOT / "val_9class.txt"
test_file = SPLIT_ROOT / "test_9class.txt"


def save_split(paths, filename):

    with open(filename, "w") as f:

        for path in paths:

            relative = path.relative_to(GRAPH_ROOT)

            f.write(str(relative).replace("\\", "/") + "\n")


save_split(train_paths, train_file)
save_split(val_paths, val_file)
save_split(test_paths, test_file)

# =====================================================
# SUMMARY
# =====================================================

print("\n")
print("=" * 60)
print("Dataset Split Completed")
print("=" * 60)

print(f"Train      : {len(train_paths)}")
print(f"Validation : {len(val_paths)}")
print(f"Test       : {len(test_paths)}")

print("\nSaved to:")
print(train_file)
print(val_file)
print(test_file)