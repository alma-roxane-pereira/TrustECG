from pathlib import Path

import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------

PREPROCESSED_ROOT = Path("data/preprocessed_9Class")

MANIFEST = PREPROCESSED_ROOT / "manifest.csv"

SPLITS = Path("data/splits")

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

manifest = pd.read_csv(MANIFEST)

# Create a lookup dictionary using the relative graph path
manifest["graph_path"] = manifest["npy_path"].apply(
    lambda x: str(Path(x).relative_to(PREPROCESSED_ROOT).with_suffix(".pt")).replace("\\", "/")
)

manifest = manifest.set_index("graph_path")


def check_split(split_file):

    print("\n" + "=" * 60)
    print(split_file.stem.upper())
    print("=" * 60)

    paths = []

    with open(split_file) as f:
        for line in f:
            paths.append(line.strip())

    subset = manifest.loc[paths]

    print(f"Total ECGs : {len(subset)}")
    print()

    counts = subset[CLASS_COLUMNS].sum()

    percentages = counts / len(subset) * 100

    result = pd.DataFrame({
        "Count": counts.astype(int),
        "Percentage": percentages.round(2)
    })

    print(result)


check_split(SPLITS / "train_9class.txt")
check_split(SPLITS / "val_9class.txt")
check_split(SPLITS / "test_9class.txt")