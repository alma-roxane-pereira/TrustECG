from torch_geometric.loader import DataLoader

from src.dataset.ecg_dataset import ECGGraphDataset

# =====================================================
# PATHS
# =====================================================

GRAPH_ROOT = "data/graphs_9Class"

TRAIN_SPLIT = "data/splits/train_9class.txt"
VAL_SPLIT = "data/splits/val_9class.txt"
TEST_SPLIT = "data/splits/test_9class.txt"

BATCH_SIZE = 32

# =====================================================
# DATASETS
# =====================================================

train_dataset = ECGGraphDataset(
    GRAPH_ROOT,
    TRAIN_SPLIT
)

val_dataset = ECGGraphDataset(
    GRAPH_ROOT,
    VAL_SPLIT
)

test_dataset = ECGGraphDataset(
    GRAPH_ROOT,
    TEST_SPLIT
)

# =====================================================
# DATALOADERS
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print()

print("Train Batches :", len(train_loader))
print("Validation Batches :", len(val_loader))
print("Test Batches :", len(test_loader))