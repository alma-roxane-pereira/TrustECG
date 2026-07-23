"""
build_graphs_9class.py

TrustECG
Graph Construction Pipeline

Converts preprocessed ECG recordings into
PyTorch Geometric graphs.

Each graph contains:

- x           : node features (12 leads × 5000 samples)
- edge_index  : physiological ECG graph
- y           : 9-class multi-label vector

Output:
data/graphs_9Class/
    JS00001.pt
    JS00002.pt
    ...
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# =====================================================
# PATHS
# =====================================================

PREPROCESSED_ROOT = Path("data/preprocessed_9Class")
GRAPH_ROOT = Path("data/graphs_9Class")

MANIFEST = PREPROCESSED_ROOT / "manifest.csv"

GRAPH_ROOT.mkdir(parents=True, exist_ok=True)

FAIL_LOG = GRAPH_ROOT / "graph_failures.log"

# =====================================================
# LOGGER
# =====================================================

logging.basicConfig(
    filename=FAIL_LOG,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s : %(message)s",
)

logger = logging.getLogger(__name__)

# =====================================================
# GRAPH STATISTICS
# =====================================================

processed_graphs = 0
failed_graphs = 0

# =====================================================
# LOAD MANIFEST
# =====================================================

manifest = pd.read_csv(MANIFEST)

print("=" * 60)
print("TrustECG Graph Construction")
print("=" * 60)

print(f"Found {len(manifest)} ECG recordings")

# =====================================================
# PHYSIOLOGICAL ECG GRAPH
# =====================================================

# Lead order
#
# 0  -> I
# 1  -> II
# 2  -> III
# 3  -> aVR
# 4  -> aVL
# 5  -> aVF
# 6  -> V1
# 7  -> V2
# 8  -> V3
# 9  -> V4
# 10 -> V5
# 11 -> V6

LEAD_NAMES = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6"
]


# Physiological connections
edges = [

    # Limb Leads
    (0,1), (1,0),
    (1,2), (2,1),
    (0,2), (2,0),

    # Augmented Leads
    (0,3), (3,0),
    (0,4), (4,0),
    (1,5), (5,1),
    (2,5), (5,2),
    (3,4), (4,3),
    (4,5), (5,4),

    # Chest Leads
    (6,7), (7,6),
    (7,8), (8,7),
    (8,9), (9,8),
    (9,10), (10,9),
    (10,11), (11,10),

    # Limb ↔ Chest Connections
    (1,6), (6,1),
    (1,7), (7,1),
    (5,8), (8,5),
    (5,9), (9,5),
    (0,10), (10,0),
    (0,11), (11,0)
]

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

print(f"Number of Nodes : {len(LEAD_NAMES)}")
print(f"Number of Edges : {edge_index.shape[1]}")

# =====================================================
# BUILD SINGLE GRAPH
# =====================================================

def build_graph(ecg_path, label_path):
    """
    Builds one PyTorch Geometric graph from a preprocessed ECG.

    Parameters
    ----------
    ecg_path : Path
        Path to preprocessed ECG (.npy)

    label_path : Path
        Path to multi-label vector (.npy)

    Returns
    -------
    Data
        PyTorch Geometric graph
    """

    # -------------------------------------------------
    # Load ECG
    # -------------------------------------------------
    ecg = np.load(ecg_path)

    # Expected shape = (12, 5000)
    if ecg.shape != (12, 5000):
        raise ValueError(
            f"{ecg_path.name}: Expected shape (12,5000), got {ecg.shape}"
        )

    # -------------------------------------------------
    # Load Labels
    # -------------------------------------------------
    labels = np.load(label_path)

    if labels.shape != (9,):
        raise ValueError(
            f"{label_path.name}: Expected label shape (9,), got {labels.shape}"
        )

    # -------------------------------------------------
    # NaN / Inf Check
    # -------------------------------------------------
    if not np.isfinite(ecg).all():
        raise ValueError("ECG contains NaN or Inf values")

    if not np.isfinite(labels).all():
        raise ValueError("Labels contain NaN or Inf values")

    # -------------------------------------------------
    # Convert to Torch
    # -------------------------------------------------
    x = torch.tensor(ecg, dtype=torch.float)

    y = torch.tensor(labels, dtype=torch.float)

    # -------------------------------------------------
    # Build Graph
    # -------------------------------------------------
    graph = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    return graph

# =====================================================
# BUILD ALL GRAPHS
# =====================================================

print("\nBuilding Graphs...\n")

for _, row in tqdm(manifest.iterrows(), total=len(manifest)):

    try:

        ecg_path = Path(row["npy_path"])
        label_path = Path(row["label_path"])

        graph = build_graph(ecg_path, label_path)

        relative_path = ecg_path.relative_to(PREPROCESSED_ROOT)
        
        save_dir = GRAPH_ROOT / relative_path.parent
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / f"{ecg_path.stem}.pt"
        torch.save(graph, save_path)

        

        

        processed_graphs += 1

    except Exception as e:

        failed_graphs += 1

        logger.error(
            f"{row['record_id']} --> {str(e)}"
        )

print("\n")
print("=" * 60)
print("Graph Construction Completed")
print("=" * 60)

print(f"Graphs Created : {processed_graphs}")
print(f"Failed Graphs  : {failed_graphs}")
print(f"Saved to       : {GRAPH_ROOT}")
print(f"Failure Log    : {FAIL_LOG}")

# =====================================================
# VERIFY ONE SAVED GRAPH
# =====================================================

sample_graph = next(GRAPH_ROOT.glob("*.pt"), None)

if sample_graph is not None:

    graph = torch.load(sample_graph)

    print("\nSample Graph")
    print("-" * 40)

    print(graph)

    print(f"x shape         : {graph.x.shape}")
    print(f"edge_index      : {graph.edge_index.shape}")
    print(f"labels          : {graph.y}")
    print(f"Number of nodes : {graph.num_nodes}")
    print(f"Number of edges : {graph.num_edges}")