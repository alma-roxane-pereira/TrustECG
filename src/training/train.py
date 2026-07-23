import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from pathlib import Path
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from tqdm import tqdm

from src.training.class_weights import compute_pos_weights

from src.dataset.ecg_dataset import ECGGraphDataset
from src.models.stgnn import STGNN


# =====================================================
# Helper Functions
# =====================================================

def compute_pos_weight(dataset, num_classes, clamp_max=10.0):
    pos_counts = torch.zeros(num_classes)
    total = 0

    for data in dataset:
        y = data.y.view(-1).float()
        pos_counts += y
        total += 1

    neg_counts = total - pos_counts
    pos_counts = pos_counts.clamp(min=1.0)

    pos_weight = torch.sqrt(neg_counts / pos_counts)
    pos_weight = pos_weight.clamp(max=clamp_max)

    return pos_weight


def compute_metrics(targets, probs, fixed_threshold=0.5):

    preds = (probs > fixed_threshold).astype(np.float32)

    macro_f1 = f1_score(
        targets,
        preds,
        average="macro",
        zero_division=0,
    )

    valid_cols = [
        c for c in range(targets.shape[1])
        if targets[:, c].sum() > 0
    ]

    if valid_cols:
        mAP = average_precision_score(
            targets[:, valid_cols],
            probs[:, valid_cols],
            average="macro",
        )

        try:
            auroc = roc_auc_score(
                targets[:, valid_cols],
                probs[:, valid_cols],
                average="macro",
            )
        except ValueError:
            auroc = float("nan")
    else:
        mAP = float("nan")
        auroc = float("nan")

    return {
        "macro_f1_at_0.5": macro_f1,
        "mAP": mAP,
        "auroc": auroc,
    }


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=5.0):

    model.train()

    running_loss = 0
    all_probs = []
    all_targets = []

    progress = tqdm(loader, desc="Training")

    for batch in progress:

        batch = batch.to(device)

        optimizer.zero_grad()

        outputs, _ = model(batch)

        targets = batch.y.view(outputs.size(0), -1).float()

        loss = criterion(outputs, targets)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()

        probs = torch.sigmoid(outputs)

        all_probs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

        progress.set_postfix(loss=loss.item())

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    metrics = compute_metrics(all_targets, all_probs)

    epoch_loss = running_loss / len(loader)

    return epoch_loss, metrics


def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():

        for batch in tqdm(loader, desc="Validation"):

            batch = batch.to(device)

            outputs, _ = model(batch)

            targets = batch.y.view(outputs.size(0), -1).float()

            loss = criterion(outputs, targets)

            running_loss += loss.item()

            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    metrics = compute_metrics(all_targets, all_probs)

    epoch_loss = running_loss / len(loader)

    return epoch_loss, metrics


# =====================================================
# Main
# =====================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Device :", device)
    print("=" * 60)

    GRAPH_ROOT = Path("data/graphs_9Class")
    NUM_CLASSES = 9

    train_dataset = ECGGraphDataset(
        graph_root=GRAPH_ROOT,
        split_file="data/splits/train_9class.txt"
    )

    val_dataset = ECGGraphDataset(
        graph_root=GRAPH_ROOT,
        split_file="data/splits/val_9class.txt"
    )

    test_dataset = ECGGraphDataset(
        graph_root=GRAPH_ROOT,
        split_file="data/splits/test_9class.txt"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train Graphs : {len(train_dataset)}")
    print(f"Validation   : {len(val_dataset)}")
    print(f"Test         : {len(test_dataset)}")

    print("\nComputing pos_weight from training set...")

    pos_weight = compute_pos_weight(
        train_dataset,
        NUM_CLASSES,
    ).to(device)

    print("pos_weight per class:", pos_weight.cpu().numpy())

    model = STGNN(num_classes=NUM_CLASSES).to(device)

    print(model)

    weights = compute_pos_weights(
    "data/preprocessed_9Class/manifest.csv"
)

    weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

    criterion = nn.BCEWithLogitsLoss(
    pos_weight=weights
)

    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    NUM_EPOCHS = 50

    best_mAP = 0.0

    CKPT_PATH = "best_model_9class.pth"

    print("\nEverything Loaded Successfully!")

    for epoch in range(NUM_EPOCHS):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("=" * 60)

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"\nTrain Loss : {train_loss:.4f} | "
            f"mAP {train_metrics['mAP']:.4f} | "
            f"AUROC {train_metrics['auroc']:.4f} | "
            f"F1@0.5 {train_metrics['macro_f1_at_0.5']:.4f}"
        )

        print(
            f"Val Loss   : {val_loss:.4f} | "
            f"mAP {val_metrics['mAP']:.4f} | "
            f"AUROC {val_metrics['auroc']:.4f} | "
            f"F1@0.5 {val_metrics['macro_f1_at_0.5']:.4f}"
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,
    eta_min=1e-6
)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Current LR : {current_lr:.2e}")

        if val_metrics["mAP"] > best_mAP:
            best_mAP = val_metrics["mAP"]

            torch.save(
                model.state_dict(),
                CKPT_PATH,
            )

            print(f"Best model saved (val mAP = {best_mAP:.4f}).")


if __name__ == "__main__":
    main()