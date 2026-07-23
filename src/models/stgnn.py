import torch
import torch.nn as nn

from .resnet1d import ResNetEncoder
from .gat_encoder import GATEncoder
from .gru_encoder import BiGRUEncoder
from .attention import AttentionPooling
from .classifier import ECGClassifier


class STGNN(nn.Module):

    def __init__(self, num_classes=9):

        super().__init__()

        self.resnet = ResNetEncoder()

        self.gat = GATEncoder()

        self.bigru = BiGRUEncoder()

        self.attention = AttentionPooling()

        self.classifier = ECGClassifier(
            input_dim=256,
            num_classes=num_classes
        )

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # -----------------------------
        # Spatial Feature Extraction
        # -----------------------------

        x = self.resnet(x)

        x = self.gat(x, edge_index)

        # -----------------------------
        # Temporal Modeling
        # -----------------------------

        x = self.bigru(x, batch)

        # -----------------------------
        # Attention Pooling
        # -----------------------------

        x, attention = self.attention(x)

        # -----------------------------
        # Classification
        # -----------------------------

        logits = self.classifier(x)

        return logits, attention