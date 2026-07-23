import torch
import torch.nn as nn

from torch_geometric.nn import GATConv


# =====================================================
# GAT Encoder
# =====================================================

class GATEncoder(nn.Module):

    def __init__(
        self,
        in_channels=128,
        hidden_channels=128,
        out_channels=128,
        heads=4,
        dropout=0.2
    ):

        super().__init__()

        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=False,
            dropout=dropout
        )

        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.gat2 = GATConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            concat=False,
            dropout=dropout
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):

        # ------------------------------------
        # First GAT Layer
        # ------------------------------------

        x = self.gat1(x, edge_index)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.dropout(x)

        # ------------------------------------
        # Second GAT Layer
        # ------------------------------------

        x = self.gat2(x, edge_index)

        x = self.bn2(x)

        x = self.relu(x)

        x = self.dropout(x)

        return x