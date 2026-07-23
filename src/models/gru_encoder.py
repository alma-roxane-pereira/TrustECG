import torch
import torch.nn as nn


# =====================================================
# BiGRU Encoder
# =====================================================

class BiGRUEncoder(nn.Module):

    def __init__(
        self,
        input_size=128,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

    def forward(self, x, batch):

        # -------------------------------------------------
        # x : (num_nodes, 128)
        # batch : graph ids of every node
        # -------------------------------------------------

        batch_size = batch.max().item() + 1

        num_nodes = x.size(0) // batch_size

        x = x.view(batch_size, num_nodes, -1)

        output, hidden = self.gru(x)

        return output