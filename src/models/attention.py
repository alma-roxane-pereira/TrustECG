import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):

    def __init__(self, input_dim=256):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x : (batch_size, num_nodes, input_dim)

        Returns
        -------
        pooled : (batch_size, input_dim)
        attention_weights : (batch_size, num_nodes)
        """

        # -------------------------------
        # Compute attention scores
        # -------------------------------

        scores = self.attention(x)

        # (batch, nodes)
        scores = scores.squeeze(-1)

        # -------------------------------
        # Normalize scores
        # -------------------------------

        attention_weights = F.softmax(scores, dim=1)

        # -------------------------------
        # Weighted sum
        # -------------------------------

        pooled = torch.sum(
            x * attention_weights.unsqueeze(-1),
            dim=1
        )

        return pooled, attention_weights