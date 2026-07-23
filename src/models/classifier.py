import torch
import torch.nn as nn


class ECGClassifier(nn.Module):

    def __init__(
        self,
        input_dim=256,
        num_classes=9,
        dropout=0.3
    ):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(input_dim, 128),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(64, num_classes)

        )

    def forward(self, x):

        return self.classifier(x)