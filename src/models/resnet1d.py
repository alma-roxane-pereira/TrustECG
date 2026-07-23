import torch
import torch.nn as nn


# =====================================================
# Residual Block
# =====================================================

class ResidualBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None

        if stride != 1 or in_channels != out_channels:

            self.downsample = nn.Sequential(

                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm1d(out_channels)

            )

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


# =====================================================
# ResNet Encoder
# =====================================================

class ResNetEncoder(nn.Module):

    def __init__(self, embedding_dim=128):

        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool1d(
                kernel_size=3,
                stride=2,
                padding=1
            )

        )

        self.layer1 = ResidualBlock1D(
            32,
            32
        )

        self.layer2 = ResidualBlock1D(
            32,
            64,
            stride=2
        )

        self.layer3 = ResidualBlock1D(
            64,
            128,
            stride=2
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(
            128,
            embedding_dim
        )

    def forward(self, x):

        # x : (N,5000)

        x = x.unsqueeze(1)

        # (N,1,5000)

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.global_pool(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        return x