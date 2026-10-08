import torch
import torch.nn as nn


class MultiScaleResidualBlock(nn.Module):
    """
    Residual Multi-Scale CNN block.

    Each block has:
    - 3x3 convolution branch
    - 5x5 convolution branch
    - Feature fusion
    - Residual connection
    - Max pooling
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # -----------------------------
        # 3x3 convolution branch
        # -----------------------------
        self.conv3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(out_channels)

        # -----------------------------
        # 5x5 convolution branch
        # -----------------------------
        self.conv5 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding=2,
            bias=False
        )

        self.bn5 = nn.BatchNorm2d(out_channels)

        # -----------------------------
        # Fuse multi-scale features
        # -----------------------------
        self.fusion = nn.Conv2d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn_fusion = nn.BatchNorm2d(out_channels)

        # -----------------------------
        # Residual connection
        # -----------------------------
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        # -----------------------------
        # Downsampling
        # -----------------------------
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        # Save input for residual connection
        residual = self.residual(x)

        # -----------------------------
        # 3x3 branch
        # -----------------------------
        branch3 = self.conv3(x)
        branch3 = self.bn3(branch3)
        branch3 = self.relu(branch3)

        # -----------------------------
        # 5x5 branch
        # -----------------------------
        branch5 = self.conv5(x)
        branch5 = self.bn5(branch5)
        branch5 = self.relu(branch5)

        # -----------------------------
        # Multi-scale feature fusion
        # -----------------------------
        combined = torch.cat(
            [branch3, branch5],
            dim=1
        )

        combined = self.fusion(combined)
        combined = self.bn_fusion(combined)

        # -----------------------------
        # Residual addition
        # -----------------------------
        output = combined + residual
        output = self.relu(output)

        # -----------------------------
        # Downsample
        # -----------------------------
        output = self.pool(output)

        return output


class RMSCNN(nn.Module):
    """
    3-Layer Residual Multi-Scale CNN.

    Input:
        [Batch, 3, 224, 224]

    Output:
        [Batch, 2]

    Classes:
        0 = CRC
        1 = Non-CRC
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # =====================================
        # RMSCNN Layer 1
        # =====================================
        self.layer1 = MultiScaleResidualBlock(
            in_channels=3,
            out_channels=32
        )

        # =====================================
        # RMSCNN Layer 2
        # =====================================
        self.layer2 = MultiScaleResidualBlock(
            in_channels=32,
            out_channels=64
        )

        # =====================================
        # RMSCNN Layer 3
        # =====================================
        self.layer3 = MultiScaleResidualBlock(
            in_channels=64,
            out_channels=128
        )

        # =====================================
        # Global Average Pooling
        # =====================================
        self.global_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        # =====================================
        # Classification layer
        # =====================================
        self.classifier = nn.Linear(
            in_features=128,
            out_features=num_classes
        )

    def extract_features(self, x):

        # Layer 1
        x = self.layer1(x)

        # Layer 2
        x = self.layer2(x)

        # Layer 3
        x = self.layer3(x)

        # Global average pooling
        x = self.global_pool(x)

        # Convert [B, 128, 1, 1]
        # to [B, 128]
        x = torch.flatten(
            x,
            start_dim=1
        )

        return x

    def forward(self, x):

        # Extract deep features
        features = self.extract_features(x)

        # Classification
        output = self.classifier(features)

        return output


def get_model(num_classes=2):
    """
    Create and return the 3-layer RMSCNN model.
    """

    model = RMSCNN(
        num_classes=num_classes
    )

    return model