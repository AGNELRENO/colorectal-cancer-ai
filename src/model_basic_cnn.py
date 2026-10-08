import torch
import torch.nn as nn


# ============================================================
# BASIC CNN - 3 CONVOLUTIONAL LAYERS
# ============================================================

class BasicCNN(nn.Module):

    def __init__(self, num_classes=2, dropout=0.3):

        super().__init__()

        # ====================================================
        # CONVOLUTIONAL LAYER 1
        # ====================================================

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.relu1 = nn.ReLU(
            inplace=True
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ====================================================
        # CONVOLUTIONAL LAYER 2
        # ====================================================

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.relu2 = nn.ReLU(
            inplace=True
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ====================================================
        # CONVOLUTIONAL LAYER 3
        # ====================================================

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.relu3 = nn.ReLU(
            inplace=True
        )

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ====================================================
        # ADAPTIVE AVERAGE POOLING
        # ====================================================

        self.global_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        # ====================================================
        # DROPOUT
        # ====================================================

        self.dropout = nn.Dropout(
            p=dropout
        )

        # ====================================================
        # FULLY CONNECTED LAYER
        # ====================================================

        self.classifier = nn.Linear(
            in_features=128,
            out_features=num_classes
        )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def extract_features(self, x):

        # ----------------------------------------------------
        # Layer 1
        # ----------------------------------------------------

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # ----------------------------------------------------
        # Layer 2
        # ----------------------------------------------------

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # ----------------------------------------------------
        # Layer 3
        # ----------------------------------------------------

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # ----------------------------------------------------
        # Adaptive Average Pooling
        # ----------------------------------------------------

        x = self.global_pool(x)

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        x = torch.flatten(
            x,
            start_dim=1
        )

        return x

    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, x):

        features = self.extract_features(x)

        # Dropout is applied only before classification.
        # The extracted 128-D features remain unchanged.
        features_dropout = self.dropout(
            features
        )

        output = self.classifier(
            features_dropout
        )

        return output


# ============================================================
# MODEL FACTORY
# ============================================================

def get_model(
    num_classes=2,
    dropout=0.3
):

    return BasicCNN(
        num_classes=num_classes,
        dropout=dropout
    )