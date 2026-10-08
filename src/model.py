import torch.nn as nn
from torchvision import models


def create_model(num_classes=2):

    # Load pretrained ResNet18
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Replace the original ImageNet classifier
    # with our CRC / Non-CRC classifier
    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model