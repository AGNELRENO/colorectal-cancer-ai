import torch.nn as nn
from torchvision import models


def create_densenet121(num_classes=2):
    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        num_classes
    )

    return model