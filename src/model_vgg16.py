import torch
import torch.nn as nn
from torchvision import models


def create_vgg16(num_classes=2):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Replace the final classifier layer
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_vgg16()
    model = model.to(device)

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    dummy_input = torch.randn(32, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)