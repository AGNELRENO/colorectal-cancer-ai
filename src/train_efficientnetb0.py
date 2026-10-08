import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def get_model(num_classes=2):
    """
    Create pretrained EfficientNet-B0
    for CRC / Non-CRC classification.
    """

    # Load pretrained EfficientNet-B0
    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=weights
    )

    # Get input features of the final classifier
    in_features = model.classifier[1].in_features

    # Replace ImageNet classifier with 2-class classifier
    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


if __name__ == "__main__":

    print("=" * 60)
    print("TESTING EFFICIENTNET-B0 MODEL FILE")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = get_model(num_classes=2)
    model = model.to(device)
    model.eval()

    x = torch.randn(
        2,
        3,
        224,
        224,
        device=device
    )

    with torch.no_grad():
        output = model(x)

    print(f"Input shape : {x.shape}")
    print(f"Output shape: {output.shape}")

    if output.shape == torch.Size([2, 2]):
        print("\nSUCCESS: EfficientNet-B0 model is working!")
    else:
        print("\nERROR: Unexpected output shape.")