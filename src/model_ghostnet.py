import torch
import torch.nn as nn
import timm


def get_model(num_classes=2):
    """
    Create a pretrained GhostNet model
    for CRC vs Non-CRC classification.
    """

    model = timm.create_model(
        "ghostnet_100",
        pretrained=True,
        num_classes=num_classes
    )

    return model


if __name__ == "__main__":
    print("TESTING GHOSTNET MODEL FILE")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = get_model()
    model = model.to(device)
    model.eval()

    # Test input
    x = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    print("SUCCESS: GhostNet model is working!")