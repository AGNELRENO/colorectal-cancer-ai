import torch

from model import create_model


# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# Create the ResNet18 model
model = create_model(num_classes=2)

# Move model to GPU/CPU
model = model.to(device)

# Create a dummy batch
dummy_input = torch.randn(32, 3, 224, 224).to(device)

# Forward pass
output = model(dummy_input)

print("Input shape :", dummy_input.shape)
print("Output shape:", output.shape)
print("Model test successful!")