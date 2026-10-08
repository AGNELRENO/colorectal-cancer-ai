import torch

from model_densenet121 import create_densenet121


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("      DENSENET121 MODEL TEST")
print("========================================")

print(f"Device : {device}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

model = create_densenet121(num_classes=2)

model = model.to(device)

model.eval()


# --------------------------------------------------
# TEST INPUT
# --------------------------------------------------

dummy_input = torch.randn(
    32,
    3,
    224,
    224
).to(device)


# --------------------------------------------------
# FORWARD PASS
# --------------------------------------------------

with torch.no_grad():
    output = model(dummy_input)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print()
print(f"Input shape  : {dummy_input.shape}")
print(f"Output shape : {output.shape}")

print()
print("Expected input shape  : [32, 3, 224, 224]")
print("Expected output shape : [32, 2]")


# --------------------------------------------------
# VERIFICATION
# --------------------------------------------------

assert output.shape == (32, 2), \
    f"Unexpected output shape: {output.shape}"

print()
print("DenseNet121 model test successful!")