import torch

from model_basic_cnn import get_model


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = get_model(
    num_classes=2,
    dropout=0.3
)

model = model.to(device)

model.eval()


# ============================================================
# TEST INPUT
# ============================================================

dummy_input = torch.randn(
    2,
    3,
    224,
    224
).to(device)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    features = model.extract_features(
        dummy_input
    )

    output = model(
        dummy_input
    )


# ============================================================
# PARAMETER COUNT
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("BASIC CNN MODEL TEST")
print("========================================")

print(
    "Input shape:",
    dummy_input.shape
)

print(
    "Feature shape:",
    features.shape
)

print(
    "Output shape:",
    output.shape
)

print(
    "Total parameters:",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)

print(
    "Dropout:",
    model.dropout.p
)


# ============================================================
# ARCHITECTURE
# ============================================================

print("\n========================================")
print("BASIC CNN ARCHITECTURE")
print("========================================")

print(model)


# ============================================================
# VALIDATION
# ============================================================

assert dummy_input.shape == (
    2,
    3,
    224,
    224
)

assert features.shape == (
    2,
    128
)

assert output.shape == (
    2,
    2
)

assert model.conv1.in_channels == 3
assert model.conv1.out_channels == 32

assert model.conv2.in_channels == 32
assert model.conv2.out_channels == 64

assert model.conv3.in_channels == 64
assert model.conv3.out_channels == 128

assert model.dropout.p == 0.3

assert model.classifier.in_features == 128
assert model.classifier.out_features == 2


# ============================================================
# SUCCESS
# ============================================================

print("\n========================================")
print("SUCCESS")
print("========================================")

print(
    "Architecture: 3 convolutional layers"
)

print(
    "Channels: 3 → 32 → 64 → 128"
)

print(
    "Input: 3 × 224 × 224"
)

print(
    "Feature dimension: 128"
)

print(
    "Dropout: 0.3"
)

print(
    "Output classes: 2"
)

print(
    "Basic CNN architecture verified successfully."
)