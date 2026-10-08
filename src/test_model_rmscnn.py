import torch

from model_rmscnn import get_model


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
# CREATE MODEL
# ============================================================

model = get_model(
    num_classes=2
)

model = model.to(device)

model.eval()


# ============================================================
# TEST INPUT
# ============================================================

x = torch.randn(
    2,
    3,
    224,
    224
).to(device)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    output = model(x)

    features = model.extract_features(x)


# ============================================================
# MODEL INFORMATION
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("RMSCNN MODEL TEST")
print("========================================")

print(
    "Input shape   :",
    x.shape
)

print(
    "Feature shape :",
    features.shape
)

print(
    "Output shape  :",
    output.shape
)

print(
    "Total parameters     :",
    total_params
)

print(
    "Trainable parameters :",
    trainable_params
)

print("\nSUCCESS: RMSCNN model is working!")