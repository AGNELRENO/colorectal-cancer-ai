import os
import matplotlib.pyplot as plt

# GhostNet training history from the completed 10-epoch run
epochs = list(range(1, 11))

train_loss = [
    0.0683, 0.0054, 0.0039, 0.0004, 0.0017,
    0.0008, 0.0002, 0.0001, 0.0002, 0.0017
]

train_acc = [
    0.9812, 0.9980, 0.9988, 1.0000, 0.9994,
    0.9999, 1.0000, 1.0000, 1.0000, 0.9999
]

val_loss = [
    0.0017, 0.0016, 0.0008, 0.0004, 0.0006,
    0.0006, 0.0003, 0.0001, 0.0001, 0.0003
]

val_acc = [
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000
]

output_dir = r"E:\colorectal-cancer-project\results\ghostnet"
os.makedirs(output_dir, exist_ok=True)


# -----------------------------
# 1. Loss Curve
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, train_loss, marker="o", label="Training Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GhostNet Training and Validation Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "ghostnet_loss_curve.png"),
    dpi=300
)
plt.close()


# -----------------------------
# 2. Accuracy Curve
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, train_acc, marker="o", label="Training Accuracy")
plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("GhostNet Training and Validation Accuracy")
plt.xticks(epochs)
plt.ylim(0.97, 1.005)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "ghostnet_accuracy_curve.png"),
    dpi=300
)
plt.close()


# -----------------------------
# 3. Individual Training Loss
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, train_loss, marker="o")

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("GhostNet Training Loss")
plt.xticks(epochs)
plt.grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "ghostnet_training_loss.png"),
    dpi=300
)
plt.close()


# -----------------------------
# 4. Individual Validation Loss
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, val_loss, marker="o")

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("GhostNet Validation Loss")
plt.xticks(epochs)
plt.grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "ghostnet_validation_loss.png"),
    dpi=300
)
plt.close()


# -----------------------------
# 5. Individual Training Accuracy
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, train_acc, marker="o")

plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("GhostNet Training Accuracy")
plt.xticks(epochs)
plt.ylim(0.97, 1.005)
plt.grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "ghostnet_training_accuracy.png"),
    dpi=300
)
plt.close()


# -----------------------------
# 6. Individual Validation Accuracy
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, val_acc, marker="o")

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("GhostNet Validation Accuracy")
plt.xticks(epochs)
plt.ylim(0.97, 1.005)
plt.grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "ghostnet_validation_accuracy.png"),
    dpi=300
)
plt.close()


print("========================================")
print("GhostNet training visualizations created")
print("========================================")
print(f"Output folder: {output_dir}")
print()
print("Created:")
print("1. ghostnet_loss_curve.png")
print("2. ghostnet_accuracy_curve.png")
print("3. ghostnet_training_loss.png")
print("4. ghostnet_validation_loss.png")
print("5. ghostnet_training_accuracy.png")
print("6. ghostnet_validation_accuracy.png")