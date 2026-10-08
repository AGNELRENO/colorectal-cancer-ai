import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
history_path = Path("checkpoints/basic_cnn_training_history.npz")
output_dir = Path("results/basic_cnn")
output_dir.mkdir(parents=True, exist_ok=True)

# Load training history
history = np.load(history_path)

train_losses = history["train_losses"]
train_accuracies = history["train_accuracies"]
val_losses = history["val_losses"]
val_accuracies = history["val_accuracies"]

epochs = range(1, len(train_losses) + 1)

# -----------------------------
# Loss Curve
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, train_losses, marker="o", label="Training Loss")
plt.plot(epochs, val_losses, marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Basic CNN Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

loss_path = output_dir / "basic_cnn_loss_curve.png"
plt.savefig(loss_path, dpi=300)
plt.close()

# -----------------------------
# Accuracy Curve
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, train_accuracies, marker="o", label="Training Accuracy")
plt.plot(epochs, val_accuracies, marker="o", label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Basic CNN Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

accuracy_path = output_dir / "basic_cnn_accuracy_curve.png"
plt.savefig(accuracy_path, dpi=300)
plt.close()

# -----------------------------
# Summary
# -----------------------------
best_epoch = int(np.argmax(val_accuracies)) + 1
best_val_accuracy = float(np.max(val_accuracies))

summary_path = output_dir / "basic_cnn_training_summary.txt"

with open(summary_path, "w") as f:
    f.write("BASIC CNN TRAINING SUMMARY\n")
    f.write("==========================\n\n")
    f.write(f"Total Epochs: {len(train_losses)}\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Best Validation Accuracy: {best_val_accuracy:.4f}\n\n")

    f.write("Epoch-wise Results\n")
    f.write("------------------\n")

    for i in range(len(train_losses)):
        f.write(
            f"Epoch {i+1}: "
            f"Train Loss={train_losses[i]:.4f}, "
            f"Train Acc={train_accuracies[i]:.4f}, "
            f"Val Loss={val_losses[i]:.4f}, "
            f"Val Acc={val_accuracies[i]:.4f}\n"
        )

print("Basic CNN training visualization completed.")
print(f"Loss curve: {loss_path}")
print(f"Accuracy curve: {accuracy_path}")
print(f"Summary: {summary_path}")