import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

HISTORY_PATH = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "rmscnn_training_history.npz"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "rmscnn"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

print("========================================")
print("RMSCNN TRAINING VISUALIZATION")
print("========================================")

history = np.load(
    HISTORY_PATH
)

print("\nAvailable history keys:")
print(history.files)


# ============================================================
# READ HISTORY
# ============================================================

train_loss = history["train_losses"]
val_loss = history["val_losses"]

train_acc = history["train_accuracies"]
val_acc = history["val_accuracies"]

epochs = range(
    1,
    len(train_loss) + 1
)


print("\nNumber of epochs:", len(train_loss))


# ============================================================
# PRINT TRAINING HISTORY
# ============================================================

print("\nTraining history:")
print("----------------------------------------")

for i in range(len(train_loss)):

    print(
        f"Epoch {i + 1:02d} | "
        f"Train Loss: {train_loss[i]:.4f} | "
        f"Val Loss: {val_loss[i]:.4f} | "
        f"Train Acc: {train_acc[i]:.4f} | "
        f"Val Acc: {val_acc[i]:.4f}"
    )


# ============================================================
# BEST VALIDATION EPOCH
# ============================================================

best_epoch = int(
    np.argmax(val_acc)
) + 1

best_val_acc = float(
    np.max(val_acc)
)


# ============================================================
# 1. LOSS CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "RMSCNN Training and Validation Loss"
)

plt.xticks(
    list(epochs)
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

loss_path = os.path.join(
    RESULT_DIR,
    "rmscnn_loss_curve.png"
)

plt.savefig(
    loss_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. ACCURACY CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    epochs,
    train_acc,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_acc,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "RMSCNN Training and Validation Accuracy"
)

plt.xticks(
    list(epochs)
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

accuracy_path = os.path.join(
    RESULT_DIR,
    "rmscnn_accuracy_curve.png"
)

plt.savefig(
    accuracy_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. SAVE NUMERIC SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULT_DIR,
    "rmscnn_training_summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "RMSCNN TRAINING SUMMARY\n"
    )

    f.write(
        "=======================\n\n"
    )

    f.write(
        f"Total epochs: {len(train_loss)}\n"
    )

    f.write(
        f"Best validation epoch: {best_epoch}\n"
    )

    f.write(
        f"Best validation accuracy: "
        f"{best_val_acc:.4f}\n\n"
    )

    f.write(
        "Epoch | Train Loss | Val Loss | "
        "Train Accuracy | Val Accuracy\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    for i in range(len(train_loss)):

        f.write(
            f"{i + 1:5d} | "
            f"{train_loss[i]:10.4f} | "
            f"{val_loss[i]:8.4f} | "
            f"{train_acc[i]:14.4f} | "
            f"{val_acc[i]:12.4f}\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("VISUALIZATION COMPLETE")
print("========================================")

print("\nBest validation epoch:", best_epoch)

print(
    "Best validation accuracy:",
    f"{best_val_acc:.4f}"
)

print("\nLoss curve:")
print(loss_path)

print("\nAccuracy curve:")
print(accuracy_path)

print("\nTraining summary:")
print(summary_path)