import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import ColonDataset, val_test_transform
from split_dataset import test_samples
from model_densenet121 import create_densenet121


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("     DENSENET121 TEST EVALUATION")
print("========================================")

print(f"Device : {device}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = ColonDataset(
    test_samples,
    transform=val_test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print(f"Test samples : {len(test_dataset)}")


# ============================================================
# MODEL
# ============================================================

model = create_densenet121(num_classes=2)

model.load_state_dict(
    torch.load(
        "best_densenet121_crc.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("========================================")
print("        DENSENET121 TEST RESULTS")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("Classification Report:")
print()

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["CRC", "Non-CRC"],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix:")
print(cm)


# ============================================================
# RESULTS DIRECTORY
# ============================================================

output_dir = Path("results/densenet121")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("DenseNet121 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    ["CRC", "Non-CRC"]
)

plt.yticks(
    [0, 1],
    ["CRC", "Non-CRC"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    output_dir / "densenet121_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    all_labels,
    all_probabilities
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("DenseNet121 ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    output_dir / "densenet121_roc_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE METRICS
# ============================================================

metrics_file = output_dir / "metrics.txt"

with open(metrics_file, "w") as f:

    f.write("DENSENET121 TEST RESULTS\n")
    f.write("========================\n\n")

    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-score  : {f1:.4f}\n")
    f.write(f"ROC-AUC   : {roc_auc:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n")


# ============================================================
# COMPLETE
# ============================================================

print()
print("========================================")
print("       EVALUATION COMPLETE")
print("========================================")

print()
print("Saved to:")
print(output_dir)

print()
print("Files created:")
print("densenet121_confusion_matrix.png")
print("densenet121_roc_curve.png")
print("metrics.txt")