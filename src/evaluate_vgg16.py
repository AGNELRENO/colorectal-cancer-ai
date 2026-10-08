import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

from dataset import ColonDataset, val_test_transform
from model_vgg16 import create_vgg16
from split_dataset import test_samples


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. TEST DATASET
# ============================================================

test_dataset = ColonDataset(
    test_samples,
    transform=val_test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print()
print("===== TEST DATASET =====")
print("Test samples:", len(test_dataset))


# ============================================================
# 3. LOAD BEST VGG16 MODEL
# ============================================================

model = create_vgg16(num_classes=2)

checkpoint_path = Path(
    "checkpoints/best_vgg16_crc.pth"
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Checkpoint loaded:", checkpoint_path)


# ============================================================
# 4. TEST PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Probability of CRC class
        # CRC = class 0
        all_probabilities.extend(
            probabilities[:, 0]
            .cpu()
            .numpy()
        )


y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)


# ============================================================
# 5. METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=0,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=0,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=0,
    zero_division=0
)

# Convert CRC class 0 into the positive class
# for ROC-AUC calculation.
y_true_crc = (
    y_true == 0
).astype(int)

roc_auc = roc_auc_score(
    y_true_crc,
    y_prob
)


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print()
print("===== VGG16 TEST RESULTS =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# 7. RESULTS DIRECTORY
# ============================================================

results_dir = Path(
    "results/vgg16"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 8. SAVE METRICS
# ============================================================

metrics_file = (
    results_dir / "metrics.txt"
)

with open(metrics_file, "w") as f:

    f.write("VGG16 Test Results\n")
    f.write("==================\n")
    f.write(f"Test samples: {len(test_dataset)}\n\n")

    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n")
    f.write(f"ROC-AUC   : {roc_auc:.4f}\n")


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print()
print("Confusion Matrix:")
print(cm)

fig, ax = plt.subplots(
    figsize=(6, 6)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["CRC", "Non-CRC"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

ax.set_title(
    "VGG16 Confusion Matrix"
)

plt.tight_layout()

cm_path = (
    results_dir /
    "vgg16_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_true_crc,
    y_prob
)

plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"VGG16 (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "VGG16 ROC Curve"
)

plt.legend(
    loc="lower right"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

roc_path = (
    results_dir /
    "vgg16_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. SAVE PREDICTIONS
# ============================================================

np.save(
    results_dir / "test_labels.npy",
    y_true
)

np.save(
    results_dir / "test_predictions.npy",
    y_pred
)

np.save(
    results_dir / "test_probabilities.npy",
    y_prob
)


# ============================================================
# 12. COMPLETE
# ============================================================

print()
print("===== FILES SAVED =====")

print(metrics_file)
print(cm_path)
print(roc_path)

print()
print("VGG16 evaluation completed successfully.")