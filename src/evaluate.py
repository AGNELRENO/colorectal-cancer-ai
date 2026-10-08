import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
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
from dataset import ColonDataset, val_test_transform
from split_dataset import test_samples
from model import create_model


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Test dataset
# --------------------------------------------------

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


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = create_model(
    num_classes=2
)

# Load trained weights
model.load_state_dict(
    torch.load(
        "best_resnet18_crc.pth",
        map_location=device
    )
)

model = model.to(device)

# Evaluation mode
model.eval()


# --------------------------------------------------
# Store predictions
# --------------------------------------------------

all_labels = []
all_predictions = []
all_probabilities = []


# --------------------------------------------------
# Test loop
# --------------------------------------------------

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

        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro"
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("========================================")
print("       RESNET18 TEST RESULTS")
print("========================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print()
print("Confusion Matrix")

print(cm)

print()
print("Classification Report")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "CRC",
            "Non-CRC"
        ]
    )
)


# ==================================================
# CONFUSION MATRIX VISUALIZATION
# ==================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "ResNet18 Confusion Matrix"
)

plt.colorbar()


# --------------------------------------------------
# Class labels
# --------------------------------------------------

class_names = [
    "CRC",
    "Non-CRC"
]


plt.xticks(
    [0, 1],
    class_names
)

plt.yticks(
    [0, 1],
    class_names
)


# --------------------------------------------------
# Axis labels
# --------------------------------------------------

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)


# --------------------------------------------------
# Display values inside matrix
# --------------------------------------------------

threshold = cm.max() / 2.0

for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            color="white"
            if cm[i, j] > threshold
            else "black",
            fontsize=14
        )


# --------------------------------------------------
# Save visualization
# --------------------------------------------------

plt.tight_layout()

plt.savefig(
    "resnet18_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Visualization complete
# --------------------------------------------------

print()
print("===== CONFUSION MATRIX VISUALIZATION =====")

print(
    "Saved:"
)

print(
    "resnet18_confusion_matrix.png"
)
# ==================================================
# ROC CURVE
# ==================================================

fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probabilities
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"ResNet18 (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ResNet18 ROC Curve"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "resnet18_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "3. resnet18_roc_curve.png"
)