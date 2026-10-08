import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATH SETUP
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.append(str(SRC_DIR))


# ============================================================
# IMPORTS
# ============================================================

from model_ghostnet import get_model
from dataset import ColonDataset, val_test_transform
from split_dataset import test_samples


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_WORKERS = 0

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "best_ghostnet_crc.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("GHOSTNET TEST EVALUATION")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# TEST DATASET
# ============================================================

print("\nLoading test dataset...")

print(
    "Testing samples:",
    len(test_samples)
)

test_dataset = ColonDataset(
    test_samples,
    transform=val_test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print(
    "Test batches:",
    len(test_loader)
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading GhostNet model...")

model = get_model(
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Checkpoint loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []


print("\nEvaluating test set...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)


        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        # ----------------------------------------------------
        # Store labels
        # ----------------------------------------------------

        all_labels.extend(
            labels.cpu().numpy()
        )


        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        all_predictions.extend(
            predictions.cpu().numpy()
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # Dataset mapping:
        #
        # 0 = CRC
        # 1 = Non-CRC
        #
        # roc_auc_score expects the probability
        # of the positive class (class 1).
        # ----------------------------------------------------

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
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
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("GHOSTNET TEST RESULTS")
print("=" * 60)

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
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "CRC",
            "Non-CRC"
        ],
        zero_division=0
    )
)


# ============================================================
# FINAL
# ============================================================

print("=" * 60)
print("GHOSTNET TEST EVALUATION COMPLETE")
print("=" * 60)