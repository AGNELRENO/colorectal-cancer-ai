import os
import numpy as np
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

from dataset import ColonDataset, val_test_transform
from model_basic_cnn import get_model


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

TEST_SAMPLES_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "test_samples.npy"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best_basic_cnn_crc.pth"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "basic_cnn"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

RESULTS_PATH = os.path.join(
    RESULT_DIR,
    "basic_cnn_test_results.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32
NUM_CLASSES = 2


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
# LOAD TEST SAMPLES
# ============================================================

test_samples = np.load(
    TEST_SAMPLES_PATH,
    allow_pickle=True
)

print("\n========================================")
print("BASIC CNN TEST EVALUATION")
print("========================================")

print(
    "Testing samples:",
    len(test_samples)
)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = ColonDataset(
    test_samples,
    transform=val_test_transform
)


# ============================================================
# TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD MODEL
# ============================================================

model = get_model(
    num_classes=NUM_CLASSES,
    dropout=0.3
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


print(
    "\nBest checkpoint loaded successfully."
)

print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation accuracy:",
    f'{checkpoint["val_accuracy"]:.4f}'
)


# ============================================================
# PREDICTION
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []


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

        outputs = model(
            images
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # Store true labels
        all_labels.extend(
            labels.cpu().numpy()
        )

        # Store predictions
        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # IMPORTANT:
        # Class 0 = CRC
        # Class 1 = Non-CRC
        #
        # sklearn ROC-AUC treats class 1 as the
        # positive class, therefore use probability
        # of class 1.

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

all_probabilities = np.array(
    all_probabilities
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
    pos_label=0,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    pos_label=0,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    pos_label=0,
    zero_division=0
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=[
        "CRC",
        "Non-CRC"
    ],
    digits=4,
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("BASIC CNN TEST RESULTS")
print("========================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print("\nConfusion Matrix:")

print(
    confusion
)

print("\nClassification Report:")

print(
    report
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    RESULTS_PATH,
    "w"
) as f:

    f.write(
        "BASIC CNN TEST RESULTS\n"
    )

    f.write(
        "======================\n\n"
    )

    f.write(
        "Model: 3-Layer Basic CNN\n"
    )

    f.write(
        "Input: 224 x 224 x 3\n"
    )

    f.write(
        "Features: 128\n"
    )

    f.write(
        "Dropout: 0.3\n\n"
    )

    f.write(
        f"Best Epoch: "
        f'{checkpoint["epoch"]}\n'
    )

    f.write(
        f"Best Validation Accuracy: "
        f'{checkpoint["val_accuracy"]:.4f}\n\n'
    )

    f.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall   : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score : {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC  : {roc_auc:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(confusion)
    )

    f.write(
        "\n\nClassification Report:\n"
    )

    f.write(
        report
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("BASIC CNN EVALUATION COMPLETE")
print("========================================")

print(
    "Results saved to:"
)

print(
    RESULTS_PATH
)