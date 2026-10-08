import os
import numpy as np
import torch
import torch.nn.functional as F

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
from model_rmscnn import get_model


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

TEST_SAMPLES = os.path.join(
    PROJECT_DIR,
    "data",
    "test_samples.npy"
)

CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best_rmscnn_crc.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "rmscnn"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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
    TEST_SAMPLES,
    allow_pickle=True
)

print("\n========================================")
print("RMSCNN TEST DATASET")
print("========================================")

print(
    "Test samples:",
    len(test_samples)
)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = ColonDataset(
    test_samples,
    transform=val_test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# CREATE MODEL
# ============================================================

model = get_model(
    num_classes=2
)

model = model.to(device)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "\nBest checkpoint loaded successfully."
)

print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    "Validation accuracy:",
    f"{checkpoint['val_accuracy']:.4f}"
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

        outputs = model(images)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )


        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Class 1 = Non-CRC
        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


# Convert to NumPy arrays

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


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("RMSCNN TEST RESULTS")
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
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


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
        digits=4,
        zero_division=0
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "rmscnn_test_results.txt"
)


with open(
    results_file,
    "w"
) as f:

    f.write(
        "RMSCNN TEST RESULTS\n"
    )

    f.write(
        "=" * 50 + "\n\n"
    )

    f.write(
        f"Test samples: {len(all_labels)}\n"
    )

    f.write(
        f"Best epoch: {checkpoint['epoch']}\n"
    )

    f.write(
        f"Best validation accuracy: "
        f"{checkpoint['val_accuracy']:.4f}\n\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall: {recall:.4f}\n"
    )

    f.write(
        f"F1 Score: {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC: {roc_auc:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\nClassification Report:\n"
    )

    f.write(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "CRC",
                "Non-CRC"
            ],
            digits=4,
            zero_division=0
        )
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "rmscnn_test_labels.npy"
    ),
    all_labels
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "rmscnn_test_predictions.npy"
    ),
    all_predictions
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "rmscnn_test_probabilities.npy"
    ),
    all_probabilities
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("RMSCNN EVALUATION COMPLETE")
print("========================================")

print(
    "Results saved to:",
    results_file
)