import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from PIL import Image

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

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "ghostnet"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("GHOSTNET MISCLASSIFICATION ANALYSIS")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
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

print(
    "Checkpoint loaded:",
    CHECKPOINT_PATH
)


# ============================================================
# TEST DATASET
# ============================================================

print("\nLoading test dataset...")

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
    "Total test images:",
    len(test_dataset)
)


# ============================================================
# FIND MISCLASSIFIED IMAGES
# ============================================================

misclassified = []

sample_index = 0


print("\nEvaluating test images...")


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

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        for batch_index in range(
            images.size(0)
        ):

            actual = labels[
                batch_index
            ].item()

            predicted = predictions[
                batch_index
            ].item()

            confidence = probabilities[
                batch_index,
                predicted
            ].item()


            if actual != predicted:

                image_path, original_label = (
                    test_samples[sample_index]
                )


                misclassified.append(
                    {
                        "index": sample_index,
                        "path": image_path,
                        "actual": actual,
                        "predicted": predicted,
                        "confidence": confidence
                    }
                )


            sample_index += 1


# ============================================================
# SUMMARY
# ============================================================

total_images = len(test_samples)

num_errors = len(
    misclassified
)

num_correct = (
    total_images
    - num_errors
)

accuracy = (
    num_correct
    / total_images
)


print("\n" + "=" * 60)
print("MISCLASSIFICATION SUMMARY")
print("=" * 60)

print(
    f"Total test images : {total_images}"
)

print(
    f"Correct           : {num_correct}"
)

print(
    f"Misclassified     : {num_errors}"
)

print(
    f"Accuracy          : {accuracy:.4f}"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def class_name(label):

    if label == 0:
        return "CRC"

    return "Non-CRC"


# ============================================================
# PRINT MISCLASSIFICATIONS
# ============================================================

if num_errors == 0:

    print(
        "\nNo misclassified images found."
    )

else:

    print(
        "\nMisclassified images:"
    )

    for number, item in enumerate(
        misclassified,
        start=1
    ):

        print(
            f"{number}. "
            f"Actual={class_name(item['actual'])}, "
            f"Predicted={class_name(item['predicted'])}, "
            f"Confidence={item['confidence']:.4f}"
        )

        print(
            f"   Image: {item['path']}"
        )


# ============================================================
# SAVE INDIVIDUAL MISCLASSIFIED IMAGES
# ============================================================

for number, item in enumerate(
    misclassified,
    start=1
):

    image = Image.open(
        item["path"]
    ).convert("RGB")

    image_array = np.array(
        image
    )


    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        image_array
    )

    plt.title(
        f"Actual: {class_name(item['actual'])}\n"
        f"Predicted: {class_name(item['predicted'])}\n"
        f"Confidence: {item['confidence']:.4f}"
    )

    plt.axis("off")

    output_path = (
        OUTPUT_DIR
        / f"ghostnet_misclassified_{number}.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# SUMMARY GRID
# ============================================================

if num_errors > 0:

    columns = min(
        3,
        num_errors
    )

    rows = (
        num_errors + columns - 1
    ) // columns


    plt.figure(
        figsize=(
            6 * columns,
            5 * rows
        )
    )


    for number, item in enumerate(
        misclassified,
        start=1
    ):

        image = Image.open(
            item["path"]
        ).convert("RGB")

        image_array = np.array(
            image
        )


        plt.subplot(
            rows,
            columns,
            number
        )

        plt.imshow(
            image_array
        )

        plt.title(
            f"Actual: {class_name(item['actual'])}\n"
            f"Predicted: {class_name(item['predicted'])}\n"
            f"Confidence: {item['confidence']:.4f}"
        )

        plt.axis("off")


    plt.tight_layout()


    grid_path = (
        OUTPUT_DIR
        / "ghostnet_misclassified_grid.png"
    )

    plt.savefig(
        grid_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Saved:",
        grid_path
    )


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "ghostnet_misclassification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GHOSTNET MISCLASSIFICATION ANALYSIS\n"
    )

    f.write(
        "====================================\n\n"
    )

    f.write(
        f"Total test images: "
        f"{total_images}\n"
    )

    f.write(
        f"Correct: "
        f"{num_correct}\n"
    )

    f.write(
        f"Misclassified: "
        f"{num_errors}\n"
    )

    f.write(
        f"Accuracy: "
        f"{accuracy:.4f}\n\n"
    )


    for number, item in enumerate(
        misclassified,
        start=1
    ):

        f.write(
            f"{number}. "
            f"Actual={class_name(item['actual'])}, "
            f"Predicted={class_name(item['predicted'])}, "
            f"Confidence={item['confidence']:.4f}\n"
        )

        f.write(
            f"   Image: {item['path']}\n\n"
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("GHOSTNET MISCLASSIFICATION ANALYSIS COMPLETE")
print("=" * 60)

print(
    "Results saved to:",
    OUTPUT_DIR
)

print(
    "Report saved to:",
    report_path
)