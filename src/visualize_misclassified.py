import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader

from dataset import ColonDataset, val_test_transform
from split_dataset import test_samples
from model import create_model


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ==================================================
# TEST DATASET
# ==================================================

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


# ==================================================
# MODEL
# ==================================================

model = create_model(
    num_classes=2
)

model.load_state_dict(
    torch.load(
        "best_resnet18_crc.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ==================================================
# CLASS NAMES
# ==================================================

class_names = {
    0: "CRC",
    1: "Non-CRC"
}


# ==================================================
# FIND MISCLASSIFIED IMAGES
# ==================================================

misclassified = []

sample_index = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for i in range(len(images)):

            actual = labels[i].item()

            predicted = predictions[i].item()

            if actual != predicted:

                image_path = test_samples[
                    sample_index
                ][0]

                confidence = probabilities[
                    i,
                    predicted
                ].item()

                misclassified.append({
                    "path": image_path,
                    "actual": actual,
                    "predicted": predicted,
                    "confidence": confidence
                })

            sample_index += 1


# ==================================================
# PRINT SUMMARY
# ==================================================

print()
print("========================================")
print("   RESNET18 MISCLASSIFICATION ANALYSIS")
print("========================================")

print(
    "Total misclassified:",
    len(misclassified)
)


# ==================================================
# VISUALIZATION
# ==================================================

if len(misclassified) == 0:

    print()
    print("No misclassified images found.")

else:

    # --------------------------------------------------
    # Maximum 6 images for presentation
    # --------------------------------------------------

    samples_to_show = misclassified[:6]

    number = len(samples_to_show)

    columns = 3

    rows = (
        number + columns - 1
    ) // columns


    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(18, 12),
        squeeze=False
    )

    axes = axes.flatten()


    # --------------------------------------------------
    # Display images
    # --------------------------------------------------

    for i, sample in enumerate(
        samples_to_show
    ):

        image = Image.open(
            sample["path"]
        ).convert("RGB")

        axes[i].imshow(
            image
        )

        axes[i].axis("off")


        # --------------------------------------------------
        # Information BELOW image
        # --------------------------------------------------

        actual_text = class_names[
            sample["actual"]
        ]

        predicted_text = class_names[
            sample["predicted"]
        ]

        confidence = (
            sample["confidence"] * 100
        )


        info = (
            f"Actual: {actual_text}\n"
            f"Predicted: {predicted_text}\n"
            f"Confidence: {confidence:.2f}%"
        )


        axes[i].text(
            0.5,
            -0.08,
            info,
            transform=axes[i].transAxes,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold"
        )


    # --------------------------------------------------
    # Hide unused axes
    # --------------------------------------------------

    for i in range(
        number,
        len(axes)
    ):

        axes[i].axis("off")


    # --------------------------------------------------
    # Main title
    # --------------------------------------------------

    fig.suptitle(
        "ResNet18 Misclassified Test Images",
        fontsize=24,
        fontweight="bold",
        y=0.98
    )


    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    plt.subplots_adjust(
        top=0.90,
        bottom=0.08,
        hspace=0.35,
        wspace=0.08
    )


    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    plt.savefig(
        "resnet18_misclassified_clear.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------
    # Complete
    # --------------------------------------------------

    print()
    print(
        "===== MISCLASSIFICATION VISUALIZATION ====="
    )

    print(
        "Saved:"
    )

    print(
        "resnet18_misclassified_clear.png"
    )