import sys
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms

sys.path.append(str(Path(__file__).resolve().parent))

from split_dataset import test_samples


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = Path(
    "checkpoints/best_vgg16_crc.pth"
)

OUTPUT_DIR = Path(
    "results/vgg16"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224

CLASS_NAMES = {
    0: "CRC",
    1: "Non-CRC"
}


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD VGG16
# ============================================================

print("===== LOADING VGG16 =====")

model = models.vgg16(
    weights=None
)

model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    2
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)
model.eval()


print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# EVALUATE EXACT TEST SPLIT
# ============================================================

print()
print(
    "===== EVALUATING EXACT TEST SPLIT ====="
)

misclassified = []

correct = 0
total = 0


for image_path, true_label in test_samples:

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()


    total += 1

    if predicted_class == true_label:

        correct += 1

    else:

        misclassified.append({
            "path": image_path,
            "true": true_label,
            "predicted": predicted_class,
            "confidence": confidence
        })


accuracy = correct / total


print(
    f"Total test images : {total}"
)

print(
    f"Correct           : {correct}"
)

print(
    f"Misclassified     : {len(misclassified)}"
)

print(
    f"Accuracy          : {accuracy:.4f}"
)


# ============================================================
# DISPLAY MISCLASSIFIED IMAGES
# ============================================================

print()
print(
    "===== MISCLASSIFIED IMAGES ====="
)


if len(misclassified) == 0:

    print(
        "No misclassified images found."
    )

else:

    print(
        f"Found {len(misclassified)} "
        "misclassified images."
    )


    # --------------------------------------------------------
    # Create one image for each misclassification
    # --------------------------------------------------------

    for index, item in enumerate(
        misclassified,
        start=1
    ):

        image = Image.open(
            item["path"]
        ).convert("RGB")

        display_image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        fig, ax = plt.subplots(
            figsize=(5, 5)
        )

        ax.imshow(
            display_image
        )

        ax.set_title(
            f"Actual: {CLASS_NAMES[item['true']]}\n"
            f"Predicted: {CLASS_NAMES[item['predicted']]} | "
            f"Confidence: {item['confidence']:.2%}"
        )

        ax.axis("off")

        output_file = (
            OUTPUT_DIR
            / f"vgg16_misclassified_{index}.png"
        )

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"{index}. "
            f"Actual={CLASS_NAMES[item['true']]}, "
            f"Predicted={CLASS_NAMES[item['predicted']]}, "
            f"Confidence={item['confidence']:.4f}"
        )

        print(
            "   Saved:",
            output_file
        )


# ============================================================
# CREATE SUMMARY GRID
# ============================================================

if len(misclassified) > 0:

    max_images = len(misclassified)

    columns = 3
    rows = (
        max_images + columns - 1
    ) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            columns * 5,
            rows * 5
        )
    )

    # Make axes iterable
    if max_images == 1:

        axes = [axes]

    else:

        axes = axes.flatten()


    for index, item in enumerate(
        misclassified
    ):

        image = Image.open(
            item["path"]
        ).convert("RGB")

        display_image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        axes[index].imshow(
            display_image
        )

        axes[index].set_title(
            f"Actual: {CLASS_NAMES[item['true']]}\n"
            f"Predicted: {CLASS_NAMES[item['predicted']]}\n"
            f"Confidence: {item['confidence']:.2%}"
        )

        axes[index].axis("off")


    # Hide unused axes
    for index in range(
        max_images,
        len(axes)
    ):

        axes[index].axis("off")


    fig.suptitle(
        "VGG16 Misclassified Test Images",
        fontsize=16
    )

    plt.tight_layout()

    summary_file = (
        OUTPUT_DIR
        / "vgg16_misclassified_grid.png"
    )

    plt.savefig(
        summary_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Summary grid saved:",
        summary_file
    )


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

summary_file = (
    OUTPUT_DIR
    / "vgg16_misclassified_summary.txt"
)

with open(
    summary_file,
    "w"
) as f:

    f.write(
        "VGG16 MISCLASSIFICATION ANALYSIS\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        f"Total test images: {total}\n"
    )

    f.write(
        f"Correct predictions: {correct}\n"
    )

    f.write(
        f"Misclassified images: "
        f"{len(misclassified)}\n"
    )

    f.write(
        f"Test accuracy: {accuracy:.4f}\n\n"
    )

    for index, item in enumerate(
        misclassified,
        start=1
    ):

        f.write(
            f"{index}. "
            f"Actual={CLASS_NAMES[item['true']]}, "
            f"Predicted={CLASS_NAMES[item['predicted']]}, "
            f"Confidence={item['confidence']:.4f}, "
            f"Path={item['path']}\n"
        )


print()
print(
    "===== VGG16 MISCLASSIFICATION "
    "VISUALIZATION COMPLETE ====="
)

print(
    "Results saved to:",
    OUTPUT_DIR
)