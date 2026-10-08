import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from model_basic_cnn import get_model


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best_basic_cnn_crc.pth"
)

TEST_SAMPLES_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "test_samples.npy"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "basic_cnn_xai",
    "misclassified"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("BASIC CNN MISCLASSIFICATION ANALYSIS")
print("========================================")

print(
    "Device:",
    device
)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = get_model(
    num_classes=2,
    dropout=0.3
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
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

model = model.to(device)
model.eval()

print(
    "\nBest Basic CNN checkpoint loaded."
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD TEST SAMPLES
# ============================================================

test_samples = np.load(
    TEST_SAMPLES_PATH,
    allow_pickle=True
)

print(
    "Test samples:",
    len(test_samples)
)


# ============================================================
# LABEL NAMES
# ============================================================

CLASS_NAMES = {
    0: "CRC",
    1: "Non-CRC"
}


# ============================================================
# FIND MISCLASSIFICATIONS
# ============================================================

misclassified = []

correct = 0
total = 0


print(
    "\n========================================"
)

print(
    "EVALUATING TEST SET"
)

print(
    "========================================"
)


with torch.no_grad():

    for image_path, true_label in test_samples:

        true_label = int(
            true_label
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = transform(
            image
        ).unsqueeze(0).to(
            device
        )

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_label = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_label
        ].item()

        total += 1

        if predicted_label == true_label:

            correct += 1

        else:

            misclassified.append({
                "image_path": image_path,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": confidence
            })


# ============================================================
# TEST RESULTS
# ============================================================

accuracy = correct / total

print(
    "\n========================================"
)

print(
    "TEST SET SUMMARY"
)

print(
    "========================================"
)

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
# MISCLASSIFICATION SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "MISCLASSIFIED IMAGES"
)

print(
    "========================================"
)


if len(misclassified) == 0:

    print(
        "No misclassified images found."
    )

else:

    for index, item in enumerate(
        misclassified,
        start=1
    ):

        print(
            f"\n[{index}]"
        )

        print(
            "Image:",
            os.path.basename(
                item["image_path"]
            )
        )

        print(
            "True:",
            CLASS_NAMES[
                item["true_label"]
            ]
        )

        print(
            "Predicted:",
            CLASS_NAMES[
                item["predicted_label"]
            ]
        )

        print(
            f"Confidence: "
            f"{item['confidence']:.4f}"
        )


# ============================================================
# VISUALIZE MISCLASSIFICATIONS
# ============================================================

print(
    "\n========================================"
)

print(
    "CREATING VISUALIZATIONS"
)

print(
    "========================================"
)


for index, item in enumerate(
    misclassified,
    start=1
):

    image = Image.open(
        item["image_path"]
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image
    )

    true_name = CLASS_NAMES[
        item["true_label"]
    ]

    predicted_name = CLASS_NAMES[
        item["predicted_label"]
    ]

    confidence = item[
        "confidence"
    ]

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        image_array
    )

    plt.title(
        f"True: {true_name}\n"
        f"Predicted: {predicted_name}\n"
        f"Confidence: {confidence:.4f}"
    )

    plt.axis(
        "off"
    )

    base_name = os.path.splitext(
        os.path.basename(
            item["image_path"]
        )
    )[0]

    output_path = os.path.join(
        RESULT_DIR,
        f"{index:02d}_{base_name}_misclassified.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    RESULT_DIR,
    "misclassification_summary.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BASIC CNN MISCLASSIFICATION ANALYSIS\n"
    )

    f.write(
        "====================================\n\n"
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
        f"Test accuracy: "
        f"{accuracy:.4f}\n\n"
    )

    f.write(
        "MISCLASSIFIED IMAGES\n"
    )

    f.write(
        "--------------------\n\n"
    )

    for index, item in enumerate(
        misclassified,
        start=1
    ):

        f.write(
            f"{index}. "
            f"{os.path.basename(item['image_path'])}\n"
        )

        f.write(
            f"   True label: "
            f"{CLASS_NAMES[item['true_label']]}\n"
        )

        f.write(
            f"   Predicted label: "
            f"{CLASS_NAMES[item['predicted_label']]}\n"
        )

        f.write(
            f"   Confidence: "
            f"{item['confidence']:.4f}\n\n"
        )


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n========================================"
)

print(
    "MISCLASSIFICATION ANALYSIS COMPLETE"
)

print(
    "========================================"
)

print(
    f"Total test images: {total}"
)

print(
    f"Misclassified: {len(misclassified)}"
)

print(
    "\nResults saved to:"
)

print(
    RESULT_DIR
)

print(
    "\nReport saved to:"
)

print(
    report_path
)