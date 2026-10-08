import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

import sys

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
    "gradcam_results"
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
print("BASIC CNN GRAD-CAM")
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

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

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
    "\nModel loaded successfully."
)


# ============================================================
# TARGET LAYER
# ============================================================

target_layer = model.conv3

activations = None
gradients = None


def forward_hook(
    module,
    input,
    output
):
    global activations
    activations = output


def backward_hook(
    module,
    grad_input,
    grad_output
):
    global gradients
    gradients = grad_output[0]


forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
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
# FIND SAMPLE BY CLASS
# ============================================================

crc_samples = []

non_crc_samples = []

for sample in test_samples:

    image_path = sample[0]
    label = int(sample[1])

    if label == 0:
        crc_samples.append(
            (image_path, label)
        )

    elif label == 1:
        non_crc_samples.append(
            (image_path, label)
        )


# Select first 3 from each class
selected_samples = (
    crc_samples[:3] +
    non_crc_samples[:3]
)


print(
    "\nSelected samples:",
    len(selected_samples)
)

print(
    "CRC samples:",
    len(crc_samples[:3])
)

print(
    "Non-CRC samples:",
    len(non_crc_samples[:3])
)


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(
    image_tensor,
    target_class
):

    global activations
    global gradients

    activations = None
    gradients = None

    model.zero_grad()

    output = model(
        image_tensor
    )

    score = output[
        0,
        target_class
    ]

    score.backward()

    # --------------------------------------------------------
    # Get activations and gradients
    # --------------------------------------------------------

    activation = activations[0]

    gradient = gradients[0]

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    weights = gradient.mean(
        dim=(1, 2)
    )

    # --------------------------------------------------------
    # Weighted combination
    # --------------------------------------------------------

    cam = torch.zeros(
        activation.shape[1:],
        device=device
    )

    for channel in range(
        activation.shape[0]
    ):

        cam += (
            weights[channel] *
            activation[channel]
        )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = F.relu(
        cam
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    cam -= cam.min()

    if cam.max() > 0:

        cam /= cam.max()

    # --------------------------------------------------------
    # Resize to image size
    # --------------------------------------------------------

    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().detach().cpu().numpy()

    return cam, output


# ============================================================
# PROCESS SAMPLES
# ============================================================

print(
    "\n========================================"
)

print(
    "GENERATING GRAD-CAM"
)

print(
    "========================================"
)


results = []


for index, (
    image_path,
    true_label
) in enumerate(
    selected_samples
):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    original_image = image.resize(
        (224, 224)
    )

    image_tensor = transform(
        image
    ).unsqueeze(0).to(
        device
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
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

    # --------------------------------------------------------
    # Generate CAM for predicted class
    # --------------------------------------------------------

    cam, _ = generate_gradcam(
        image_tensor,
        predicted_class
    )

    # --------------------------------------------------------
    # Prepare overlay
    # --------------------------------------------------------

    original_array = np.array(
        original_image
    ) / 255.0

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Original
    axes[0].imshow(
        original_array
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis(
        "off"
    )

    # Heatmap
    axes[1].imshow(
        cam,
        cmap="jet"
    )

    axes[1].set_title(
        "Grad-CAM"
    )

    axes[1].axis(
        "off"
    )

    # Overlay
    axes[2].imshow(
        original_array
    )

    axes[2].imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    true_name = (
        "CRC"
        if true_label == 0
        else "Non-CRC"
    )

    predicted_name = (
        "CRC"
        if predicted_class == 0
        else "Non-CRC"
    )

    axes[2].set_title(
        f"Overlay\n"
        f"True: {true_name} | "
        f"Pred: {predicted_name}\n"
        f"Confidence: {confidence:.4f}"
    )

    axes[2].axis(
        "off"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = os.path.join(
        RESULT_DIR,
        f"{index + 1:02d}_{base_name}_gradcam.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({
        "image": image_path,
        "true_label": true_name,
        "predicted_label": predicted_name,
        "confidence": confidence,
        "output": output_path
    })

    print(
        f"\n[{index + 1}/6]"
    )

    print(
        "Image:",
        os.path.basename(image_path)
    )

    print(
        "True:",
        true_name
    )

    print(
        "Predicted:",
        predicted_name
    )

    print(
        f"Confidence: {confidence:.4f}"
    )

    print(
        "Saved:",
        output_path
    )


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULT_DIR,
    "gradcam_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BASIC CNN GRAD-CAM RESULTS\n"
    )

    f.write(
        "==========================\n\n"
    )

    f.write(
        "Target layer: conv3\n"
    )

    f.write(
        "Number of images: 6\n\n"
    )

    for result in results:

        f.write(
            f"Image: "
            f"{os.path.basename(result['image'])}\n"
        )

        f.write(
            f"True label: "
            f"{result['true_label']}\n"
        )

        f.write(
            f"Predicted label: "
            f"{result['predicted_label']}\n"
        )

        f.write(
            f"Confidence: "
            f"{result['confidence']:.4f}\n"
        )

        f.write(
            f"Grad-CAM: "
            f"{result['output']}\n\n"
        )


# ============================================================
# REMOVE HOOKS
# ============================================================

forward_handle.remove()
backward_handle.remove()


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n========================================"
)

print(
    "GRAD-CAM COMPLETE"
)

print(
    "========================================"
)

print(
    "\nTarget layer: conv3"
)

print(
    "Images generated:",
    len(results)
)

print(
    "\nResults saved to:"
)

print(
    RESULT_DIR
)

print(
    "\nSummary saved to:"
)

print(
    summary_path
)