import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


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
from split_dataset import test_samples


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "best_ghostnet_crc.pth"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "ghostnet_xai"
    / "gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_CLASSES = 2

NUM_IMAGES_PER_CLASS = 3


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("GHOSTNET GRAD-CAM")
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

print("\nLoading GhostNet...")

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
# FIND TARGET LAYER
# ============================================================

# GhostNet's final convolutional feature block
# is used as the Grad-CAM target layer.

target_layer = model.blocks[-1]


# ============================================================
# HOOK STORAGE
# ============================================================

activations = None
gradients = None


# ============================================================
# FORWARD HOOK
# ============================================================

def forward_hook(
    module,
    input,
    output
):

    global activations

    activations = output


# ============================================================
# BACKWARD HOOK
# ============================================================

def backward_hook(
    module,
    grad_input,
    grad_output
):

    global gradients

    gradients = grad_output[0]


# ============================================================
# REGISTER HOOKS
# ============================================================

forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),
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
    ]
)


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(
    image_path,
    actual_label,
    output_path
):

    global activations
    global gradients


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    input_tensor = transform(
        original_image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        device
    )

    input_tensor.requires_grad_(True)


    # --------------------------------------------------------
    # Clear gradients
    # --------------------------------------------------------

    model.zero_grad()

    activations = None
    gradients = None


    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(
        input_tensor
    )


    probabilities = torch.softmax(
        output,
        dim=1
    )


    predicted_class = torch.argmax(
        output,
        dim=1
    ).item()


    confidence = probabilities[
        0,
        predicted_class
    ].item()


    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    score = output[
        0,
        predicted_class
    ]

    score.backward()


    # --------------------------------------------------------
    # Get activations and gradients
    # --------------------------------------------------------

    if activations is None:

        raise RuntimeError(
            "Grad-CAM activations were not captured."
        )

    if gradients is None:

        raise RuntimeError(
            "Grad-CAM gradients were not captured."
        )


    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    weights = torch.mean(
        gradients,
        dim=(2, 3),
        keepdim=True
    )


    # --------------------------------------------------------
    # Weighted activation maps
    # --------------------------------------------------------

    cam = torch.sum(
        weights * activations,
        dim=1
    )


    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = F.relu(
        cam
    )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    cam = cam.squeeze(
        0
    ).detach().cpu().numpy()


    # --------------------------------------------------------
    # Normalize CAM
    # --------------------------------------------------------

    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max - cam_min != 0:

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min
        )

    else:

        cam = np.zeros_like(
            cam
        )


    # --------------------------------------------------------
    # Resize CAM to original image size
    # --------------------------------------------------------

    original_array = np.array(
        original_image
    )

    height, width = (
        original_array.shape[:2]
    )

    cam = cv2.resize(
        cam,
        (width, height)
    )


    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = (
        0.4 * heatmap
        + 0.6 * original_array
    )

    overlay = np.uint8(
        np.clip(
            overlay,
            0,
            255
        )
    )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    plt.figure(
        figsize=(15, 5)
    )


    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original_array
    )

    plt.title(
        f"Original\nActual: "
        f"{'CRC' if actual_label == 0 else 'Non-CRC'}"
    )

    plt.axis("off")


    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        cam,
        cmap="jet"
    )

    plt.title(
        "Grad-CAM"
    )

    plt.axis("off")


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        overlay
    )

    plt.title(
        f"Predicted: "
        f"{'CRC' if predicted_class == 0 else 'Non-CRC'}\n"
        f"Confidence: {confidence:.4f}"
    )

    plt.axis("off")


    plt.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    return (
        predicted_class,
        confidence
    )


# ============================================================
# SELECT TEST IMAGES
# ============================================================

crc_samples = []

non_crc_samples = []


for sample in test_samples:

    image_path, label = sample

    if label == 0:

        if len(crc_samples) < NUM_IMAGES_PER_CLASS:

            crc_samples.append(
                sample
            )

    elif label == 1:

        if len(non_crc_samples) < NUM_IMAGES_PER_CLASS:

            non_crc_samples.append(
                sample
            )

    if (
        len(crc_samples)
        == NUM_IMAGES_PER_CLASS
        and
        len(non_crc_samples)
        == NUM_IMAGES_PER_CLASS
    ):

        break


selected_samples = (
    crc_samples
    + non_crc_samples
)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

print("\nGenerating Grad-CAM images...")

for index, (
    image_path,
    actual_label
) in enumerate(
    selected_samples,
    start=1
):

    output_path = (
        OUTPUT_DIR
        / f"ghostnet_gradcam_{index}.png"
    )


    predicted_class, confidence = (
        generate_gradcam(
            image_path,
            actual_label,
            output_path
        )
    )


    actual_name = (
        "CRC"
        if actual_label == 0
        else "Non-CRC"
    )

    predicted_name = (
        "CRC"
        if predicted_class == 0
        else "Non-CRC"
    )


    print(
        f"{index}. "
        f"Actual={actual_name}, "
        f"Predicted={predicted_name}, "
        f"Confidence={confidence:.4f}"
    )

    print(
        "   Saved:",
        output_path
    )


# ============================================================
# REMOVE HOOKS
# ============================================================

forward_handle.remove()
backward_handle.remove()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)

print(
    "GHOSTNET GRAD-CAM COMPLETE"
)

print("=" * 60)

print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)