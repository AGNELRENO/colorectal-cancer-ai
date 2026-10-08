import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    "results/vgg16_xai/gradcam"
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
# LOAD VGG16 MODEL
# ============================================================

print("===== LOADING VGG16 =====")

# Same VGG16 architecture used during training
model = models.vgg16(
    weights=None
)

# ------------------------------------------------------------
# IMPORTANT:
# Disable in-place ReLU operations.
#
# Grad-CAM needs gradients from the convolutional layers.
# VGG16 normally uses inplace=True ReLU operations, which can
# cause an autograd error when backward hooks are used.
# ------------------------------------------------------------

for layer in model.features:

    if isinstance(layer, nn.ReLU):

        layer.inplace = False


# Replace final classifier layer with 2 classes
model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    2
)


# ============================================================
# LOAD TRAINED CHECKPOINT
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
# GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # ----------------------------------------------------
        # Forward hook
        # ----------------------------------------------------

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        # ----------------------------------------------------
        # Backward hook
        # ----------------------------------------------------

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    # ========================================================
    # SAVE ACTIVATION
    # ========================================================

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    # ========================================================
    # SAVE GRADIENT
    # ========================================================

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    # ========================================================
    # GENERATE GRAD-CAM
    # ========================================================

    def generate(
        self,
        image_tensor,
        class_index
    ):

        # Clear previous gradients
        self.model.zero_grad()

        # Forward pass
        output = self.model(
            image_tensor
        )

        # Score corresponding to predicted class
        score = output[
            0,
            class_index
        ]

        # Backward pass
        score.backward()

        # Retrieve activations and gradients
        activations = self.activations
        gradients = self.gradients

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted combination of activation maps
        # ----------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # Keep only positive contributions
        cam = F.relu(
            cam
        )

        # ----------------------------------------------------
        # Resize CAM to original input size
        # ----------------------------------------------------

        cam = F.interpolate(
            cam,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        # ----------------------------------------------------
        # Normalize CAM between 0 and 1
        # ----------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        if (
            cam_max - cam_min
        ) > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = torch.zeros_like(
                cam
            )

        return (
            output,
            cam.detach().cpu().numpy()
        )

    # ========================================================
    # REMOVE HOOKS
    # ========================================================

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# TARGET LAYER
# ============================================================

# VGG16 final convolutional layer
#
# VGG16 feature index 28 = final Conv2d layer
#
target_layer = model.features[28]

gradcam = GradCAM(
    model,
    target_layer
)


# ============================================================
# SELECT TEST IMAGES
# ============================================================

print()
print(
    "===== SELECTING TEST IMAGES ====="
)

# CRC = class 0
crc_samples = [
    sample
    for sample in test_samples
    if sample[1] == 0
]

# Non-CRC = class 1
non_crc_samples = [
    sample
    for sample in test_samples
    if sample[1] == 1
]


# Select 3 images from each class
selected_samples = (
    crc_samples[:3]
    +
    non_crc_samples[:3]
)


print(
    "CRC images selected:",
    3
)

print(
    "Non-CRC images selected:",
    3
)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

print()
print(
    "===== GENERATING GRAD-CAM ====="
)


for index, (
    image_path,
    true_label
) in enumerate(
    selected_samples,
    start=1
):

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = Image.open(
        image_path
    ).convert("RGB")


    # ========================================================
    # PREPARE INPUT
    # ========================================================

    input_tensor = transform(
        image
    ).unsqueeze(0).to(
        DEVICE
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = (
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

        confidence = (
            probabilities[
                0,
                predicted_class
            ].item()
        )


    # ========================================================
    # GENERATE GRAD-CAM
    # ========================================================

    _, cam = gradcam.generate(
        input_tensor,
        predicted_class
    )


    # ========================================================
    # RESIZE ORIGINAL IMAGE
    # ========================================================

    display_image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    axes[0].imshow(
        display_image
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")


    # ========================================================
    # GRAD-CAM HEATMAP
    # ========================================================

    axes[1].imshow(
        cam,
        cmap="jet"
    )

    axes[1].set_title(
        "Grad-CAM"
    )

    axes[1].axis("off")


    # ========================================================
    # GRAD-CAM OVERLAY
    # ========================================================

    axes[2].imshow(
        display_image
    )

    axes[2].imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        "Grad-CAM Overlay"
    )

    axes[2].axis("off")


    # ========================================================
    # FIGURE TITLE
    # ========================================================

    fig.suptitle(
        f"True: {CLASS_NAMES[true_label]} | "
        f"Predicted: {CLASS_NAMES[predicted_class]} | "
        f"Confidence: {confidence:.2%}"
    )


    plt.tight_layout()


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    output_file = (
        OUTPUT_DIR
        / f"vgg16_gradcam_{index}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print(
        f"{index}. "
        f"True={CLASS_NAMES[true_label]}, "
        f"Predicted={CLASS_NAMES[predicted_class]}, "
        f"Confidence={confidence:.4f}"
    )

    print(
        "   Saved:",
        output_file
    )


# ============================================================
# CLEANUP
# ============================================================

gradcam.close()


print()
print(
    "===== VGG16 GRAD-CAM COMPLETE ====="
)

print(
    "Results saved to:",
    OUTPUT_DIR
)