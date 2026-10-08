import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.models import densenet121


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path("best_densenet121_crc.pth")

DATA_DIR = Path("data/colon_image_sets")

OUTPUT_DIR = Path(
    "results/densenet121_xai/gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_IMAGES_PER_CLASS = 3


# ============================================================
# IMAGE TRANSFORM
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
# LOAD MODEL
# ============================================================

model = densenet121(
    weights=None
)

model.classifier = torch.nn.Linear(
    model.classifier.in_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(
        checkpoint
    )

model = model.to(DEVICE)
model.eval()


print("=" * 70)
print("DENSENET121 GRAD-CAM")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# GRAD-CAM STORAGE
# ============================================================

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# DenseNet121 final convolutional feature layer
target_layer = model.features.denseblock4

forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(image_tensor, target_class):

    global activations
    global gradients

    activations = None
    gradients = None

    image_tensor = image_tensor.unsqueeze(0).to(
        DEVICE
    )

    model.zero_grad()

    output = model(image_tensor)

    score = output[0, target_class]

    score.backward()

    # --------------------------------------------------------
    # DenseNet feature map
    # --------------------------------------------------------

    feature_maps = activations[0]

    grads = gradients[0]

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    weights = grads.mean(
        dim=(1, 2)
    )

    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    cam = torch.zeros(
        feature_maps.shape[1:],
        device=DEVICE
    )

    for i, weight in enumerate(weights):
        cam += weight * feature_maps[i]

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = F.relu(cam)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    cam -= cam.min()

    if cam.max() > 0:
        cam /= cam.max()

    cam = cam.detach().cpu().numpy()

    # --------------------------------------------------------
    # Resize to 224 x 224
    # --------------------------------------------------------

    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    )

    cam_image = cam_image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    cam = np.asarray(
        cam_image
    ) / 255.0

    return cam, output


# ============================================================
# IMAGE PROCESSING
# ============================================================

classes = {
    0: "CRC",
    1: "Non-CRC"
}


for class_index, class_name in classes.items():

    class_dir = (
        DATA_DIR /
        (
            "colon_aca"
            if class_index == 0
            else "colon_n"
        )
    )

    image_paths = sorted(
        class_dir.glob("*.jpeg")
    )

    # Use the first few test-like examples
    image_paths = image_paths[
        :NUM_IMAGES_PER_CLASS
    ]

    print()
    print(
        f"Processing {class_name} images..."
    )

    for image_number, image_path in enumerate(
        image_paths,
        start=1
    ):

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        original_image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = transform(
            original_image
        )

        # ----------------------------------------------------
        # First obtain prediction
        # ----------------------------------------------------

        with torch.no_grad():

            input_tensor = image_tensor.unsqueeze(
                0
            ).to(DEVICE)

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

        # ----------------------------------------------------
        # Generate Grad-CAM for predicted class
        # ----------------------------------------------------

        cam, _ = generate_gradcam(
            image_tensor,
            predicted_class
        )

        # ----------------------------------------------------
        # Prepare original image
        # ----------------------------------------------------

        display_image = original_image.resize(
            (224, 224)
        )

        display_image = np.asarray(
            display_image
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        # Original
        axes[0].imshow(
            display_image
        )

        axes[0].set_title(
            "Original Image"
        )

        axes[0].axis("off")

        # Heatmap
        axes[1].imshow(
            cam,
            cmap="jet"
        )

        axes[1].set_title(
            "Grad-CAM"
        )

        axes[1].axis("off")

        # Overlay
        axes[2].imshow(
            display_image
        )

        axes[2].imshow(
            cam,
            cmap="jet",
            alpha=0.45
        )

        axes[2].set_title(
            f"Prediction: {classes[predicted_class]}\n"
            f"Confidence: {confidence:.2%}"
        )

        axes[2].axis("off")

        plt.tight_layout()

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR /
            f"{class_name.lower()}_"
            f"{image_number}_gradcam.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


# ============================================================
# REMOVE HOOKS
# ============================================================

forward_handle.remove()
backward_handle.remove()


print()
print("=" * 70)
print("DENSENET121 GRAD-CAM COMPLETE")
print("=" * 70)

print(
    "Results saved to:",
    OUTPUT_DIR
)