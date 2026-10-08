import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

from model import create_model
from dataset import val_test_transform
from split_dataset import test_samples


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = Path("best_resnet18_crc.pth")
OUTPUT_DIR = Path("gradcam_results")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {
    0: "CRC",
    1: "Non-CRC"
}

# Number of examples for each class
NUM_IMAGES_PER_CLASS = 3


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD MODEL
# ============================================================

model = create_model(num_classes=2)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("ResNet18 model loaded successfully.")


# ============================================================
# TARGET LAYER
# ============================================================

target_layer = model.layer4[-1]


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(image_path, output_path):

    activations = []
    gradients = []

    # --------------------------------------------------------
    # Hooks
    # --------------------------------------------------------

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = val_test_transform(
        original_image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(input_tensor)

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
    # Backward pass
    # --------------------------------------------------------

    model.zero_grad()

    target_score = output[
        0,
        predicted_class
    ]

    target_score.backward()

    # --------------------------------------------------------
    # Get activations and gradients
    # --------------------------------------------------------

    activation = activations[0][0]
    gradient = gradients[0][0]

    # --------------------------------------------------------
    # Calculate Grad-CAM weights
    # --------------------------------------------------------

    weights = gradient.mean(
        dim=(1, 2),
        keepdim=True
    )

    cam = (
        weights * activation
    ).sum(dim=0)

    cam = torch.relu(cam)

    cam = cam.detach().cpu().numpy()

    # --------------------------------------------------------
    # Normalize CAM
    # --------------------------------------------------------

    cam = cam - cam.min()

    if cam.max() != 0:
        cam = cam / cam.max()

    # --------------------------------------------------------
    # Resize CAM
    # --------------------------------------------------------

    original_width, original_height = (
        original_image.size
    )

    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    )

    cam_image = cam_image.resize(
        (original_width, original_height)
    )

    cam = np.array(
        cam_image
    ) / 255.0

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.imshow(original_image)

    plt.title(
        f"Original\n"
        f"Prediction: {CLASS_NAMES[predicted_class]}"
    )

    plt.axis("off")

    plt.subplot(1, 2, 2)

    plt.imshow(original_image)

    plt.imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    plt.title(
        f"Grad-CAM\n"
        f"{CLASS_NAMES[predicted_class]} "
        f"({confidence:.2%})"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    forward_handle.remove()
    backward_handle.remove()

    return predicted_class, confidence


# ============================================================
# SEPARATE TEST IMAGES BY ACTUAL CLASS
# ============================================================

crc_samples = [
    (path, label)
    for path, label in test_samples
    if label == 0
]

non_crc_samples = [
    (path, label)
    for path, label in test_samples
    if label == 1
]


# ============================================================
# SELECT REPRESENTATIVE IMAGES
# ============================================================

selected_samples = []

selected_samples.extend(
    crc_samples[:NUM_IMAGES_PER_CLASS]
)

selected_samples.extend(
    non_crc_samples[:NUM_IMAGES_PER_CLASS]
)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

print()
print("Generating Grad-CAM visualizations...")
print()

results = []

for index, (image_path, true_label) in enumerate(
    selected_samples,
    start=1
):

    output_path = (
        OUTPUT_DIR /
        f"gradcam_{index}_{CLASS_NAMES[true_label]}.png"
    )

    predicted_class, confidence = generate_gradcam(
        image_path,
        output_path
    )

    results.append(
        (
            image_path.name,
            CLASS_NAMES[true_label],
            CLASS_NAMES[predicted_class],
            confidence,
            output_path
        )
    )

    print(
        f"[{index}/{len(selected_samples)}] "
        f"{image_path.name}"
    )

    print(
        f"    Actual     : "
        f"{CLASS_NAMES[true_label]}"
    )

    print(
        f"    Prediction : "
        f"{CLASS_NAMES[predicted_class]}"
    )

    print(
        f"    Confidence : "
        f"{confidence:.4f}"
    )

    print(
        f"    Saved      : "
        f"{output_path}"
    )

    print()


# ============================================================
# SUMMARY
# ============================================================

print("========================================")
print("       GRAD-CAM COMPLETE")
print("========================================")

print(
    f"Generated images: {len(results)}"
)

print(
    f"Output folder: {OUTPUT_DIR}"
)

print()

for result in results:

    image_name, actual, prediction, confidence, path = result

    print(
        f"{image_name} | "
        f"Actual: {actual} | "
        f"Predicted: {prediction} | "
        f"Confidence: {confidence:.4f}"
    )