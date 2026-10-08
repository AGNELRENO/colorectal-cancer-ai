import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image

from dataset import val_test_transform
from model_rmscnn import get_model


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best_rmscnn_crc.pth"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "rmscnn_xai",
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

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = get_model(
    num_classes=2
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("\nRMSCNN checkpoint loaded.")
print(
    "Best epoch:",
    checkpoint["epoch"]
)
print(
    "Best validation accuracy:",
    f"{checkpoint['val_accuracy']:.4f}"
)


# ============================================================
# TARGET LAYER
# ============================================================

# Last convolutional layer of RMSCNN
target_layer = model.layer3.conv5


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


forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# FIND TEST IMAGES
# ============================================================

test_samples = np.load(
    os.path.join(
        DATA_DIR,
        "test_samples.npy"
    ),
    allow_pickle=True
)


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(
    image_path,
    true_label
):

    global activations
    global gradients

    activations = None
    gradients = None

    # Load image
    original_image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = val_test_transform(
        original_image
    ).unsqueeze(0).to(device)

    # Forward pass
    output = model(
        input_tensor
    )

    probabilities = F.softmax(
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

    # Backward pass for predicted class
    model.zero_grad()

    score = output[
        0,
        predicted_class
    ]

    score.backward()

    # Get activations and gradients
    feature_maps = activations[0]
    grads = gradients[0]

    # Global average pooling of gradients
    weights = grads.mean(
        dim=(1, 2)
    )

    # Weighted feature maps
    cam = torch.zeros(
        feature_maps.shape[1:],
        device=device
    )

    for i in range(
        feature_maps.shape[0]
    ):

        cam += (
            weights[i] *
            feature_maps[i]
        )

    # ReLU
    cam = F.relu(cam)

    # Normalize
    cam -= cam.min()

    if cam.max() > 0:

        cam /= cam.max()

    # Resize to original image size
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(
            original_image.height,
            original_image.width
        ),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().detach().cpu().numpy()

    # Original image array
    image_array = np.asarray(
        original_image
    ).astype(
        np.float32
    ) / 255.0

    # ========================================================
    # SAVE VISUALIZATION
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Original
    axes[0].imshow(
        image_array
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(
        image_array
    )

    axes[1].imshow(
        cam,
        cmap="jet",
        alpha=0.5
    )

    axes[1].set_title(
        "Grad-CAM"
    )

    axes[1].axis("off")

    # Heatmap only
    axes[2].imshow(
        cam,
        cmap="jet"
    )

    axes[2].set_title(
        "Activation Map"
    )

    axes[2].axis("off")

    fig.suptitle(
        f"True: {true_label} | "
        f"Predicted: {predicted_class} | "
        f"Confidence: {confidence:.4f}",
        fontsize=12
    )

    plt.tight_layout()

    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = os.path.join(
        RESULT_DIR,
        f"{filename}_gradcam.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return (
        predicted_class,
        confidence,
        output_path
    )


# ============================================================
# SELECT CORRECT CRC / NON-CRC IMAGES
# ============================================================

selected = {
    0: [],
    1: []
}

print("\nSearching for correctly classified examples...")


with torch.no_grad():

    for sample in test_samples:

        image_path = sample[0]
        true_label = int(sample[1])

        image = Image.open(
            image_path
        ).convert("RGB")

        tensor = val_test_transform(
            image
        ).unsqueeze(0).to(device)

        output = model(
            tensor
        )

        predicted = torch.argmax(
            output,
            dim=1
        ).item()

        if (
            predicted == true_label
            and len(selected[true_label]) < 3
        ):

            selected[true_label].append(
                image_path
            )

        if (
            len(selected[0]) >= 3
            and len(selected[1]) >= 3
        ):
            break


print(
    "CRC examples selected:",
    len(selected[0])
)

print(
    "Non-CRC examples selected:",
    len(selected[1])
)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

print("\n========================================")
print("GENERATING RMSCNN GRAD-CAM")
print("========================================")


class_names = {
    0: "CRC",
    1: "Non-CRC"
}


results = []


for true_label in [0, 1]:

    print(
        f"\n{class_names[true_label]} examples:"
    )

    for image_path in selected[true_label]:

        predicted, confidence, output_path = generate_gradcam(
            image_path,
            true_label
        )

        print(
            f"Image: {os.path.basename(image_path)}"
        )

        print(
            f"True: {class_names[true_label]}"
        )

        print(
            f"Predicted: {class_names[predicted]}"
        )

        print(
            f"Confidence: {confidence:.4f}"
        )

        print(
            f"Saved: {output_path}"
        )

        results.append({
            "image": image_path,
            "true": true_label,
            "predicted": predicted,
            "confidence": confidence,
            "output": output_path
        })


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULT_DIR,
    "gradcam_summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "RMSCNN Grad-CAM Results\n"
    )

    f.write(
        "=======================\n\n"
    )

    for result in results:

        f.write(
            f"Image: "
            f"{os.path.basename(result['image'])}\n"
        )

        f.write(
            f"True Class: "
            f"{class_names[result['true']]}\n"
        )

        f.write(
            f"Predicted Class: "
            f"{class_names[result['predicted']]}\n"
        )

        f.write(
            f"Confidence: "
            f"{result['confidence']:.4f}\n"
        )

        f.write(
            f"Output: "
            f"{result['output']}\n\n"
        )


# ============================================================
# REMOVE HOOKS
# ============================================================

forward_handle.remove()
backward_handle.remove()


print("\n========================================")
print("RMSCNN GRAD-CAM COMPLETE")
print("========================================")

print(
    "Results saved to:"
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