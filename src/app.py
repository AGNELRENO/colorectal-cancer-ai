import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT RMSCNN
# ============================================================

from model_rmscnn import RMSCNN


# ============================================================
# MODEL PATH
# ============================================================

RMSCNN_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_rmscnn_crc.pth"
)


# ============================================================
# MEDGEMMA PATHS
# ============================================================

MEDGEMMA_MODEL = Path(
    r"E:\medgemma\medgemma-1.5-4b-it-q4_0.gguf"
)

MEDGEMMA_MMPROJ = Path(
    r"E:\medgemma\mmproj-medgemma-1.5-4b-it-q4_0.gguf"
)


# ============================================================
# RESULTS PATHS
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "results"

RMSCNN_RESULTS = RESULTS_DIR / "rmscnn"
RMSCNN_XAI = RESULTS_DIR / "rmscnn_xai"

STREAMLIT_TEMP_DIR = RESULTS_DIR / "streamlit_temp"
STREAMLIT_TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CRC Histopathology AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 750;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.72;
        margin-top: 4px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 700;
        margin-top: 35px;
        margin-bottom: 15px;
    }

    .section-subtitle {
        font-size: 15px;
        opacity: 0.70;
        margin-bottom: 15px;
    }

    .result-card {
        border: 1px solid rgba(128,128,128,0.30);
        border-radius: 14px;
        padding: 22px;
        min-height: 145px;
        background: rgba(128,128,128,0.04);
    }

    .result-label {
        font-size: 15px;
        opacity: 0.70;
        margin-bottom: 10px;
    }

    .result-value {
        font-size: 27px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-sub {
        font-size: 13px;
        opacity: 0.65;
    }

    .model-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .notice {
        border-left: 4px solid #6c63ff;
        padding: 12px 16px;
        border-radius: 5px;
        background: rgba(108,99,255,0.08);
        margin: 12px 0;
    }

    .warning-box {
        border-left: 4px solid #f0ad4e;
        padding: 12px 16px;
        border-radius: 5px;
        background: rgba(240,173,78,0.08);
        margin: 12px 0;
    }

    .image-caption {
        text-align: center;
        font-size: 13px;
        opacity: 0.70;
        margin-top: 5px;
    }

    [data-testid="column"] {
        padding-left: 7px;
        padding-right: 7px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔬 CRC Histopathology AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explainable Multi-Stage Computer Vision Framework for '
    'Colorectal Cancer Histopathology Analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🔬 Project")

    st.markdown(
        """
        **Domain:**  
        Computer Vision / Medical Imaging

        **Task:**  
        CRC vs Non-CRC Classification

        **Dataset:**  
        LC25000 Colon Histopathology

        **Primary Model:**  
        RMSCNN

        **Explainability:**  
        Grad-CAM

        **VLM:**  
        MedGemma 1.5 4B
        """
    )

    st.divider()

    st.markdown("### Navigation")

    section_choice = st.radio(
        "Go to section",
        [
            "Overview",
            "Image Analysis",
            "Model Comparison",
            "Model Visualizations",
            "RMSCNN Training",
            "Research Report",
        ]
    )

    st.divider()

    st.caption(
        "Research prototype only. "
        "Model predictions are not clinical diagnoses."
    )


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "CRC / Colon Adenocarcinoma",
    1: "Non-CRC / Benign Colon Tissue",
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# LOAD RMSCNN
# ============================================================

@st.cache_resource
def load_rmscnn():

    model = RMSCNN(
        num_classes=2
    )

    checkpoint = torch.load(
        RMSCNN_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# SAFE IMAGE DISPLAY
# ============================================================

def display_image(
    image,
    caption=None,
    width=420,
):

    if image is None:
        return

    st.image(
        image,
        caption=caption,
        width=width,
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    model,
    image,
):

    image_rgb = image.convert("RGB")

    tensor = IMAGE_TRANSFORM(
        image_rgb
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = int(
        predicted.item()
    )

    confidence_value = float(
        confidence.item()
    )

    return (
        predicted_class,
        confidence_value,
        probabilities[0].detach().cpu().numpy(),
        tensor,
    )


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.backward_hook
            )
        )

    def forward_hook(
        self,
        module,
        input,
        output,
    ):

        self.activations = output.detach()

    def backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = (
            grad_output[0].detach()
        )

    def generate(
        self,
        image_tensor,
        target_class,
    ):

        self.model.zero_grad()

        output = self.model(
            image_tensor
        )

        score = output[
            0,
            target_class
        ]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1
        )

        cam = F.relu(cam)

        cam = cam[
            0
        ].cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# CREATE GRAD-CAM
# ============================================================

def create_gradcam(
    model,
    image,
    image_tensor,
    target_class,
):

    # Actual RMSCNN architecture:
    # layer3 is the final multi-scale residual block.
    target_layer = model.layer3

    gradcam = GradCAM(
        model,
        target_layer
    )

    heatmap = gradcam.generate(
        image_tensor,
        target_class
    )

    gradcam.close()

    original = np.array(
        image.convert("RGB")
    )

    heatmap_resized = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0],
        )
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.60,
        heatmap_color,
        0.40,
        0,
    )

    heatmap_image = Image.fromarray(
        heatmap_color
    )

    overlay_image = Image.fromarray(
        overlay
    )

    return (
        heatmap_image,
        overlay_image,
        heatmap_resized,
    )


# ============================================================
# SAVE IMAGE FOR MEDGEMMA
# ============================================================

def save_temp_image(
    image,
    filename="streamlit_gradcam.png",
):

    output_path = (
        STREAMLIT_TEMP_DIR
        / filename
    )

    image.save(
        output_path
    )

    return output_path


# ============================================================
# MEDGEMMA ANALYSIS
# ============================================================

def run_medgemma(
    image_path,
    true_label,
    predicted_label,
    confidence,
):

    if not MEDGEMMA_MODEL.exists():
        return (
            "MedGemma model was not found at:\n"
            f"{MEDGEMMA_MODEL}"
        )

    if not MEDGEMMA_MMPROJ.exists():
        return (
            "MedGemma mmproj file was not found at:\n"
            f"{MEDGEMMA_MMPROJ}"
        )

    prompt = f"""
Analyze the provided RMSCNN Grad-CAM visualization for a research
project on colorectal cancer histopathology image classification.

RMSCNN classification mapping:
0 = CRC
1 = Non-CRC

True label:
{true_label}

Predicted label:
{predicted_label}

Model confidence:
{confidence:.2%}

Give ONLY these four sections:

1. VISUAL OBSERVATIONS

Describe only directly observable tissue, cellular, color,
texture, and architectural patterns.

2. GRAD-CAM INTERPRETATION

Describe where the heatmap is concentrated and what visual
patterns the model may be attending to.

3. RELATION TO MODEL PREDICTION

Explain how the highlighted visual patterns may have contributed
to the RMSCNN prediction.

Do not imply that these patterns prove cancer.

4. LIMITATION

State that Grad-CAM represents model attention and does not
provide confirmed tumor boundaries, segmentation, or a clinical
diagnosis.

Do not claim that a highlighted region is definitely a tumor,
cancer, nucleus, gland, necrosis, or any other specific
pathological structure.

Do not provide a clinical diagnosis.

Do not describe hidden reasoning.
"""

    command = [
        "llama-mtmd-cli",

        "-m",
        str(MEDGEMMA_MODEL),

        "--mmproj",
        str(MEDGEMMA_MMPROJ),

        "--image",
        str(image_path),

        "-ngl",
        "20",

        "-c",
        "4096",

        "-n",
        "350",

        "-p",
        prompt,
    ]

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )

        if result.returncode != 0:

            return (
                "MedGemma execution failed.\n\n"
                + result.stderr
            )

        return result.stdout.strip()

    except subprocess.TimeoutExpired:

        return (
            "MedGemma analysis timed out."
        )

    except FileNotFoundError:

        return (
            "llama-mtmd-cli was not found. "
            "Make sure llama.cpp is installed "
            "and available in PATH."
        )

    except Exception as e:

        return (
            f"MedGemma error: {str(e)}"
        )


# ============================================================
# MODEL COMPARISON DATA
# ============================================================

MODEL_RESULTS = pd.DataFrame(
    {
        "Model": [
            "Basic CNN",
            "VGG16",
            "ResNet18",
            "DenseNet121",
            "GhostNet",
            "RMSCNN",
        ],

        "Accuracy": [
            0.9409,
            0.9660,
            0.9872,
            0.9949,
            0.9994,
            0.9968,
        ],

        "Precision": [
            0.9589,
            0.9854,
            0.9873,
            0.9949,
            1.0000,
            0.9936,
        ],

        "Recall": [
            0.9223,
            0.9465,
            0.9872,
            0.9987,
            0.9987,
            1.0000,
        ],

        "F1": [
            0.9403,
            0.9656,
            0.9872,
            0.9949,
            0.9994,
            0.9968,
        ],

        "ROC-AUC": [
            0.9781,
            0.9930,
            0.9968,
            1.0000,
            1.0000,
            1.0000,
        ],
    }
)


# ============================================================
# MODEL SUMMARY PATHS
# ============================================================

MODEL_SUMMARIES = {
    "Basic CNN":
        RESULTS_DIR / "model_summaries" / "basic_cnn_summary.txt",

    "VGG16":
        RESULTS_DIR / "model_summaries" / "vgg16_summary.txt",

    "ResNet18":
        RESULTS_DIR / "model_summaries" / "resnet18_summary.txt",

    "DenseNet121":
        RESULTS_DIR / "model_summaries" / "densenet121_summary.txt",

    "GhostNet":
        RESULTS_DIR / "model_summaries" / "ghostnet_summary.txt",

    "RMSCNN":
        RESULTS_DIR / "model_summaries" / "rmscnn_summary.txt",
}


# ============================================================
# MODEL VISUALIZATION PATHS
# ============================================================

MODEL_VISUALIZATIONS = {

    "Basic CNN": [
        RESULTS_DIR / "basic_cnn" / "basic_cnn_accuracy_curve.png",
        RESULTS_DIR / "basic_cnn" / "basic_cnn_loss_curve.png",
        RESULTS_DIR / "basic_cnn" / "basic_cnn_anova_top30.png",
    ],

    "VGG16": [
        RESULTS_DIR / "vgg16" / "vgg16_confusion_matrix.png",
        RESULTS_DIR / "vgg16" / "vgg16_roc_curve.png",
        RESULTS_DIR / "vgg16" / "vgg16_misclassified_grid.png",
    ],

    "ResNet18": [
        RESULTS_DIR / "resnet18_fusion_svm" / "confusion_matrix.png",
        RESULTS_DIR / "resnet18_fusion_svm" / "roc_curve.png",
        RESULTS_DIR / "resnet18_xai" / "top30_anova_features.png",
    ],

    "DenseNet121": [
        RESULTS_DIR / "densenet121_fusion_svm" / "densenet121_confusion_matrix.png",
        RESULTS_DIR / "densenet121_fusion_svm" / "densenet121_roc_curve.png",
    ],

    "GhostNet": [
        RESULTS_DIR / "ghostnet_fusion_svm" / "ghostnet_confusion_matrix.png",
        RESULTS_DIR / "ghostnet_fusion_svm" / "ghostnet_roc_curve.png",
        RESULTS_DIR / "ghostnet_fusion_svm" / "ghostnet_misclassified_grid.png",
    ],

    "RMSCNN": [
        RESULTS_DIR / "rmscnn" / "rmscnn_accuracy_curve.png",
        RESULTS_DIR / "rmscnn" / "rmscnn_loss_curve.png",
    ],
}


# ============================================================
# HELPER: FIND EXISTING FILE
# ============================================================

def find_existing_file(
    candidates,
):

    for path in candidates:

        if path.exists():
            return path

    return None


# ============================================================
# OVERVIEW
# ============================================================

if section_choice == "Overview":

    st.markdown(
        '<div class="section-title">'
        'Project Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="notice">

        This research prototype performs binary classification of
        colorectal histopathology images into CRC and Non-CRC
        categories.

        The workflow combines deep CNN features, handcrafted
        image features, feature selection, RBF-SVM classification,
        explainable AI using Grad-CAM, and research-oriented
        MedGemma interpretation.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PROJECT CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            """
            <div class="result-card">
                <div class="result-label">Dataset</div>
                <div class="result-value">LC25000</div>
                <div class="result-sub">
                    10,000 colon images used
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="result-card">
                <div class="result-label">Models</div>
                <div class="result-value">6</div>
                <div class="result-sub">
                    CNN architectures evaluated
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="result-card">
                <div class="result-label">Primary Model</div>
                <div class="result-value">RMSCNN</div>
                <div class="result-sub">
                    Selected for XAI and VLM
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            """
            <div class="result-card">
                <div class="result-label">RMSCNN Fusion Accuracy</div>
                <div class="result-value">99.68%</div>
                <div class="result-sub">
                    RBF-SVM with fused features
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Research Workflow</div>',
        unsafe_allow_html=True
    )

    workflow = """
    Dataset
    →
    Preprocessing
    →
    Group-Aware Split
    →
    Six CNN Models
    →
    Deep Feature Extraction
    +
    44 Handcrafted Features
    →
    Feature Fusion
    →
    SelectKBest / ANOVA
    →
    StandardScaler
    →
    RBF-SVM
    →
    Model Comparison
    →
    RMSCNN Selection
    →
    Grad-CAM
    →
    MedGemma
    →
    Streamlit Research Prototype
    """

    st.code(
        workflow,
        language="text"
    )

    st.markdown(
        '<div class="section-title">'
        'Six-Model Experimental Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    comparison_display = MODEL_RESULTS.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]:

        comparison_display[column] = (
            comparison_display[column] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

elif section_choice == "Image Analysis":

    st.markdown(
        '<div class="section-title">'
        '🔬 Histopathology Image Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Upload a colorectal histopathology image for RMSCNN analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload histopathology image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        model = load_rmscnn()

        (
            predicted_class,
            confidence,
            probabilities,
            image_tensor,
        ) = predict_image(
            model,
            image
        )

        prediction_name = CLASS_NAMES[
            predicted_class
        ]

        # ----------------------------------------------------
        # RESULT CARDS
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Prediction
                    </div>

                    <div class="result-value">
                        {prediction_name}
                    </div>

                    <div class="result-sub">
                        RMSCNN binary classification result
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Confidence
                    </div>

                    <div class="result-value">
                        {confidence:.2%}
                    </div>

                    <div class="result-sub">
                        Model probability — not clinical certainty
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            dataset_label = (
                "colon_aca"
                if predicted_class == 0
                else "colon_n"
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Class
                    </div>

                    <div class="result-value">
                        {dataset_label}
                    </div>

                    <div class="result-sub">
                        Dataset class mapping
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="section-title">'
            'Uploaded Image'
            '</div>',
            unsafe_allow_html=True
        )

        # Limit displayed size
        display_image(
            image,
            caption="Uploaded histopathology image",
            width=520,
        )

        st.markdown(
            '<div class="section-title">'
            'Visual Explanation'
            '</div>',
            unsafe_allow_html=True
        )

        (
            heatmap_image,
            overlay_image,
            heatmap_array,
        ) = create_gradcam(
            model,
            image,
            image_tensor,
            predicted_class,
        )

        gc1, gc2, gc3 = st.columns(3)

        with gc1:

            st.markdown(
                "#### Original image"
            )

            display_image(
                image,
                width=360,
            )

        with gc2:

            st.markdown(
                "#### Grad-CAM heatmap"
            )

            display_image(
                heatmap_image,
                width=360,
            )

        with gc3:

            st.markdown(
                "#### Grad-CAM overlay"
            )

            display_image(
                overlay_image,
                width=360,
            )

        st.markdown(
            """
            <div class="warning-box">

            Grad-CAM indicates image regions that contributed to
            the model prediction. It is an explainability method,
            not a tumor segmentation method and not a clinical
            diagnostic tool.

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # PROBABILITY DISTRIBUTION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Prediction Probability'
            '</div>',
            unsafe_allow_html=True
        )

        probability_df = pd.DataFrame(
            {
                "Class": [
                    "CRC",
                    "Non-CRC",
                ],

                "Probability": [
                    float(probabilities[0]),
                    float(probabilities[1]),
                ],
            }
        )

        st.bar_chart(
            probability_df.set_index("Class")
        )

        # ----------------------------------------------------
        # MEDGEMMA
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🤖 MedGemma Research Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "MedGemma receives the Grad-CAM visualization and "
            "provides a research-oriented natural-language "
            "interpretation."
        )

        if st.button(
            "Generate MedGemma Analysis",
            type="primary",
        ):

            overlay_path = save_temp_image(
                overlay_image
            )

            with st.spinner(
                "Running MedGemma analysis..."
            ):

                medgemma_output = run_medgemma(
                    overlay_path,
                    "Unknown",
                    prediction_name,
                    confidence,
                )

            st.session_state[
                "medgemma_output"
            ] = medgemma_output

        if (
            "medgemma_output"
            in st.session_state
        ):

            st.markdown(
                "### Research-Oriented Interpretation"
            )

            st.markdown(
                st.session_state[
                    "medgemma_output"
                ]
            )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif section_choice == "Model Comparison":

    st.markdown(
        '<div class="section-title">'
        '📊 Six-Model Performance Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        MODEL_RESULTS.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1": "{:.2%}",
                "ROC-AUC": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # ACCURACY CHART
    # --------------------------------------------------------

    st.markdown(
        "### Test Accuracy"
    )

    accuracy_chart = (
        MODEL_RESULTS
        .set_index("Model")[
            ["Accuracy"]
        ]
        .copy()
    )

    st.bar_chart(
        accuracy_chart
    )

    # --------------------------------------------------------
    # ALL METRICS
    # --------------------------------------------------------

    st.markdown(
        "### Complete Performance Comparison"
    )

    metric_chart = (
        MODEL_RESULTS
        .set_index("Model")[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
            ]
        ]
    )

    st.bar_chart(
        metric_chart
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_model = MODEL_RESULTS.loc[
        MODEL_RESULTS["Accuracy"].idxmax()
    ]

    st.markdown(
        f"""
        <div class="notice">

        <b>Selected primary model among the final comparison:</b>
        RMSCNN achieved <b>{best_model["Accuracy"]:.2%}</b>
        fusion-SVM test accuracy.

        RMSCNN was therefore selected for subsequent
        explainability and MedGemma research analysis.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL VISUALIZATIONS
# ============================================================

elif section_choice == "Model Visualizations":

    st.markdown(
        '<div class="section-title">'
        '📈 Model Summaries & Visualizations'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "All six trained models are included below."
    )

    for model_name in [
        "Basic CNN",
        "VGG16",
        "ResNet18",
        "DenseNet121",
        "GhostNet",
        "RMSCNN",
    ]:

        st.markdown(
            f"## {model_name}"
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary_path = MODEL_SUMMARIES[
            model_name
        ]

        if summary_path.exists():

            with st.expander(
                f"View {model_name} model summary"
            ):

                try:

                    summary_text = (
                        summary_path.read_text(
                            encoding="utf-8",
                            errors="replace",
                        )
                    )

                    st.code(
                        summary_text,
                        language="text"
                    )

                except Exception as e:

                    st.error(
                        f"Could not read summary: {e}"
                    )

        else:

            st.info(
                f"Summary file not found:\n"
                f"{summary_path}"
            )

        # ----------------------------------------------------
        # VISUALIZATIONS
        # ----------------------------------------------------

        existing_images = []

        for image_path in MODEL_VISUALIZATIONS[
            model_name
        ]:

            if image_path.exists():
                existing_images.append(
                    image_path
                )

        if not existing_images:

            st.info(
                "No saved visualizations found "
                "for this model."
            )

        else:

            cols = st.columns(3)

            for index, image_path in enumerate(
                existing_images
            ):

                with cols[
                    index % 3
                ]:

                    st.image(
                        str(image_path),
                        caption=image_path.name,
                        width=360,
                    )

        st.divider()


# ============================================================
# RMSCNN TRAINING
# ============================================================

elif section_choice == "RMSCNN Training":

    st.markdown(
        '<div class="section-title">'
        '📈 RMSCNN Training Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    accuracy_path = (
        RMSCNN_RESULTS
        / "rmscnn_accuracy_curve.png"
    )

    loss_path = (
        RMSCNN_RESULTS
        / "rmscnn_loss_curve.png"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            "### Training & Validation Accuracy"
        )

        if accuracy_path.exists():

            st.image(
                str(accuracy_path),
                width=650,
            )

        else:

            st.warning(
                f"Not found:\n{accuracy_path}"
            )

    with c2:

        st.markdown(
            "### Training & Validation Loss"
        )

        if loss_path.exists():

            st.image(
                str(loss_path),
                width=650,
            )

        else:

            st.warning(
                f"Not found:\n{loss_path}"
            )

    st.markdown(
        '<div class="section-title">'
        'RMSCNN Training Summary'
        '</div>',
        unsafe_allow_html=True
    )

    training_summary = (
        RMSCNN_RESULTS
        / "rmscnn_training_summary.txt"
    )

    if training_summary.exists():

        st.code(
            training_summary.read_text(
                encoding="utf-8",
                errors="replace",
            ),
            language="text",
        )

    else:

        st.info(
            "RMSCNN training summary file "
            "was not found."
        )


# ============================================================
# RESEARCH REPORT
# ============================================================

elif section_choice == "Research Report":

    st.markdown(
        '<div class="section-title">'
        '📋 AI-Assisted Research Report'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="warning-box">

        <b>Research-use notice:</b>

        This report is an AI-assisted research output based on
        the RMSCNN classification result and Grad-CAM visualization.
        It must not be interpreted as a clinical diagnosis.

        </div>
        """,
        unsafe_allow_html=True
    )

    if (
        "medgemma_output"
        in st.session_state
    ):

        st.markdown(
            "### MedGemma Interpretation"
        )

        st.markdown(
            st.session_state[
                "medgemma_output"
            ]
        )

        report_text = (
            "CRC HISTOPATHOLOGY AI\n"
            "AI-ASSISTED RESEARCH REPORT\n"
            "\n"
            "====================================\n\n"
            "MEDGEMMA INTERPRETATION\n\n"
            + st.session_state[
                "medgemma_output"
            ]
            + "\n\n"
            "====================================\n\n"
            "LIMITATION\n\n"
            "This output is intended only for research "
            "and demonstration purposes. Grad-CAM represents "
            "model attention and does not provide confirmed "
            "tumor boundaries, segmentation, or a clinical "
            "diagnosis.\n"
        )

        st.download_button(
            label="Download Research Report",
            data=report_text,
            file_name="crc_ai_assisted_research_report.txt",
            mime="text/plain",
        )

    else:

        st.info(
            "Run MedGemma analysis from the "
            "'Image Analysis' section first."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CRC Histopathology AI • RMSCNN • Grad-CAM • MedGemma 1.5 4B • "
    "Research Prototype"
)