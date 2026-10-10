"""
CRC-Lens v2 | Explainable colorectal histopathology workbench
CRC-Lens | LC25000 colon: colon_aca vs colon_n

Models: Basic CNN, VGG16, ResNet18, DenseNet121, EfficientNet-B0, GhostNet,
        Residual Multi-Scale CNN (proposed)

Run from the project root:   streamlit run app.py
"""
import ast
import hashlib
import importlib
import importlib.util
import io
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from torchvision import transforms
    TORCH_OK = True
except Exception:
    TORCH_OK = False
try:
    import timm
    TIMM_OK = True
except Exception:
    TIMM_OK = False
try:
    import reportlab  # noqa: F401
    PDF_OK = True
except Exception:
    PDF_OK = False

st.set_page_config(page_title="CRC-Lens", page_icon="🔬", layout="wide")

# ----------------------------------------------------------------------------
# Style: content is capped in width and every image is capped in size
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container { max-width: 1180px; padding-top: 1.5rem; }
[data-testid="stImage"] img { max-width: 100%; height: auto; border-radius: 12px; }
.hero { padding:1.1rem 1.4rem; border-radius:16px; border:1px solid #26304a;
        background:linear-gradient(120deg,#1a1240 0%,#0f2a3a 100%); margin-bottom:1rem; }
.hero h1 { margin:0; font-size:1.9rem;
           background:linear-gradient(90deg,#ff4d8d,#22d3c5);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { margin:.25rem 0 0 0; color:#a9b4d0; }
.card { border:1px solid #2b3350; border-radius:14px; padding:1rem 1.2rem; background:#121826; margin-bottom:.6rem; }
.card h4 { margin:.1rem 0 .4rem 0; }
.lbl { color:#9aa7c7; font-size:.78rem; letter-spacing:.06em; text-transform:uppercase; }
.pred { font-size:1.8rem; font-weight:700; line-height:1.15; }
.conf { font-size:2.2rem; font-weight:600; }
.bad  { border-left:6px solid #ff4d8d; }
.good { border-left:6px solid #22d3c5; }
.flow { text-align:left; }
.flow b { display:block; margin:.2rem 0; }
.flow span { color:#8e9ac0; display:block; margin:.1rem 0; }
.banner { background:#3a2a05; border:1px solid #8a6a10; color:#ffd36b; padding:.5rem .9rem; border-radius:10px; margin-bottom:.8rem; }
.foot { color:#8e9ac0; font-size:.8rem; margin-top:1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

DISPLAY = 300   # every image in the visual panels is shown at exactly this size (px)
LBL_N, LBL_A = "Normal Colon Tissue", "Colon Adenocarcinoma"

# ----------------------------------------------------------------------------
# Project facts (copied from the completed experiments; edit here if you retrain)
# ----------------------------------------------------------------------------
MODELS = ["Basic CNN", "VGG16", "ResNet18", "DenseNet121", "EfficientNet-B0", "GhostNet",
          "Residual Multi-Scale CNN (Proposed)"]
PROPOSED = MODELS[-1]
WEIGHT_KEYS = {
    "Basic CNN": ["basic", "simple"],
    "VGG16": ["vgg"],
    "ResNet18": ["resnet"],
    "DenseNet121": ["densenet"],
    "EfficientNet-B0": ["efficientnet", "effnet"],
    "GhostNet": ["ghost"],
    PROPOSED: ["rmscnn", "rms_cnn", "multiscale", "multi_scale", "residual_multi", "proposed"],
}
METRICS = pd.DataFrame(
    [
        ["Basic CNN", 92.98, 100.0, 86.29, 100.0, 92.64, 99.90],
        ["VGG16", 100, 100, 100, 100, 100, 100],
        ["ResNet18", 99.93, 100.0, 99.87, 100.0, 99.94, 100.0],
        ["DenseNet121", 100, 100, 100, 100, 100, 100],
        ["EfficientNet-B0", 100, 100, 100, 100, 100, 100],
        ["GhostNet", 100, 100, 100, 100, 100, 100],
        [PROPOSED, 99.9338, 99.8708, 100.0, 99.8643, 99.9354, 99.9998],
    ],
    columns=["Model", "Accuracy (%)", "Precision (%)", "Sensitivity (%)", "Specificity (%)", "F1 (%)", "ROC-AUC (%)"],
)
ARCH = pd.DataFrame(
    [
        ["Basic CNN", "Sequential convolutional blocks", "No", "No", "Yes"],
        ["VGG16", "Stacked 3×3 convolution blocks", "No", "No", "No"],
        ["ResNet18", "Residual blocks with skip connections", "Residual", "No", "Moderate"],
        ["DenseNet121", "Dense feature reuse across layers", "Dense", "No", "Moderate"],
        ["EfficientNet-B0", "MBConv-based efficient CNN", "Block-level", "Limited", "Yes"],
        ["GhostNet", "Ghost feature generation", "Block-level", "Limited", "Yes"],
        [PROPOSED, "Parallel multi-scale residual blocks", "Residual", "Yes", "Yes"],
    ],
    columns=["Model", "Core design", "Connection strategy", "Multi-scale processing", "Efficiency focus"],
)
PARAMS = {"Basic CNN": 93_730, "VGG16": 134_268_738, "ResNet18": 11_177_538, "DenseNet121": 6_955_906,
          "EfficientNet-B0": 4_010_110, "GhostNet": None, PROPOSED: 2_156_322}
COMPLEX = {  # MACs(G), FLOPs(G), inference ms, images/s — only where you measured them
    "Basic CNN": (0.5058, 1.0115, 15.476, 64.62),
    "ResNet18": (1.8136, 3.6271, 27.807, 35.96),
    PROPOSED: (3.8969, 7.7937, 62.067, 16.11),
}
FLOW = {
    PROPOSED: ("Parallel multi-scale residual feature extraction",
               ["Input 224×224×3", "Stem: Conv 3×3, 3→32", "Block 1: 3×3 + 5×5 + Dilated 3×3 → 1×1 fusion + residual",
                "Block 2: 64→128, stride 2", "Block 3: 128→256, stride 2", "Global Average Pool → Dropout 0.3", "FC 256→2"]),
    "ResNet18": ("Residual learning with skip connections",
                 ["Input 224×224×3", "Conv 7×7 stem + MaxPool", "4 stages × 2 BasicBlocks (64, 128, 256, 512)",
                  "Global Average Pool", "FC 512→2"]),
    "VGG16": ("Deep stack of 3×3 convolutions",
              ["Input 224×224×3", "13 conv layers (3×3) in 5 blocks with MaxPool", "Flatten 25088",
               "FC 4096 → FC 4096", "FC 4096→2"]),
    "DenseNet121": ("Dense connectivity and feature reuse",
                    ["Input 224×224×3", "Conv 7×7 stem + MaxPool", "4 dense blocks (6, 12, 24, 16 layers) with transitions",
                     "Global Average Pool", "FC 1024→2"]),
    "EfficientNet-B0": ("MBConv blocks with compound scaling",
                        ["Input 224×224×3", "Conv 3×3 stem, 32 channels", "16 MBConv blocks (squeeze-and-excitation)",
                         "Conv 1×1 → 1280", "Global Average Pool → Dropout", "FC 1280→2"]),
    "GhostNet": ("Cheap ghost feature generation",
                 ["Input 224×224×3", "Conv 3×3 stem", "Ghost bottleneck stages", "Global Average Pool",
                  "Conv 1×1 head → 1280", "FC 1280→2"]),
    "Basic CNN": ("Shallow sequential CNN (93,730 parameters)",
                  ["Input 224×224×3", "Sequential Conv → Pool blocks", "Classifier head", "Output: 2 classes",
                   "Exact layer list: see your src/ file"]),
}
INTERP = {
    "Basic CNN": "The shallow sequential design is computationally light, but its limited representation capacity is reflected in lower sensitivity and F1-score.",
    "VGG16": "Uses deep stacked 3×3 convolutions and achieves perfect performance on the evaluated test set, but has a substantially larger parameter count than the proposed model.",
    "ResNet18": "Residual skip connections support strong optimization and near-perfect classification with 11.18M parameters.",
    "DenseNet121": "Dense feature reuse provides strong representation capability and achieves perfect performance on the evaluated test set.",
    "EfficientNet-B0 and GhostNet": "These architectures emphasize efficient feature extraction and achieve perfect performance in the evaluated experiment.",
}

# ----------------------------------------------------------------------------
# Weights discovery + model loading
# ----------------------------------------------------------------------------
EXTS = {".pth", ".pt", ".pkl", ".ckpt", ".bin"}


def find_weights(model, repo, wdir):
    keys = WEIGHT_KEYS[model]
    dirs = [wdir, repo, repo / "models", repo / "checkpoints", repo / "saved_models", repo / "weights", repo / "src"]
    hits = []
    for d in dirs:
        if d.exists():
            it = d.rglob("*") if (d == wdir and d != repo) else d.glob("*")
            hits += [f for f in it if f.suffix.lower() in EXTS and any(k in f.name.lower() for k in keys)]
    hits = sorted(set(hits), key=lambda f: ("best" not in f.name.lower(), len(f.name)))
    return hits[0] if hits else None


@st.cache_data(show_spinner=False)
def detect_classes(root: str):
    base, out = Path(root), []
    files = list((base / "src").rglob("*.py")) + [f for f in base.glob("*.py") if f.name not in ("app.py", "app_v2.py")]
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        for n in tree.body:
            if isinstance(n, ast.ClassDef) and any(ast.unparse(b).endswith("Module") for b in n.bases):
                out.append(f"{f.relative_to(base).as_posix()}:{n.name}")
    return out


def import_class(root, spec):
    rel, cls = spec.rsplit(":", 1)
    root = str(Path(root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    modname = ".".join(Path(rel).with_suffix("").parts)
    try:
        mod = importlib.import_module(modname)
    except Exception:
        sp = importlib.util.spec_from_file_location(Path(rel).stem, Path(root) / rel)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
    return getattr(mod, cls)


def _clean(sd):
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@st.cache_resource(show_spinner="Loading model…")
def load_model(name, path, mtime, class_spec, root):
    """-> (model, error_message)"""
    try:
        try:
            return torch.jit.load(path, map_location="cpu").eval(), None
        except Exception:
            pass
        ckpt = _torch_load(path)
        if isinstance(ckpt, nn.Module):
            m = ckpt
        else:
            tv = torchvision.models
            if name == "VGG16":
                m = tv.vgg16(weights=None); m.classifier[6] = nn.Linear(4096, 2)
            elif name == "ResNet18":
                m = tv.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, 2)
            elif name == "DenseNet121":
                m = tv.densenet121(weights=None); m.classifier = nn.Linear(m.classifier.in_features, 2)
            elif name == "EfficientNet-B0":
                m = tv.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
            elif name == "GhostNet":
                if not TIMM_OK:
                    return None, "needs `pip install timm`"
                m = timm.create_model("ghostnet_100", pretrained=False, num_classes=2)
            else:
                if not class_spec or class_spec == "(none)":
                    return None, "pick its class in 'Custom architectures'"
                cls = import_class(root, class_spec)
                try:
                    m = cls()
                except TypeError:
                    m = cls(num_classes=2)
            m.load_state_dict(_clean(ckpt))
        for mod in m.modules():
            if hasattr(mod, "inplace"):
                mod.inplace = False       # needed for clean Grad-CAM
        return m.eval(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:140]}"

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔬 CRC-Lens")
    repo_root = Path(st.text_input("Project folder", value="."))
    weights_dir = Path(st.text_input("Weights folder", value="models"))
    cancer_idx = st.selectbox("Output index of colon_aca (adenocarcinoma)", [0, 1], index=0,
                              help="ImageFolder sorts alphabetically: colon_aca=0, colon_n=1")
    use_imagenet = st.checkbox("ImageNet normalisation", value=True,
                               help="Untick if you trained on raw 0–1 pixels")
    alpha = st.slider("Heatmap opacity", 0.1, 0.9, 0.45, 0.05)
    strict_check = st.checkbox("Reject non-histology images", value=True,
                               help="Blocks photos/graphics that are not H&E-stained tissue")
    relaxed = st.checkbox("Relaxed validation", value=False,
                          help="Use if valid tissue images are being rejected") if strict_check else False

    classes = detect_classes(str(repo_root))
    with st.expander("Custom architectures (Basic CNN, proposed model)"):
        st.caption("Model classes found in the project (src/ and root). Importing a file runs it, so keep training code "
                   "under `if __name__ == '__main__':`.")
        spec = {}
        for m in ("Basic CNN", PROPOSED):
            guess = [i for i, c in enumerate(classes)
                     if any(k in c.lower() for k in (["basic", "simple"] if m == "Basic CNN" else ["multi", "rms"]))]
            opts = ["(none)"] + classes
            spec[m] = st.selectbox(m, opts, index=(guess[0] + 1) if guess else 0, key=f"spec_{m}")
    with st.expander("Weights file overrides"):
        override = {m: st.text_input(m, value="", key=f"ov_{m}", placeholder="auto-detect") for m in MODELS}

found = {}
for m in MODELS:
    p = Path(override[m]) if override[m].strip() else find_weights(m, repo_root, weights_dir)
    found[m] = p if (p and Path(p).exists()) else None
avail = [m for m in MODELS if found[m]]

with st.sidebar:
    run_models = st.multiselect("Models to run", avail, default=avail,
                                help="Untick heavy models (e.g. VGG16) to speed things up")

loaded, status = {}, {}
for m in MODELS:
    if not found[m]:
        status[m] = "No weights found"
    elif m not in run_models:
        status[m] = "Found (not selected)"
    elif not TORCH_OK:
        status[m] = "PyTorch missing"
    else:
        mdl, err = load_model(m, str(found[m]), found[m].stat().st_mtime, spec.get(m, ""), str(repo_root))
        if mdl is None:
            status[m] = f"Error: {err}"
        else:
            loaded[m] = mdl
            status[m] = "Loaded ✓"

DEMO = len(loaded) == 0
with st.sidebar:
    primary = st.selectbox("Primary model (explanations)", list(loaded) if loaded else ["Demo heuristic"],
                           index=(list(loaded).index(PROPOSED) if PROPOSED in loaded else 0) if loaded else 0)
    with st.expander("Model status", expanded=DEMO):
        st.dataframe(pd.DataFrame({"Model": MODELS,
                                   "Weights": [found[m].name if found[m] else "—" for m in MODELS],
                                   "Status": [status[m] for m in MODELS]}), hide_index=True)
    use_medgemma = st.checkbox("MedGemma narrative (needs GPU + HF access)", value=False)
    medgemma_id = st.text_input("MedGemma id", "google/medgemma-4b-it") if use_medgemma else ""

# ----------------------------------------------------------------------------
# Inference + explanation
# ----------------------------------------------------------------------------
def prep_square(img, size=256):
    w, h = img.size
    if w != h:
        side = max(w, h)
        img = ImageOps.pad(img, (side, side), color=(255, 255, 255))
    return img.resize((size, size), Image.LANCZOS)


def validate_histology(img, strict=True):
    """Heuristic gate: does this look like an H&E-stained histology image?
    Returns (ok, message, metrics)."""
    small = img.convert("RGB").resize((256, 256))
    rgb = np.asarray(small).astype(np.float32)
    hsv = np.asarray(small.convert("HSV")).astype(np.float32)   # PIL HSV: all channels 0-255
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    background = (S < 25) & (V > 200)            # white slide background / glare
    tissue = ~background
    tissue_frac = float(tissue.mean())
    if tissue.sum() < 500:
        return False, "The image is almost entirely blank or white.", {"tissue_frac": tissue_frac}

    t = tissue
    mean_sat = float(S[t].mean())
    # H&E: green is the weakest channel in pink (eosin) and purple (haematoxylin) pixels
    g_min = float(((G <= R) & (G <= B))[t].mean())
    # stain hues: purple/magenta/pink/red-pink  (PIL hue >=170 or <=12)
    sat_mask = t & (S > 30)
    stain_hue = float((((H >= 170) | (H <= 12))[sat_mask]).mean()) if sat_mask.sum() else 0.0
    br_ratio = float(B[t].mean() / (R[t].mean() + 1e-6))     # rejects pure red/orange objects
    gray_std = float(rgb.mean(-1)[t].std())

    m = {"tissue_frac": tissue_frac, "mean_sat": mean_sat, "g_min": g_min,
         "stain_hue": stain_hue, "br_ratio": br_ratio, "gray_std": gray_std}

    k = 1.0 if strict else 0.85                  # relaxed mode loosens every threshold
    problems = []
    if tissue_frac < 0.40 * k:
        problems.append("too little stained tissue")
    if mean_sat < 30 * k:
        problems.append("colours are too grey/washed out for H&E staining")
    if g_min < 0.85 * k:
        problems.append("colour distribution does not match H&E (pink/purple) staining")
    if stain_hue < 0.80 * k:
        problems.append("dominant colours are outside the pink–purple stain range")
    if not (0.6 * k <= br_ratio <= 1.8):
        problems.append("red/blue balance is not typical of stained tissue")
    if gray_std < 12 * k:
        problems.append("image is too flat – no tissue texture")

    if problems:
        return False, ("Invalid image: this does not look like an H&E-stained colon histopathology image ("
                       + "; ".join(problems) + ")."), m
    return True, "Image looks like H&E histopathology.", m


def _tfm():
    t = [transforms.Resize((224, 224)), transforms.ToTensor()]
    if use_imagenet:
        t.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(t)


def demo_aca(img):
    a = np.asarray(img.convert("RGB").resize((128, 128))).astype(float) / 255
    d = ((a[..., 2] > a[..., 1] + 0.05) & (a.mean(-1) < 0.55)).mean()
    return float(1 / (1 + np.exp(-12 * (d - 0.25))))


def model_probs(model, imgs):
    """-> [N,2] = (p_normal, p_adenocarcinoma) in the sidebar's class mapping."""
    x = torch.stack([_tfm()(i.convert("RGB")) for i in imgs])
    with torch.no_grad():
        out = model(x)
    if out.ndim == 1:
        out = out.unsqueeze(1)
    p = torch.cat([1 - torch.sigmoid(out), torch.sigmoid(out)], 1) if out.shape[1] == 1 else F.softmax(out, 1)
    p = p.numpy()
    return np.stack([p[:, 1 - cancer_idx], p[:, cancer_idx]], 1)


def aca_list(imgs, name):
    if DEMO or name not in loaded:
        return np.array([demo_aca(i) for i in imgs])
    return model_probs(loaded[name], imgs)[:, 1]


def run_all(work):
    if DEMO:
        p = demo_aca(work)
        return {"Demo heuristic": np.array([1 - p, p])}
    return {n: model_probs(m, [work])[0] for n, m in loaded.items()}


def gradcam(model, work, target):
    x = _tfm()(work).unsqueeze(0).requires_grad_(True)
    acts, hooks = [], []

    def hk(_m, _i, o):
        if torch.is_tensor(o) and o.ndim == 4 and o.shape[-1] >= 2 and o.shape[-2] >= 2 and o.requires_grad:
            acts.append(o)

    for m in model.modules():
        hooks.append(m.register_forward_hook(hk))
    try:
        out = model(x)
    finally:
        for h in hooks:
            h.remove()
    if not acts:
        raise RuntimeError("no spatial layer found")
    a = acts[-1]
    if out.ndim == 2 and out.shape[1] > 1:
        score = out[0, target]
    else:
        score = out.reshape(-1)[0] * (1 if target == 1 else -1)
    g = torch.autograd.grad(score, a)[0]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=work.size[::-1], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    cam -= cam.min()
    return cam / (cam.max() + 1e-8)


def occlusion(model, work, target_pos, grid=8):
    """Model-agnostic fallback when Grad-CAM can't hook the network."""
    arr = np.asarray(work).copy()
    S = arr.shape[0]
    step = S // grid
    mean = arr.mean((0, 1))
    tiles = []
    for r in range(grid):
        for c in range(grid):
            a = arr.copy()
            a[r * step:(r + 1) * step, c * step:(c + 1) * step] = mean
            tiles.append(Image.fromarray(a.astype(np.uint8)))
    base = model_probs(model, [work])[0, target_pos]
    drop = np.clip(base - model_probs(model, tiles)[:, target_pos], 0, None).reshape(grid, grid)
    cam = np.asarray(Image.fromarray(drop.astype(np.float32)).resize((S, S), Image.BICUBIC))
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


def demo_cam(img):
    a = np.asarray(img.convert("RGB")).astype(float) / 255
    d = Image.fromarray((((a[..., 2] > a[..., 1] + 0.05) & (a.mean(-1) < 0.55)) * 255).astype(np.uint8))
    c = np.asarray(d.filter(ImageFilter.GaussianBlur(14))).astype(float)
    c -= c.min()
    return c / (c.max() + 1e-8)


def explain(work, target_pos):
    if DEMO:
        return demo_cam(work), "Demo heat map"
    model = loaded[primary]
    try:
        return gradcam(model, work, cancer_idx if target_pos == 1 else 1 - cancer_idx), "Grad-CAM"
    except Exception:
        return occlusion(model, work, target_pos), "Occlusion sensitivity (Grad-CAM unavailable for this model)"


def colorize(cam):
    return Image.fromarray((mpl.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8))


def blend(img, cam):
    heat = np.asarray(colorize(cam)).astype(float) / 255
    base = np.asarray(img.convert("RGB")).astype(float) / 255
    m = (cam[..., None] ** 0.8) * alpha
    return Image.fromarray((np.clip(base * (1 - m) + heat * m, 0, 1) * 255).astype(np.uint8))


def show(img, caption=None):
    st.image(img.resize((DISPLAY, DISPLAY)), width=DISPLAY, caption=caption)


def perturb(img, kind, lv):
    """Apply one robustness perturbation using a guaranteed Python float.

    Streamlit/NumPy sliders and np.linspace can sometimes supply NumPy scalar/
    array values. Pillow expects native scalar values for several filters;
    converting here prevents the ambiguous-array ValueError.
    """
    try:
        arr = np.asarray(lv)
        lv = float(arr.reshape(-1)[0])
    except Exception:
        lv = float(lv)

    if kind == "Brightness":
        return ImageEnhance.Brightness(img).enhance(float(1.0 + lv))
    if kind == "Contrast":
        return ImageEnhance.Contrast(img).enhance(float(1.0 + lv))
    if kind == "Blur":
        radius = float(abs(lv) * 6.0)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    if kind == "Stain shift (pink↔purple)":
        a = np.asarray(img).astype(np.float32).copy()
        a[..., 0] *= float(1.0 - 0.25 * lv)
        a[..., 2] *= float(1.0 + 0.25 * lv)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if kind == "JPEG compression":
        buf = io.BytesIO()
        quality = int(max(5, min(95, 95 - abs(lv) * 90)))
        img.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "Rotation":
        return img.rotate(float(lv * 180.0), resample=Image.BILINEAR)
    return img

# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------
def make_txt(r):
    lines = [
        "CRC-Lens technical report (research prototype, not for clinical diagnosis)",
        f"Generated: {r['time']}", f"Image: {r['file']}", f"Primary model: {r['model']}", "",
        f"Prediction: {r['label']}", f"Model confidence: {r['conf']:.2f}%",
        f"{LBL_N}: {r['p_n']:.4f}%", f"{LBL_A}: {r['p_a']:.4f}%", "",
        f"Explanation method: {r['method']}",
        f"Heatmap area at or above 50% of peak activation: {r['area50']:.2f}%", "",
        "All models:",
    ] + [f"  {a}: {b} ({c})" for a, b, c in r["rows"]] + [
        "", "Grad-CAM shows model attribution only. It is not cell segmentation, a tumour boundary or a clinical finding."]
    return "\n".join(lines)


def make_pdf(r, imgs):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as RL, Paragraph, SimpleDocTemplate, Spacer, Table
    ss = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm)
    row = []
    for im in imgs:
        b = io.BytesIO()
        im.resize((256, 256)).save(b, "PNG")
        b.seek(0)
        row.append(RL(b, width=52 * mm, height=52 * mm))
    story = [Paragraph("CRC-Lens technical report", ss["Title"]),
             Paragraph("Research prototype - not for clinical diagnosis", ss["Italic"]), Spacer(1, 6),
             Paragraph(f"Image: {r['file']} | Generated: {r['time']} | Primary model: {r['model']}", ss["Normal"]),
             Spacer(1, 8), Paragraph(f"<b>Prediction:</b> {r['label']} ({r['conf']:.2f}% confidence)", ss["Heading3"]),
             Paragraph(f"{LBL_N}: {r['p_n']:.4f}% | {LBL_A}: {r['p_a']:.4f}%", ss["Normal"]), Spacer(1, 8),
             Table([row]), Spacer(1, 4),
             Paragraph("Left to right: original, heatmap, overlay.", ss["Italic"]), Spacer(1, 8),
             Paragraph(f"Explanation method: {r['method']}. Heatmap area at or above 50% of peak: {r['area50']:.2f}%.",
                       ss["Normal"]), Spacer(1, 8), Paragraph("All models", ss["Heading3"])]
    story += [Paragraph(f"{a}: {b} ({c})", ss["Normal"]) for a, b, c in r["rows"]]
    story += [Spacer(1, 10), Paragraph("The heatmap shows model attribution only. It is not cell segmentation, a tumour "
                                        "boundary or a clinical finding.", ss["Italic"])]
    doc.build(story)
    return buf.getvalue()


@st.cache_resource(show_spinner=False)
def load_medgemma(mid):
    from transformers import pipeline
    return pipeline("image-text-to-text", model=mid, torch_dtype=torch.bfloat16, device_map="auto")


def medgemma_text(img, p):
    out = load_medgemma(medgemma_id)(text=[
        {"role": "system", "content": [{"type": "text", "text": "You are an expert gastrointestinal pathologist."}]},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text":
            f"A CNN estimated {p:.0%} probability of adenocarcinoma for this colon H&E patch. Describe visible "
            "histological features and a cautious impression. Do not give a definitive diagnosis."}]}], max_new_tokens=300)
    return out[0]["generated_text"][-1]["content"]

# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.markdown('<div class="hero"><h1>CRC-Lens</h1><p>Explainable image classification for colorectal '
            'histopathology research</p></div>', unsafe_allow_html=True)
if DEMO:
    st.markdown('<div class="banner">⚠️ <b>Demo mode</b> – no usable weights were found, so results come from a '
                'colour heuristic purely to preview the layout. Open <b>Model status</b> in the sidebar to see why '
                'each model was skipped.</div>', unsafe_allow_html=True)

tab_an, tab_err, tab_cmp = st.tabs(
    ["🔬 Image Analysis", "🔍 Error Analysis", "📊 Model Comparison"])


def current_image():
    an = st.session_state.get("analysis")
    if not an:
        return None, None
    return prep_square(Image.open(io.BytesIO(an["bytes"])).convert("RGB")), an["name"]


# ============================ Image Analysis ================================
with tab_an:
    up = st.file_uploader("Upload a histopathology image", type=["jpg", "jpeg", "png"])
    data, fname = None, None
    if up is not None:
        data, fname = up.getvalue(), up.name
    else:
        sd = repo_root / "data" / "colon_image_sets"
        samples = sorted(list((sd / "colon_aca").glob("*.jpeg"))[:8] + list((sd / "colon_n").glob("*.jpeg"))[:8])
        if samples:
            with st.expander("…or pick a sample from the dataset"):
                pick = st.selectbox("Sample", ["—"] + [f"{s.parent.name}/{s.name}" for s in samples])
                if pick != "—":
                    data, fname = (sd / pick).read_bytes(), pick.split("/")[-1]
    if data is None:
        st.caption("Upload a PNG or JPEG histopathology image to begin analysis.")
    else:
        try:
            pil_in = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            pil_in = None
            st.error("❌ Invalid file: this could not be read as an image.")
            st.session_state.pop("analysis", None)

        if pil_in is not None:
            sq = prep_square(pil_in)
            show(sq, "Selected image")
            ok, msg, vm = (True, "", {}) if not strict_check else validate_histology(sq, strict=not relaxed)
            if not ok:
                st.session_state.pop("analysis", None)      # clear any previous result
                st.error(f"❌ {msg}")
                st.caption("Please upload a colon tissue H&E microscopy image (e.g. from LC25000 or a "
                           "histopathology source). Natural photos, drawings, screenshots and X-ray/CT "
                           "images are not supported.")
                with st.expander("Why was this rejected?"):
                    st.json({k: round(v, 3) for k, v in vm.items()})
            else:
                if st.button("Analyze image", type="primary"):
                    st.session_state["analysis"] = {"bytes": data, "name": fname}

    work, name = current_image()
    if work is not None:
        st.divider()
        key = (hashlib.md5(st.session_state["analysis"]["bytes"]).hexdigest(), tuple(loaded), cancer_idx, use_imagenet)
        cache = st.session_state.setdefault("pred_cache", {})
        if key not in cache:
            with st.spinner("Running models…"):
                cache.clear()
                cache[key] = run_all(work)
        preds = cache[key]
        pmain = preds[primary if primary in preds else list(preds)[0]]
        is_a = pmain[1] >= pmain[0]
        label = LBL_A if is_a else LBL_N
        conf = float(max(pmain)) * 100
        if conf < 85:
            st.warning("Low model confidence. The image may be out-of-distribution (different stain, "
                       "scanner or tissue type), so treat this result with extra caution.")

        st.header("Analysis result")
        c1, c2 = st.columns([3, 1])
        c1.markdown(f'<div class="card {"bad" if is_a else "good"}"><div class="lbl">Model prediction · {primary}</div>'
                    f'<div class="pred">{label}</div><div class="lbl" style="text-transform:none;margin-top:.5rem">'
                    f'Model confidence is not the same as clinical certainty.</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="card"><div class="lbl">Confidence</div><div class="conf">{conf:.2f}%</div></div>',
                    unsafe_allow_html=True)

        st.header("Visual explanation")
        tgt = st.radio("Explain which class?", ["Predicted class", LBL_A, LBL_N], horizontal=True)
        target_pos = (1 if is_a else 0) if tgt == "Predicted class" else (1 if tgt == LBL_A else 0)
        cam, method = explain(work, target_pos)
        heat, over = colorize(cam), blend(work, cam)
        area50 = float((cam >= 0.5).mean() * 100)
        a, b, c = st.columns(3)
        for col, title, im in ((a, "Original image", work), (b, f"{method.split(' (')[0]} heatmap", heat),
                               (c, f"{method.split(' (')[0]} overlay", over)):
            with col:
                st.markdown(f"**{title}**")
                show(im)
        if "Occlusion" in method:
            st.info("Grad-CAM could not hook this network, so occlusion sensitivity is shown instead.")

        arr = np.asarray(work).astype(float)
        tone = "purple/blue-leaning" if arr[..., 2].mean() > arr[..., 0].mean() else "pink-leaning"
        st.markdown(f'<div class="card"><h4>Original image</h4>The uploaded image is a microscopic colorectal tissue '
                    f'image with {tone} staining and visible variation in tissue texture and structure. This describes '
                    f'visual appearance only and does not independently determine pathological status.</div>'
                    f'<div class="card"><h4>{method.split(" (")[0]} heatmap</h4>Warmer colours represent higher relative '
                    f'activation and mark regions that contributed more strongly to the prediction. They do not identify '
                    f'confirmed cancer cells or define a tumour boundary.</div>'
                    f'<div class="card"><h4>Overlay</h4>The overlay places the activation map on the original tissue so '
                    f'you can see where highlighted regions fall in the field of view. It represents model attribution, '
                    f'not cell segmentation or a clinical finding.</div>', unsafe_allow_html=True)

        st.header("Class probabilities")
        for lab, v in ((LBL_N, pmain[0]), (LBL_A, pmain[1])):
            st.progress(float(v), text=f"{lab}: {v * 100:.4f}%")

        if len(preds) > 1:
            st.header("All models at a glance")
            rows = [(n, (LBL_A if p[1] >= p[0] else LBL_N), f"{max(p) * 100:.2f}%") for n, p in preds.items()]
            st.dataframe(pd.DataFrame(rows, columns=["Model", "Prediction", "Confidence"]), hide_index=True)
            votes = sum(1 for p in preds.values() if p[1] >= p[0])
            st.caption(f"{votes} of {len(preds)} models predict {LBL_A}; {len(preds) - votes} predict {LBL_N}.")
        else:
            rows = [(n, (LBL_A if p[1] >= p[0] else LBL_N), f"{max(p) * 100:.2f}%") for n, p in preds.items()]

        st.header("Model interpretation")
        st.markdown(f'<div class="card"><h4>🎯 Why did the model make this prediction?</h4>The model classified this '
                    f'image as <b>{label}</b>. The visualization highlights tissue regions that contributed most '
                    f'strongly to this prediction. Highlighted regions represent model attribution and should not be '
                    f'read as confirmed cancer cells, a tumour boundary, or a clinical diagnosis.</div>',
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="lbl">Prediction</div><div style="font-size:1.25rem;font-weight:600">{label}</div>',
                    unsafe_allow_html=True)
        m2.markdown(f'<div class="lbl">Model confidence</div><div style="font-size:1.25rem;font-weight:600">{conf:.2f}%</div>',
                    unsafe_allow_html=True)
        m3.markdown(f'<div class="lbl">Explanation method</div><div style="font-size:1.25rem;font-weight:600">'
                    f'{method.split(" (")[0]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><h4>🔥 How to read the heatmap</h4>Warmer regions indicate relatively stronger '
                    'model activation, cooler regions lower activation. It explains model behaviour; it is not cell '
                    'segmentation or a clinical finding.</div>', unsafe_allow_html=True)
        st.info("Detailed numerical measurements are kept in the technical TXT/PDF report and hidden from this view.")

        if use_medgemma and TORCH_OK:
            if st.button("Generate MedGemma narrative"):
                with st.spinner("MedGemma is reading the image…"):
                    try:
                        st.markdown(medgemma_text(work, float(pmain[1])))
                    except Exception as e:
                        st.error(f"MedGemma failed: {e}")

        st.header("Summary")
        st.markdown(f'<div class="card">The model predicts <b>{label}</b> with <b>{conf:.2f}%</b> model confidence. '
                    f'The heatmap and overlay show the image regions that contributed to this prediction. This '
                    f'explanation describes model behaviour and is not a clinical diagnosis.</div>', unsafe_allow_html=True)
        rep = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "file": name, "model": primary, "label": label,
               "conf": conf, "p_n": pmain[0] * 100, "p_a": pmain[1] * 100, "method": method, "area50": area50,
               "rows": rows}
        d1, d2 = st.columns(2)
        d1.download_button("Download technical report (TXT)", make_txt(rep), "crc_lens_report.txt")
        if PDF_OK:
            d2.download_button("Download technical report (PDF)", make_pdf(rep, [work, heat, over]), "crc_lens_report.pdf")
        else:
            d2.caption("Install `reportlab` to enable the PDF report.")

# ============================== Error Analysis ==============================
with tab_err:
    st.header("🔍 Misclassified Image Analysis")
    st.caption(
        "Complete misclassification gallery for every model for which your project "
        "contains generated error visualizations. Images are shown from the actual "
        "results folders; nothing is generated or invented by the dashboard."
    )

    MISCLASSIFIED_DIRS = {
        "Basic CNN": repo_root / "results" / "basic_cnn_xai" / "misclassified",
        "VGG16": repo_root / "results" / "vgg16_xai" / "misclassified",
        "ResNet18": repo_root / "results" / "resnet18_xai" / "misclassified",
        "DenseNet121": repo_root / "results" / "densenet121_xai" / "misclassified",
        "EfficientNet-B0": repo_root / "results" / "efficientnetb0_xai" / "misclassified",
        "GhostNet": repo_root / "results" / "ghostnet_xai" / "misclassified",
        "RMSCNN ⭐ Proposed": repo_root / "results" / "rmscnn_xai" / "misclassified",
    }

    MISCLASSIFIED_FALLBACKS = {
        "Basic CNN": [repo_root / "results" / "basic_cnn" / "misclassified"],
        "VGG16": [repo_root / "results" / "vgg16" / "misclassified"],
        "ResNet18": [repo_root / "results" / "resnet18" / "misclassified", repo_root / "resnet18_misclassified_images"],
        "DenseNet121": [repo_root / "results" / "densenet121" / "misclassified"],
        "EfficientNet-B0": [repo_root / "results" / "efficientnet_b0_xai" / "misclassified", repo_root / "results" / "efficientnet_b0" / "misclassified"],
        "GhostNet": [repo_root / "results" / "ghostnet" / "misclassified"],
        "RMSCNN ⭐ Proposed": [repo_root / "results" / "rmscnn" / "misclassified"],
    }

    def resolve_misclassified_dir(model_name):
        primary_dir = MISCLASSIFIED_DIRS[model_name]
        if primary_dir.exists():
            return primary_dir
        for candidate in MISCLASSIFIED_FALLBACKS.get(model_name, []):
            if candidate.exists():
                return candidate
        return primary_dir

    def get_misclassified_images(directory):
        if not directory.exists():
            return []
        files = sorted(directory.glob("*_misclassified.png"))
        if not files:
            files = sorted(directory.glob("*.png"))
        return files

    selected_error_model = st.selectbox(
        "Select model",
        ["All Models"] + list(MISCLASSIFIED_DIRS.keys()),
        key="misclassified_model",
    )

    if selected_error_model == "All Models":
        st.markdown("### All model misclassifications")
        st.info(
            "Each expandable section contains the complete set of available "
            "misclassified images for that model."
        )

        for model_name in MISCLASSIFIED_DIRS:
            actual_dir = resolve_misclassified_dir(model_name)
            image_files = get_misclassified_images(actual_dir)

            with st.expander(
                f"{model_name} — {len(image_files)} misclassified images",
                expanded=(model_name == "RMSCNN ⭐ Proposed"),
            ):
                if image_files:
                    st.caption(f"Source: {actual_dir}")
                    for start in range(0, len(image_files), 4):
                        row = image_files[start:start + 4]
                        cols = st.columns(4)
                        for col, image_path in zip(cols, row):
                            with col:
                                try:
                                    st.image(str(image_path), width="stretch")
                                    st.caption(image_path.stem)
                                except Exception as exc:
                                    st.error(f"Could not load {image_path.name}: {exc}")
                else:
                    st.warning(
                        "No misclassified PNG files were found for this model in the "
                        "expected results folders."
                    )

    else:
        actual_dir = resolve_misclassified_dir(selected_error_model)
        image_files = get_misclassified_images(actual_dir)
        st.markdown(f"### {selected_error_model} — Misclassified Samples")

        if image_files:
            st.success(f"{len(image_files)} misclassified visualizations found.")
            st.caption(f"Source: {actual_dir}")

            for start in range(0, len(image_files), 4):
                row = image_files[start:start + 4]
                cols = st.columns(4)
                for col, image_path in zip(cols, row):
                    with col:
                        try:
                            st.image(str(image_path), width="stretch")
                            st.caption(image_path.stem)
                        except Exception as exc:
                            st.error(f"Could not load {image_path.name}: {exc}")

            report_candidates = [
                actual_dir / "misclassification_summary.txt",
                actual_dir / "misclassification_report.txt",
            ]
            report_file = next((p for p in report_candidates if p.exists()), None)
            if report_file:
                st.divider()
                st.subheader("📄 Misclassification Report")
                with st.expander("View report"):
                    report_text = report_file.read_text(encoding="utf-8", errors="replace")
                    st.code(report_text, language="text")
                safe_name = selected_error_model.replace(" ", "_").replace("⭐", "").strip("_")
                st.download_button(
                    "⬇️ Download report",
                    data=report_file.read_bytes(),
                    file_name=f"{safe_name}_misclassification_report.txt",
                    mime="text/plain",
                )
        else:
            st.warning(
                f"No misclassified images were found for {selected_error_model}.\n\n"
                f"Checked: {actual_dir}"
            )

# ============================== Model Comparison ============================
with tab_cmp:
    plt.rcParams.update({"axes.facecolor": "#0e1117", "figure.facecolor": "#0e1117", "text.color": "#e6ebff",
                         "axes.labelcolor": "#e6ebff", "xtick.color": "#c9d2f0", "ytick.color": "#c9d2f0",
                         "axes.edgecolor": "#3a4468", "axes.grid": True, "grid.color": "#222b45"})
    short = {PROPOSED: "RM-S CNN\n(Proposed)"}
    names = [short.get(m, m) for m in MODELS]

    st.subheader("Model Comparison & Architecture Analysis")
    st.caption("Primary group-aware LC25000 colon classification experiments. InceptionV3 is intentionally excluded.")

    st.markdown("### 1. Models Evaluated")
    st.dataframe(pd.DataFrame({"Model": MODELS}), hide_index=True)

    st.markdown("### 2. Architecture Comparison")
    st.dataframe(ARCH, hide_index=True)

    st.markdown("### 3. Architecture Visualization")
    sel = st.selectbox("Select a model", MODELS, index=len(MODELS) - 1)
    sub, steps = FLOW[sel]
    html = "".join(f"<b>{s}</b>" + ("<span>↓</span>" if i < len(steps) - 1 else "") for i, s in enumerate(steps))
    st.markdown(f'<div class="card flow"><h3 style="margin:0">{sel}</h3><span>{sub}</span><br>{html}</div>',
                unsafe_allow_html=True)
    if sel == PROPOSED:
        st.success("Proposed architecture: multi-scale convolution branches are fused and combined with a residual "
                   "shortcut before activation.")

    st.markdown("### 4. Classification Performance Comparison")
    st.dataframe(METRICS.style.format({c: "{:.4f}" for c in METRICS.columns[1:]}), hide_index=True)

    x = np.arange(len(MODELS))
    f1, a1 = plt.subplots(figsize=(8.5, 3.6))
    a1.bar(x, METRICS["Accuracy (%)"], color="#7c6cff")
    a1.set_ylim(90, 100.6); a1.set_xticks(x, names, fontsize=8); a1.set_ylabel("Accuracy (%) – axis starts at 90")
    st.markdown("**Accuracy comparison**"); st.pyplot(f1); plt.close(f1)

    cols = ["Accuracy (%)", "Precision (%)", "Sensitivity (%)", "Specificity (%)", "F1 (%)"]
    f2, a2 = plt.subplots(figsize=(8.5, 3.8))
    w = 0.16
    for i, c in enumerate(cols):
        a2.bar(x + (i - 2) * w, METRICS[c], w, label=c.replace(" (%)", ""))
    a2.set_ylim(80, 101); a2.set_xticks(x, names, fontsize=8); a2.set_ylabel("Score (%) – axis starts at 80")
    a2.legend(ncol=5, fontsize=7, loc="lower right")
    st.markdown("**Overall performance metrics**"); st.pyplot(f2); plt.close(f2)

    f3, a3 = plt.subplots(figsize=(8.5, 3.2))
    a3.bar(x, METRICS["ROC-AUC (%)"], color="#22d3c5")
    a3.set_ylim(99.5, 100.05); a3.set_xticks(x, names, fontsize=8); a3.set_ylabel("ROC-AUC (%) – axis starts at 99.5")
    st.markdown("**ROC-AUC comparison**"); st.pyplot(f3); plt.close(f3)

    st.markdown("### 5. Parameter Efficiency & Computational Complexity")
    rows = []
    for m in MODELS:
        cx = COMPLEX.get(m)
        p = PARAMS[m]
        if cx:
            extra = [f"{cx[0]:,.4f}", f"{cx[1]:,.4f}", f"{cx[2]:,.3f}", f"{cx[3]:,.2f}"]
        else:
            extra = ["—", "—", "—", "—"]
        status_txt = "Measured" if cx else ("Parameters only" if p else "Not recorded")
        rows.append([m, f"{p:,}" if p else "N/A"] + extra + [status_txt])
    st.dataframe(pd.DataFrame(rows, columns=["Model", "Parameters", "MACs (G)", "FLOPs (G)", "Inference (ms)",
                                             "Images/s", "Status"]), hide_index=True)

    f4, a4 = plt.subplots(figsize=(8.5, 3.8))
    ys = np.arange(len(MODELS))[::-1]
    for y, m in zip(ys, MODELS):
        p = PARAMS[m]
        if p:
            a4.barh(y, p, color="#ff4d8d" if m == PROPOSED else "#7c6cff")
            a4.text(p * 1.15, y, f"{p:,}", va="center", fontsize=8)
        else:
            a4.text(1e5, y, "N/A (not recorded)", va="center", fontsize=8, color="#ffd36b")
    a4.set_xscale("log"); a4.set_yticks(ys, [m.replace(" (Proposed)", "*") for m in MODELS], fontsize=8)
    a4.set_xlim(5e4, 1e9); a4.set_xlabel("Parameters (log scale)  ·  * = proposed model")
    st.markdown("**Parameter count – all primary models**"); st.pyplot(f4); plt.close(f4)
    st.caption("GhostNet is marked N/A because its project-specific parameter count was not recorded; "
               "no external value is substituted.")

    st.markdown("### 5A. Parameter Efficiency vs. F1 Score")
    st.caption("Lower parameter count and higher F1 indicate a more efficient architecture. GhostNet is omitted from the plot because its project-specific parameter count was not recorded.")

    f5, a5 = plt.subplots(figsize=(9.2, 4.6))
    f5.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.18)

    # True values remain on the axes; labels are deliberately offset so the
    # near-100% F1 models do not overlap each other.
    label_offsets = {
        "Basic CNN": (8, 10),
        "VGG16": (8, 14),
        "ResNet18": (8, -20),
        "DenseNet121": (8, -4),
        "EfficientNet-B0": (8, 22),
        PROPOSED: (8, -34),
    }

    for m, f1v in zip(MODELS, METRICS["F1 (%)"]):
        pcount = PARAMS[m]
        if not pcount:
            continue
        is_prop = m == PROPOSED
        a5.scatter(
            pcount, f1v,
            s=125 if is_prop else 82,
            c="#ff4d8d" if is_prop else "#7c6cff",
            edgecolors="#ffffff",
            linewidths=0.8,
            zorder=4,
        )
        dx, dy = label_offsets.get(m, (8, 8))
        label = "RMSCNN ⭐" if is_prop else m
        a5.annotate(
            label,
            (pcount, f1v),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold" if is_prop else "normal",
            color="#ff4d8d" if is_prop else "#e6ebff",
            arrowprops=(dict(arrowstyle="-", color="#ff4d8d", lw=0.8, alpha=0.65) if is_prop else None),
        )

    a5.axhline(99.9, color="#68739a", linestyle="--", linewidth=0.8, alpha=0.55)
    a5.text(0.99, 99.9, "99.9% F1", transform=a5.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5, color="#9aa7c7")
    a5.set_xscale("log")
    a5.set_xlim(5e4, 2.2e8)
    a5.set_ylim(92.0, 100.65)
    a5.set_xlabel("Trainable parameters (log scale)", labelpad=8)
    a5.set_ylabel("F1 score (%)", labelpad=8)
    a5.grid(True, which="major", alpha=0.28)
    a5.grid(False, which="minor", axis="y")
    a5.set_title("High F1 with fewer parameters is preferred", loc="left", fontsize=11, fontweight="bold", pad=10)

    # Minimal legend: proposed model vs. other architectures.
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ff4d8d", markeredgecolor="#ffffff",
               markersize=8, label="RMSCNN (Proposed)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7c6cff", markeredgecolor="#ffffff",
               markersize=7, label="Other models"),
    ]
    a5.legend(handles=legend_items, loc="lower right", frameon=True, fontsize=8)

    st.pyplot(f5, width="stretch")
    plt.close(f5)

    st.markdown(
        '<div class="card"><h4>How to read this chart</h4>'
        'Models farther <b>left</b> use fewer parameters, while models higher on the chart have better F1. '
        '<b>RMSCNN ⭐</b> is highlighted because it combines near-perfect F1 with a substantially smaller parameter count '
        'than ResNet18, DenseNet121 and VGG16 in the recorded experiments.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 6. Architecture–Performance Interpretation")
    for k, v in INTERP.items():
        st.markdown(f"**{k}:** {v}")
    red = (PARAMS["ResNet18"] - PARAMS[PROPOSED]) / PARAMS["ResNet18"] * 100
    st.markdown(f"**Residual Multi-Scale CNN (Proposed):** Combines residual learning with parallel 3×3, 5×5 and dilated "
                f"3×3 branches. It achieves 99.93% accuracy and 99.94% F1-score using 2.16M parameters. Compared with "
                f"ResNet18 this is approximately {red:.1f}% fewer parameters, although the parallel branches increase "
                f"FLOPs and CPU inference time.")

    st.markdown("### ⭐ Proposed Model Highlight")
    h = st.columns(5)
    for col, (lab, val) in zip(h, [("Accuracy", "99.93%"), ("F1 Score", "99.94%"), ("Sensitivity", "100.00%"),
                                   ("ROC-AUC", "99.9998%"), ("Parameters", "2.16M")]):
        col.metric(lab, val)
    st.info("The proposed model is not presented as the highest-accuracy model. Its contribution is the combination of "
            "multi-scale residual feature extraction and a favorable parameter-efficiency trade-off.")

    # ------------------------------------------------------------------
    # All-model training evidence
    # ------------------------------------------------------------------
    st.markdown("### Training evidence — all models")
    st.caption(
        "This section is model-agnostic: it no longer defaults to whichever single plot happens to be selected. "
        "The overview compares every evaluated model, while the training-curve gallery shows every available "
        "accuracy/loss plot found in the project results."
    )

    # Overall model-performance graph. This uses the recorded test metrics above,
    # so it remains an actual quantitative comparison rather than a montage of images.
    metric_cols = st.columns(2)
    with metric_cols[0]:
        f_train_acc, ax_train_acc = plt.subplots(figsize=(6.8, 4.2))
        x_all = np.arange(len(MODELS))
        acc_vals = METRICS["Accuracy (%)"].to_numpy(dtype=float)
        bars = ax_train_acc.bar(x_all, acc_vals)
        ax_train_acc.set_xticks(x_all, [m.replace(" (Proposed)", "\n⭐ RMSCNN") if m == PROPOSED else m.replace("-", "-\n") for m in MODELS], fontsize=8)
        ax_train_acc.set_ylabel("Accuracy (%)")
        ax_train_acc.set_ylim(max(90, float(acc_vals.min()) - 2), 100.6)
        ax_train_acc.set_title("Overall accuracy — all models", pad=10, fontweight="bold")
        ax_train_acc.grid(axis="y", alpha=0.22)
        for b, v in zip(bars, acc_vals):
            ax_train_acc.text(b.get_x() + b.get_width()/2, v + 0.08, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.5)
        f_train_acc.tight_layout()
        st.pyplot(f_train_acc, width="stretch")
        plt.close(f_train_acc)

    with metric_cols[1]:
        f_train_f1, ax_train_f1 = plt.subplots(figsize=(6.8, 4.2))
        f1_vals = METRICS["F1 (%)"].to_numpy(dtype=float)
        bars = ax_train_f1.bar(x_all, f1_vals)
        ax_train_f1.set_xticks(x_all, [m.replace(" (Proposed)", "\n⭐ RMSCNN") if m == PROPOSED else m.replace("-", "-\n") for m in MODELS], fontsize=8)
        ax_train_f1.set_ylabel("F1 Score (%)")
        ax_train_f1.set_ylim(max(90, float(f1_vals.min()) - 2), 100.6)
        ax_train_f1.set_title("Overall F1 score — all models", pad=10, fontweight="bold")
        ax_train_f1.grid(axis="y", alpha=0.22)
        for b, v in zip(bars, f1_vals):
            ax_train_f1.text(b.get_x() + b.get_width()/2, v + 0.08, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.5)
        f_train_f1.tight_layout()
        st.pyplot(f_train_f1, width="stretch")
        plt.close(f_train_f1)

    st.markdown("#### Training curves available")
    curve_root = repo_root / "results"
    all_pngs = list(curve_root.rglob("*.png")) if curve_root.exists() else []
    all_pngs += list(repo_root.glob("*.png"))
    all_pngs = sorted(set(p.resolve() for p in all_pngs if p.exists()))

    accuracy_plots = [p for p in all_pngs if "accuracy_curve" in p.name.lower()]
    loss_plots = [p for p in all_pngs if "loss_curve" in p.name.lower()]

    def _model_order(paths):
        order = {m.lower().replace(" (proposed)", ""): i for i, m in enumerate(MODELS)}
        def key(p):
            n = p.name.lower()
            hit = 999
            for m, i in order.items():
                if m.replace(" ", "") in n.replace(" ", ""):
                    hit = i
                    break
            return (hit, n)
        return sorted(paths, key=key)

    accuracy_plots = _model_order(accuracy_plots)
    loss_plots = _model_order(loss_plots)

    if accuracy_plots:
        st.markdown("**Accuracy curves — all available models**")
        acc_cols = st.columns(3)
        for i, plot_path in enumerate(accuracy_plots):
            with acc_cols[i % 3]:
                st.image(str(plot_path), width="stretch", caption=plot_path.name)
    else:
        st.info("No per-model accuracy-curve PNGs were found in the project results folders.")

    if loss_plots:
        st.markdown("**Loss curves — all available models**")
        loss_cols = st.columns(3)
        for i, plot_path in enumerate(loss_plots):
            with loss_cols[i % 3]:
                st.image(str(plot_path), width="stretch", caption=plot_path.name)
    else:
        st.info("No per-model loss-curve PNGs were found in the project results folders.")

    st.caption(
        "The overall graphs use the recorded model metrics. The individual training-curve panels are shown only "
        "when the corresponding PNG evidence exists in the project results; no curve is fabricated when training history "
        "data is unavailable."
    )
st.markdown('<div class="foot">Research prototype · Not for clinical diagnosis</div>', unsafe_allow_html=True)