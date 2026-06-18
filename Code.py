import json
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
import streamlit as st


# ============================================================
# MODEL
# ============================================================

class EVA02AircraftClassifier(nn.Module):
    def __init__(self, model_name, num_classes, image_size):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            drop_rate=0.0,
        )

        with torch.no_grad():
            feat_dim = self.backbone(
                torch.randn(1, 3, image_size, image_size)
            ).shape[-1]

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.15),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Aircraft Identifier",
    page_icon="✈️",
    layout="centered",
)

st.title("✈️ Aircraft Identifier")
st.write("Upload a photo of an aircraft. The AI model predicts the aircraft variant.")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="selmamalak/aircraft-eva021",
        filename="aircraft_eva02_finetuned.pth",
    )

    kb_path = hf_hub_download(
        repo_id="selmamalak/aircraft-eva021",
        filename="aircraft_knowledge_base.json",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model = EVA02AircraftClassifier(
        ckpt["model_name"],
        ckpt["num_classes"],
        ckpt["image_size"],
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    with open(kb_path, "r") as f:
        kb = json.load(f)["variants"]

    return model, kb, ckpt["class_names"], ckpt["image_size"], device


MODEL, KB, CLASS_NAMES, IMAGE_SIZE, DEVICE = load_model()


# ============================================================
# TRANSFORM
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def lookup_specs(name):
    if name in KB:
        return KB[name]

    for k in KB:
        if k in name or name in k:
            return KB[k]

    return None


@torch.no_grad()
def classify_pil_image(img):
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    probs = torch.softmax(MODEL(tensor), dim=1)
    top_probs, top_idx = probs.topk(5)

    return top_probs, top_idx


# ============================================================
# APP
# ============================================================

uploaded = st.file_uploader(
    "Upload aircraft image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing aircraft..."):
        top_probs, top_idx = classify_pil_image(image)

    best_variant = CLASS_NAMES[top_idx[0][0].item()]
    confidence = top_probs[0][0].item() * 100

    st.success(f"Prediction: {best_variant}")
    st.metric("Confidence", f"{confidence:.2f}%")

    st.subheader("Top 5 Predictions")

    for i in range(5):
        variant = CLASS_NAMES[top_idx[0][i].item()]
        score = top_probs[0][i].item() * 100

        st.progress(min(score / 100, 1.0))
        st.write(f"{i+1}. {variant} — {score:.2f}%")

    specs = lookup_specs(best_variant)

    if specs:
        st.subheader("Aircraft Information")
        st.json(specs)
