import os
import json
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

# =============================================================================
# Optional gradio_client patch
# Wird nur angewendet, wenn gradio_client installiert ist.
# =============================================================================
try:
    import gradio_client.utils as gc_utils

    _original_get_type = gc_utils.get_type

    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "bool"
        return _original_get_type(schema)

    gc_utils.get_type = _patched_get_type

    if hasattr(gc_utils, "_json_schema_to_python_type"):
        _original_json_schema = gc_utils._json_schema_to_python_type

        def _patched_json_schema(schema, defs=None):
            if isinstance(schema, bool):
                return "Any"
            return _original_json_schema(schema, defs)

        gc_utils._json_schema_to_python_type = _patched_json_schema

    print("gradio_client patch applied")

except ImportError:
    print("gradio_client not installed - skipping patch")

# =============================================================================

import gradio as gr


class EVA02AircraftClassifier(nn.Module):
    def __init__(self, model_name, num_classes, image_size):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            drop_rate=0.0
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


print("Loading model...")

model_path = hf_hub_download(
    repo_id="selmamalak/aircraft-eva021",
    filename="aircraft_eva02_finetuned.pth"
)

kb_path = hf_hub_download(
    repo_id="selmamalak/aircraft-eva021",
    filename="aircraft_knowledge_base.json"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(
    model_path,
    map_location=DEVICE,
    weights_only=False
)

CLASS_NAMES = ckpt["class_names"]
IMAGE_SIZE = ckpt["image_size"]

MODEL = EVA02AircraftClassifier(
    ckpt["model_name"],
    ckpt["num_classes"],
    IMAGE_SIZE
)

MODEL.load_state_dict(ckpt["model_state_dict"])
MODEL.to(DEVICE)
MODEL.eval()

with open(kb_path, "r") as f:
    KB = json.load(f)["variants"]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

print(f"Model ready! Device: {DEVICE}")


def lookup_specs(name):
    if name in KB:
        return KB[name]

    for k in KB:
        if k in name or name in k:
            return KB[k]

    return None


@torch.no_grad()
def classify(image):
    if image is None:
        return "Please upload an image."

    img = Image.fromarray(image).convert("RGB")

    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    probs = torch.softmax(MODEL(tensor), dim=1)
    top_probs, top_idx = probs.topk(5)

    top_variant = CLASS_NAMES[top_idx[0][0].item()]
    top_conf = top_probs[0][0].item()

    specs = lookup_specs(top_variant)

    if top_conf > 0.7:
        level = "HIGH CONFIDENCE"
    elif top_conf > 0.3:
        level = "MEDIUM CONFIDENCE"
    else:
        level = "LOW CONFIDENCE"

    out = f"=== {level} ===\n\n"
    out += "TOP-5 PREDICTIONS:\n"
    out += "-" * 45 + "\n"

    for i in range(5):
        v = CLASS_NAMES[top_idx[0][i].item()]
        c = top_probs[0][i].item() * 100
        bar = "#" * int(c / 3)
        marker = " <<< BEST" if i == 0 else ""

        out += (
            f"  #{i+1}  {v:25s} "
            f"{c:5.1f}%  {bar}{marker}\n"
        )

    out += "\n" + "=" * 45 + "\n"

    if specs:
        out += f"\nAIRCRAFT: {top_variant}\n"
        out += "-" * 45 + "\n"
        out += f"  Manufacturer : {specs['manufacturer']}\n"
        out += f"  Family       : {specs['family']}\n"
        out += f"  Type         : {specs['type']}\n"
        out += f"  Category     : {specs['category']}\n"
        out += f"  Engines      : {specs['engines']}x {specs['engine_type']}\n"
        out += f"  Range        : {specs['range_km']:,} km\n"
        out += f"  Passengers   : {specs['max_passengers']}\n"
        out += f"  Length       : {specs['length_m']} m\n"
        out += f"  Wingspan     : {specs['wingspan_m']} m\n"
        out += f"  First Flight : {specs['first_flight']}\n"
        out += f"  Military     : {'Yes' if specs['military'] else 'No'}\n"
        out += f"  In Production: {'Yes' if specs['still_in_production'] else 'No'}\n"
        out += f"  Status       : {specs['status']}\n"

    return out


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(
        type="numpy",
        label="Upload Aircraft Photo"
    ),
    outputs=gr.Textbox(
        label="Results",
        lines=25
    ),
    title="Aircraft Identifier",
    description=(
        "AI-powered aircraft classification: "
        "102 variants, 92.35% accuracy. "
        "Upload a photo of any aircraft."
    ),
    flagging_mode="never",
)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)
