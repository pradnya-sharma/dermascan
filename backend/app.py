from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import torch
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO

from dip_model import (
    build_feature_vector,
    dip_preprocess,
    segment_lesion,
    HybridModel,
    GradCAM
)

app = Flask(__name__)
CORS(app)

# ================= LOAD MODELS =================
svm_clf = joblib.load("models/svm_classifier.pkl")
rf_clf  = joblib.load("models/rf_classifier.pkl")

model = HybridModel(num_classes=6)
model.load_state_dict(torch.load("models/hybrid_model.pth", map_location="cpu"))
model.eval()

print(model)

# 👇 IMPORTANT: choose last conv layer of your CNN
target_layer = model.cnn.layer4[-1]   # adjust if needed
gradcam = GradCAM(model, target_layer)

class_names = [
    "Benign","Malignant","Melanoma Invasive",
    "Melanoma in situ","Nevus","Nevus, Dysplastic"
]

# ================= HELPER =================
def to_base64(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# ================= ROUTE =================
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).convert("RGB")

    # ===== DIP FEATURES =====
    fvec, _ = build_feature_vector(img)

    svm_pred = svm_clf.predict([fvec])[0]
    svm_prob = svm_clf.predict_proba([fvec])[0].max()

    rf_pred = rf_clf.predict([fvec])[0]
    rf_prob = rf_clf.predict_proba([fvec])[0].max()

    # ===== PREPROCESS =====
    enhanced, _ = dip_preprocess(img)
    segmented, _, _ = segment_lesion(enhanced)

    rgb = cv2.resize(segmented, (224,224)) / 255.0
    tensor = torch.tensor(rgb.transpose(2,0,1)).unsqueeze(0).float()

    # ===== DL PRED =====
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        dl_idx = torch.argmax(probs).item()
        dl_conf = probs[dl_idx].item()

    # ===== REAL GRAD-CAM =====
    cam = gradcam.generate(tensor, dl_idx)

   

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap * 0.4 + (rgb * 255).astype(np.uint8) * 0.6

    # ===== IMAGES =====
    original_b64 = to_base64(np.array(img))
    segmented_b64 = to_base64((segmented * 255).astype(np.uint8))
    heatmap_b64 = to_base64(overlay)

    # ===== FEATURES =====
    features = {
        "asymmetry_score": float(fvec[0]),
        "compactness": float(fvec[1]),
        "hue_std": float(fvec[2]),
        "lesion_area": float(fvec[3]),

        "area": float(fvec[3]),
        "mean_intensity": float(np.mean(rgb)),
        "std_deviation": float(np.std(rgb)),
        "border_roughness": float(fvec[1]),
        "glcm_contrast": float(fvec[4]) if len(fvec)>4 else 0,
        "fft_energy_ratio": float(fvec[5]) if len(fvec)>5 else 0
    }

    # ===== RESPONSE =====
    return jsonify({
        "is_malignant": "Malignant" in class_names[dl_idx],
        "confidence": float(dl_conf * 100),
        "label": class_names[dl_idx],

        "original_b64": original_b64,
        "segmented_b64": segmented_b64,
        "heatmap_b64": heatmap_b64,

        "features": features,

        "model_comparison": {
            "svm_confidence": float(svm_prob * 100),
            "rf_confidence": float(rf_prob * 100)
        }
    })

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)