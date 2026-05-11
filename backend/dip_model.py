import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import timm
import torch.nn.functional as F
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()

        grads = self.gradients[0]
        acts = self.activations[0]

        weights = torch.mean(grads, dim=(1, 2))

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam)
        cam = cam.detach().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


# ==============================
# 🔹 HAIR REMOVAL
# ==============================
def remove_hair(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    img_no_hair = cv2.inpaint(img, hair_mask, 3, cv2.INPAINT_TELEA)
    return img_no_hair, hair_mask


# ==============================
# 🔹 DIP PREPROCESSING
# ==============================
def dip_preprocess(img_input):
    if isinstance(img_input, Image.Image):
        img = np.array(img_input.convert("RGB"))
    else:
        img = img_input.copy()

    img, hair_mask = remove_hair(img)
    img = cv2.medianBlur(img, 5)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    return img, hair_mask


# ==============================
# 🔹 SEGMENTATION
# ==============================
def segment_lesion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.uint8(labels == largest) * 255
    else:
        mask = thresh

    edges = cv2.Canny(mask, 100, 200)
    segmented = cv2.bitwise_and(img, img, mask=mask)

    return segmented, mask, edges


# ==============================
# 🔹 ABCD FEATURES
# ==============================
def extract_abcd_features(img, mask):
    features = {}

    flipped_h = cv2.flip(mask, 0)
    flipped_v = cv2.flip(mask, 1)

    def asymmetry_score(m1, m2):
        union = cv2.bitwise_or(m1, m2)
        diff  = cv2.bitwise_xor(m1, m2)
        return float(np.sum(diff) / (np.sum(union) + 1e-6))

    features["asymmetry_h"] = asymmetry_score(mask, flipped_h)
    features["asymmetry_v"] = asymmetry_score(mask, flipped_v)
    features["asymmetry_score"] = (features["asymmetry_h"] + features["asymmetry_v"]) / 2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        features["compactness"] = (perimeter**2) / (4 * np.pi * area + 1e-6)

        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)
        features["border_roughness"] = perimeter / (hull_perimeter + 1e-6)

        features["lesion_area"] = area
        features["lesion_perimeter"] = perimeter

        x, y, w, h = cv2.boundingRect(contour)
        features["bounding_box_ratio"] = w / (h + 1e-6)
    else:
        for k in ["compactness","border_roughness","lesion_area","lesion_perimeter","bounding_box_ratio"]:
            features[k] = 0.0

    lesion_pixels = img[mask > 0]

    if len(lesion_pixels) > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_pixels = hsv[mask > 0]

        features["color_mean_r"] = float(np.mean(lesion_pixels[:, 0]))
        features["color_std_r"] = float(np.std(lesion_pixels[:, 0]))
        features["color_std_g"] = float(np.std(lesion_pixels[:, 1]))
        features["color_std_b"] = float(np.std(lesion_pixels[:, 2]))
        features["hue_std"] = float(np.std(hsv_pixels[:, 0]))
        features["saturation_mean"] = float(np.mean(hsv_pixels[:, 1]))
    else:
        for k in ["color_mean_r","color_std_r","color_std_g","color_std_b","hue_std","saturation_mean"]:
            features[k] = 0.0

    return features

# ==============================
# 🔹 TEXTURE FEATURES
# ==============================
def extract_texture_features(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lesion_gray = gray.copy()
    lesion_gray[mask == 0] = 0

    glcm = graycomatrix(
        lesion_gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = {
    "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
    "glcm_dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
    "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
    "glcm_energy": float(graycoprops(glcm, "energy").mean()),
    "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
}

    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp[mask > 0], bins=10, range=(0, 10), density=True)

    for i, val in enumerate(hist):
        features[f"lbp_{i}"] = float(val)

    return features


# ==============================
# 🔹 FFT FEATURES
# ==============================
def extract_frequency_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4

    low_mask = np.zeros((h, w))
    high_mask = np.ones((h, w))

    cv2.circle(low_mask, (cx, cy), radius, 1, -1)
    cv2.circle(high_mask, (cx, cy), radius, 0, -1)

    low_energy = np.sum(magnitude * low_mask)
    high_energy = np.sum(magnitude * high_mask)

    return {
    "fft_low_energy": float(low_energy),
    "fft_high_energy": float(high_energy),
    "fft_energy_ratio": float(high_energy / (low_energy + 1e-6)),
    "fft_mean": float(magnitude.mean()),
    "fft_std": float(magnitude.std()),
}


# ==============================
# 🔹 FINAL FEATURE VECTOR
# ==============================
def build_feature_vector(img_input):
    enhanced, _ = dip_preprocess(img_input)
    _, mask, _ = segment_lesion(enhanced)

    f1 = extract_abcd_features(enhanced, mask)
    f2 = extract_texture_features(enhanced, mask)
    f3 = extract_frequency_features(enhanced)

    all_features = {**f1, **f2, **f3}

    values = np.array(list(all_features.values()), dtype=np.float32)
    values = np.nan_to_num(values)

    return values, list(all_features.keys())


# ==============================
# 🔹 HYBRID MODEL
# ==============================
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Identity()

        self.cnn_proj = nn.Linear(512, 256)
        self.vit_proj = nn.Linear(768, 256)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit.forward_features(x).mean(dim=1)

        fused = (self.cnn_proj(cnn_feat) + self.vit_proj(vit_feat)) / 2
        fused = self.dropout(fused)

        return self.classifier(fused)