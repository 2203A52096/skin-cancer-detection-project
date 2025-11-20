import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Skin Cancer Detector", layout="wide")

skin_classes = {
    0: 'Pigmented Benign Keratosis',
    1: 'Melanoma',
    2: 'Vascular Lesion',
    3: 'Actinic Keratosis',
    4: 'Squamous Cell Carcinoma',
    5: 'Basal Cell Carcinoma',
    6: 'Seborrheic Keratosis',
    7: 'Dermatofibroma',
    8: 'Nevus'
}

# -------------------------
# RULE-BASED IMAGE ANALYSIS
# -------------------------

def analyze_image(img):

    # Resize for processing
    img_resized = cv2.resize(img, (256, 256))

    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 1. Color intensity (dark lesions → melanoma / nevus)
    darkness = np.mean(gray)

    # 2. Redness (vascular lesion)
    redness = np.mean(img_resized[:, :, 2]) - np.mean(img_resized[:, :, 1])

    # 3. Texture variance (rough surface → actinic keratosis / SCC)
    texture = np.var(gray)

    # 4. Border irregularity
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    irregularity = 0
    if len(contours) > 0:
        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if area != 0:
            irregularity = (perimeter ** 2) / (4 * np.pi * area)

    # --------------------------------------
    # RULES (Hand-crafted for better accuracy)
    # --------------------------------------

    # MELANOMA – Very dark + irregular border + high texture
    if darkness < 70 and irregularity > 1.6 and texture > 400:
        return "Melanoma"

    # VASCULAR LESION – Strong redness
    if redness > 25:
        return "Vascular Lesion"

    # ACTINIC KERATOSIS – Rough texture + lighter color
    if texture > 380 and darkness > 120:
        return "Actinic Keratosis"

    # SQUAMOUS CELL CARCINOMA – Rough + irregular + medium dark
    if texture > 350 and irregularity > 1.4:
        return "Squamous Cell Carcinoma"

    # BASAL CELL CARCINOMA – Light center, dark edges
    center = gray[100:150, 100:150].mean()
    edges = np.mean([gray[0:50], gray[206:256]])
    if center > edges + 25:
        return "Basal Cell Carcinoma"

    # SEBORRHEIC KERATOSIS – Very textured + brownish
    if texture > 300 and darkness < 120:
        return "Seborrheic Keratosis"

    # PIGMENTED BENIGN KERATOSIS – Brown tint + smooth texture
    if darkness < 140 and texture < 200:
        return "Pigmented Benign Keratosis"

    # DERMATOFIBROMA – Smooth + round + natural skin color
    if irregularity < 1.3 and 120 < darkness < 180:
        return "Dermatofibroma"

    # NEVUS – Smooth, dark, round
    if 80 < darkness < 130 and irregularity < 1.4:
        return "Nevus"

    # Default fallback
    return "Nevus"


# -------------------------
# STREAMLIT UI
# -------------------------

st.title("🩺 Skin Cancer Detection (No ML Model Used)")
st.write("This app analyzes skin lesion images using rule-based image processing.")

uploaded = st.file_uploader("Upload a skin lesion image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image…"):
        result = analyze_image(img_np)

    st.success(f"### 🧪 Predicted Condition: **{result}**")

    st.info("""
    **Note:** This prediction is generated using rule-based image analysis, not a deep learning model.
    For medical decisions, always consult a dermatologist.
    """)

