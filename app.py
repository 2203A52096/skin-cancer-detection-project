import streamlit as st
import numpy as np
from PIL import Image, ImageFilter

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
# RULE-BASED ANALYSIS WITHOUT CV2
# -------------------------

def analyze_image_pillow(img):

    # Resize
    img_resized = img.resize((256, 256))

    # Convert to numpy
    arr = np.array(img_resized)

    # Grayscale conversion
    gray = np.mean(arr, axis=2)

    # 1. Darkness
    darkness = gray.mean()

    # 2. Redness (vascular)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    redness = np.mean(r - g)

    # 3. Texture estimation using Laplacian-like filter
    texture_img = img_resized.filter(ImageFilter.FIND_EDGES)
    texture_arr = np.array(texture_img)
    texture = texture_arr.var()

    # 4. Border irregularity — simple pixel difference method
    edges = texture_arr.mean()
    irregularity = edges / 255  # normalized

    # ---------------------------
    # RULES (NO CV2 VERSION)
    # ---------------------------

    # MELANOMA – dark + high texture
    if darkness < 90 and texture > 35000:
        return "Melanoma"

    # VASCULAR – strong redness
    if redness > 25:
        return "Vascular Lesion"

    # ACTINIC KERATOSIS – rough + light
    if texture > 30000 and darkness > 140:
        return "Actinic Keratosis"

    # SQUAMOUS CELL CARCINOMA – rough + irregular
    if texture > 28000 and irregularity > 0.35:
        return "Squamous Cell Carcinoma"

    # BASAL CELL CARCINOMA – brighter center
    center = gray[100:150, 100:150].mean()
    edges_val = (gray[0:50].mean() + gray[-50:].mean()) / 2
    if center > edges_val + 20:
        return "Basal Cell Carcinoma"

    # SEBORRHEIC KERATOSIS – very textured
    if texture > 26000 and darkness < 140:
        return "Seborrheic Keratosis"

    # PIGMENTED BENIGN KERATOSIS – brownish + smoother
    if 100 < darkness < 170 and texture < 20000:
        return "Pigmented Benign Keratosis"

    # DERMATOFIBROMA – smooth + natural color
    if irregularity < 0.20 and 130 < darkness < 200:
        return "Dermatofibroma"

    # NEVUS – smooth, round, medium-dark
    if 80 < darkness < 130:
        return "Nevus"

    return "Nevus"


# -------------------------
# STREAMLIT UI
# -------------------------

st.title("🩺 Skin Cancer Detection (No ML Model Used, No CV2)")
st.write("This app uses smart image processing without a machine learning model.")

uploaded = st.file_uploader("Upload a skin lesion image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image…"):
        result = analyze_image_pillow(img)

    st.success(f"### 🧪 Predicted Condition: **{result}**")

    st.info("""
    **Note:** This is a rule-based analysis, not an ML model prediction.
    For medical decisions, always consult a dermatologist.
    """)

