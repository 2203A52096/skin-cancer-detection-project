import streamlit as st
from PIL import Image

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Skin Cancer Guide",
    layout="wide",
    page_icon="🩺"
)

# ---------------------- SIDEBAR DARK MODE TOGGLE ----------------------
st.sidebar.markdown("## 🌓 Theme")
dark_mode = st.sidebar.checkbox("Enable Dark Mode")

# ---------------------- THEME CSS ----------------------
if dark_mode:
    # 🌙 DARK MODE CSS
    st.markdown("""
    <style>
    body { background-color: #0E1117; }
    .main-title { color: #FF5F5F; }
    .sub-title { color: #CCCCCC; }
    .info-card, .result-box, .upload-box {
        background: #1E1E1E !important;
        color: white !important;
        border: 1px solid #444;
    }
    .menu-box { background: #1E1E1E; color: white; border:1px solid #444; }
    .menu-box:hover { background: #292929; border-color: #FF5F5F; }
    </style>
    """, unsafe_allow_html=True)

else:
    # ☀️ LIGHT MODE CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    body {
        font-family: 'Poppins', sans-serif;
        background: #FAFAFA;
    }
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        color: #ff4b4b;
        margin-bottom: 10px;
    }

    .sub-title {
        font-size: 20px;
        font-weight: 400;
        text-align: center;
        margin-bottom: 30px;
        color: #444444;
    }

    .menu-box {
        background: white;
        padding: 18px;
        border-radius: 18px;
        margin-bottom: 10px;
        font-size: 17px;
        border: 1px solid #e6e6e6;
        transition: all 0.2s;
    }
    .menu-box:hover {
        background: #ffecec;
        border-color: #ff6b6b;
    }

    .upload-box {
        background: #ffffff;
        padding: 30px;
        border-radius: 20px;
        border: 2px dashed #ff6b6b;
        text-align: center;
    }

    .result-box {
        background: #fff5f5;
        padding: 25px;
        border-radius: 20px;
        border-left: 6px solid #ff4b4b;
        margin-top: 25px;
    }

    .info-card {
        background: #ffffff;
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #dddddd;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------- RULE-BASED PREDICTOR ----------------------
SKIN_CLASSES = {
    0: "Pigmented Benign Keratosis",
    1: "Melanoma",
    2: "Vascular Lesion",
    3: "Actinic Keratosis",
    4: "Squamous Cell Carcinoma",
    5: "Basal Cell Carcinoma",
    6: "Seborrheic Keratosis",
    7: "Dermatofibroma",
    8: "Nevus"
}

def simple_rule_predict(image):
    img = image.convert("RGB")
    pixels = img.getdata()

    r = sum([p[0] for p in pixels]) / len(pixels)
    g = sum([p[1] for p in pixels]) / len(pixels)
    b = sum([p[2] for p in pixels]) / len(pixels)

    if r > g and r > b:
        return SKIN_CLASSES[1]      # Melanoma
    elif b > r and b > g:
        return SKIN_CLASSES[2]      # Vascular Lesion
    elif g > r and g > b:
        return SKIN_CLASSES[8]      # Nevus
    else:
        return SKIN_CLASSES[5]      # Basal Cell Carcinoma

# ---------------------- SIDEBAR NAVIGATION ----------------------
page = st.sidebar.radio(
    "📌 Navigate",
    [
        "🏠 Home",
        "📤 Upload & Predict",
        "💊 Treatment Plan",
        "👨‍⚕️ Doctor’s Advice",
        "ℹ️ About"
    ]
)

# ---------------------- HOME PAGE ----------------------
if page == "🏠 Home":
    st.markdown("<div class='main-title'>🩺 Skin Cancer Companion</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Your friendly assistant to understand skin conditions.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🌟 What This App Offers</h3>
            <p>This guide helps you understand skin cancer types and next steps.</p>
            <ul>
                <li>Upload & analyze skin images</li>
                <li>Instant classification</li>
                <li>Treatment suggestions</li>
                <li>Doctor-style advice</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image("https://i.imgur.com/oYiTqum.png", use_container_width=True)

# ---------------------- UPLOAD PAGE ----------------------
elif page == "📤 Upload & Predict":
    st.markdown("<div class='main-title'>📤 Upload & Predict</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)

        st.markdown("### 🖼 Uploaded Image:")
        st.image(img, width=350)

        predicted = simple_rule_predict(img)

        st.markdown(f"""
        <div class='result-box'>
            <h3>🔍 Predicted Skin Cancer Type:</h3>
            <h2 style='color:#ff4b4b'>{predicted}</h2>
        </div>
        """, unsafe_allow_html=True)

# ---------------------- TREATMENT PLAN ----------------------
elif page == "💊 Treatment Plan":
    st.markdown("<div class='main-title'>💊 Treatment Plans</div>", unsafe_allow_html=True)

    choice = st.selectbox("Choose a skin cancer type", list(SKIN_CLASSES.values()))

    treatment = {
        "Melanoma": "Surgery, immunotherapy, targeted therapy.",
        "Basal Cell Carcinoma": "Excision, cryotherapy, topical chemotherapy.",
        "Squamous Cell Carcinoma": "Surgery, radiation therapy.",
        "Nevus": "Usually harmless; monitor or remove.",
        "Pigmented Benign Keratosis": "Cryotherapy or surface removal.",
        "Seborrheic Keratosis": "Laser therapy, freezing.",
        "Vascular Lesion": "Laser treatment.",
        "Actinic Keratosis": "Cryotherapy and topical drugs.",
        "Dermatofibroma": "Harmless; remove if painful."
    }

    st.markdown(f"""
    <div class='info-card'>
        <h3>💡 Treatment for {choice}</h3>
        <p>{treatment.get(choice)}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------- DOCTOR'S ADVICE ----------------------
elif page == "👨‍⚕️ Doctor’s Advice":
    st.markdown("<div class='main-title'>👨‍⚕️ Doctor’s Advice</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
        <ul>
            <li>Watch for sudden size or color changes.</li>
            <li>Avoid intense sun exposure (12–4 PM).</li>
            <li>Use SPF 30+ every day.</li>
            <li>Do not scratch or pick lesions.</li>
            <li>Seek a dermatologist if changes persist.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------------- ABOUT PAGE ----------------------
elif page == "ℹ️ About":
    st.markdown("<div class='main-title'>ℹ️ About This App</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
        <h3>📌 Purpose</h3>
        <p>This application is built to help users understand skin cancer types and guide them with treatment options and general advice.</p>

        <h3>👩‍💻 Developer</h3>
        <p>Developed by a passionate ML enthusiast for healthcare awareness and guidance.</p>

        <h3>⚠️ Disclaimer</h3>
        <p>This tool is for informational use only. Always consult a certified dermatologist for diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
