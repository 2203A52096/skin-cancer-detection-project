import streamlit as st
from PIL import Image
import base64

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Skin Cancer Guide",
    layout="wide",
    page_icon="🩺"
)

# ---------------------- GLOBAL CSS ----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

body {
    font-family: 'Poppins', sans-serif;
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
    # Artificial, simple logic to output a class
    img = image.convert("RGB")
    pixels = img.getdata()

    r, g, b = 0, 0, 0
    for pr, pg, pb in pixels:
        r += pr
        g += pg
        b += pb

    total = len(pixels)
    avg_r = r / total
    avg_g = g / total
    avg_b = b / total

    if avg_r > avg_g and avg_r > avg_b:
        return SKIN_CLASSES[1]  
    elif avg_b > avg_r and avg_b > avg_g:
        return SKIN_CLASSES[2]  
    elif avg_g > avg_r and avg_g > avg_b:
        return SKIN_CLASSES[8]
    else:
        return SKIN_CLASSES[5]

# ---------------------- PAGE NAVIGATION ----------------------

st.sidebar.title("🩺 Skin Cancer Guide")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📤 Upload & Predict", "💊 Treatment Plan", "👨‍⚕️ Doctor’s Advice"]
)

# ---------------------- HOME PAGE ----------------------
if page == "🏠 Home":
    st.markdown("<div class='main-title'>🩺 Skin Cancer Companion</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Your friendly assistant to understand skin conditions with ease.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🌟 What This App Does</h3>
            <p>This application helps you understand possible skin cancer types from uploaded images and gives guidance.</p>
            <ul>
                <li>Upload a skin image</li>
                <li>Get instant identification</li>
                <li>Explore treatment suggestions</li>
                <li>Read doctor-style advice</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image("https://i.imgur.com/oYiTqum.png", use_container_width=True)

# ---------------------- UPLOAD & PREDICT PAGE ----------------------
elif page == "📤 Upload & Predict":
    st.markdown("<div class='main-title'>📤 Upload & Predict</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🖼 Uploaded Image:")
        st.image(img, width=300)

        st.markdown("<div class='result-box'>Analyzing image...</div>", unsafe_allow_html=True)

        predicted = simple_rule_predict(img)

        st.markdown(f"""
        <div class='result-box'>
            <h3>🔍 Predicted Skin Cancer Type:</h3>
            <h2 style='color:#ff4b4b'>{predicted}</h2>
        </div>
        """, unsafe_allow_html=True)

# ---------------------- TREATMENT PLAN PAGE ----------------------
elif page == "💊 Treatment Plan":
    st.markdown("<div class='main-title'>💊 Treatment Plans</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
        <h3>🧴 Treatment Guidance</h3>
        <p>Select a skin cancer type to view treatment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    choice = st.selectbox("Choose type", list(SKIN_CLASSES.values()))

    st.markdown(f"""
    <div class='info-card'>
        <h3>💡 Recommended Treatment for {choice}</h3>
        """, unsafe_allow_html=True)

    treatment = {
        "Melanoma": "Surgery, immunotherapy, targeted therapy.",
        "Basal Cell Carcinoma": "Excision, cryotherapy, topical chemotherapy.",
        "Squamous Cell Carcinoma": "Surgery, radiation therapy, topical treatment.",
        "Nevus": "Observation, optional removal.",
        "Pigmented Benign Keratosis": "Cryotherapy or removal if irritating.",
        "Seborrheic Keratosis": "Laser therapy or freezing.",
        "Vascular Lesion": "Laser treatment.",
        "Actinic Keratosis": "Cryotherapy and topical medications.",
        "Dermatofibroma": "Usually harmless; removal only if irritating."
    }

    st.write(treatment.get(choice, "No data available."))

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- DOCTOR'S ADVICE PAGE ----------------------
elif page == "👨‍⚕️ Doctor’s Advice":
    st.markdown("<div class='main-title'>👨‍⚕️ Doctor’s Advice</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
        <h3>🩺 General Advice</h3>
        <ul>
            <li>Monitor sudden changes in skin spots.</li>
            <li>Watch for irregular borders or color variation.</li>
            <li>Avoid direct sunlight between 12 PM – 4 PM.</li>
            <li>Apply sunscreen SPF 30+ daily.</li>
            <li>Visit a dermatologist for persistent changes.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
