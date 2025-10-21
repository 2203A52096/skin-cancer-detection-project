import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Safe Skin", page_icon="🩺", layout="wide")

# ---------------- CUSTOM STYLES ----------------
st.markdown("""
<style>
    /* Page Background and Font */
    body {
        background-color: #F8FAFC;
        color: #1E293B;
    }
    /* Title Styling */
    .main-title {
        text-align: center;
        color: #D6336C;
        font-weight: 800;
        font-size: 45px;
    }
    .tagline {
        text-align: center;
        color: #475569;
        font-style: italic;
        font-size: 20px;
    }
    /* Section Headers */
    h2, h3 {
        color: #0F172A;
        font-weight: 700;
    }
    /* Highlight text */
    .highlight {
        color: #D6336C;
        font-weight: 700;
    }
    /* Navigation Buttons */
    div[data-testid="column"] button {
        width: 100%;
        height: 3.2em;
        border-radius: 12px;
        font-weight: bold;
        color: white;
        border: none;
        background: linear-gradient(90deg, #D6336C, #9333EA);
    }
    div[data-testid="column"] button:hover {
        background: linear-gradient(90deg, #9333EA, #D6336C);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = "skincancercnn.h5"

    if not os.path.exists(model_path):
        st.error("❌ <b>Model file not found!</b> Please ensure 'skincancercnn.h5' is in the same folder as app.py", unsafe_allow_html=True)
        return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ <b>Model loaded successfully!</b>", icon="✅")
        return model
    except Exception as e:
        st.error(f"⚠️ <b>Error loading model:</b> {e}", unsafe_allow_html=True)
        return None


# Load model once
model = load_model()

# ---------------- NAVIGATION BAR ----------------
st.markdown("<h1 class='main-title'>🩺 Safe Skin</h1>", unsafe_allow_html=True)
st.markdown("<p class='tagline'>AI-powered Skin Cancer Detection & Support System</p>", unsafe_allow_html=True)
st.write("---")

col1, col2, col3 = st.columns(3)
with col1:
    home_btn = st.button("🏠 Home")
with col2:
    pred_btn = st.button("🔬 Prediction")
with col3:
    sol_btn = st.button("💊 Solution")

# Determine active page
if home_btn or (not pred_btn and not sol_btn):
    page = "home"
elif pred_btn:
    page = "prediction"
elif sol_btn:
    page = "solution"
else:
    page = "home"

# ---------------- HOME PAGE ----------------
if page == "home":
    st.markdown("<h2>🌟 Project Overview</h2>", unsafe_allow_html=True)
    st.write("""
    **<span class='highlight'>Safe Skin</span>** is an AI-driven system built to **detect and classify skin cancer** from dermoscopic images.  
    Early detection is critical for improving survival rates — and this tool supports both **dermatologists** and **patients** through quick, reliable analysis.
    """, unsafe_allow_html=True)

    st.markdown("<h2>💡 Features</h2>", unsafe_allow_html=True)
    st.write("""
    - Upload dermoscopic skin images for **AI-based lesion classification**.  
    - Get **instant predictions** with **confidence scores**.  
    - Receive **treatment suggestions** and **estimated recovery times**.  
    - Designed for both **medical professionals** and **self-screening users**.
    """)

    st.markdown("<h2>🎯 Goals</h2>", unsafe_allow_html=True)
    st.write("""
    - Promote **early detection** of skin cancer.  
    - Empower dermatologists with **AI-assisted diagnostics**.  
    - Offer **accessible care** to rural and remote communities.  
    """)

    st.markdown("<h2>🚀 Advantages</h2>", unsafe_allow_html=True)
    st.write("""
    - Improves **accuracy** and **speed** of skin cancer detection.  
    - Minimizes **human error** in visual diagnosis.  
    - Can be integrated with **tele-dermatology and mobile health apps**.  
    - Incorporates **Grad-CAM** for **visual explanations** of AI predictions.
    """)

# ---------------- PREDICTION PAGE ----------------
elif page == "prediction":
    st.markdown("<h2>🔬 Skin Lesion Prediction</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload a dermoscopic image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='📸 Uploaded Image', use_column_width=True, output_format="JPEG")

        if st.button("🔍 Predict"):
            st.info("🧠 Analyzing your image...")
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            class_names = [
                "Melanocytic nevi (Benign)",
                "Melanoma (Malignant)",
                "Benign keratosis (Benign)",
                "Basal cell carcinoma (Malignant)",
                "Actinic keratoses (Precancerous)",
                "Vascular lesions (Benign)",
                "Dermatofibroma (Benign)"
            ]

            predicted_class = class_names[np.argmax(preds)]
            confidence = np.max(preds)

            st.success(f"### ✅ Prediction: <span class='highlight'>{predicted_class}</span>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {confidence * 100:.2f}%")

# ---------------- SOLUTION PAGE ----------------
elif page == "solution":
    st.markdown("<h2>💊 Treatment & Recovery Plan</h2>", unsafe_allow_html=True)
    st.write("Select the **type of skin cancer** to get recovery duration and treatment advice:")

    cancer_type = st.selectbox(
        "Select Skin Cancer Type",
        [
            "Melanocytic nevi (Benign)",
            "Melanoma (Malignant)",
            "Benign keratosis (Benign)",
            "Basal cell carcinoma (Malignant)",
            "Actinic keratoses (Precancerous)",
            "Vascular lesions (Benign)",
            "Dermatofibroma (Benign)"
        ]
    )

    if cancer_type:
        st.markdown(f"### 🧠 Selected: <span class='highlight'>{cancer_type}</span>", unsafe_allow_html=True)
        st.write("---")

        if "Melanoma" in cancer_type:
            st.markdown("⏳ **Expected Recovery:** <span class='highlight'>4–6 months</span>", unsafe_allow_html=True)
            st.write("""
            **Medication Plan:**
            - Surgical removal of affected tissue  
            - Targeted or immunotherapy for advanced cases  
            - Frequent skin checkups (every 3–6 months)  
            - Balanced diet, hydration, and sun protection
            """)

        elif "Basal cell carcinoma" in cancer_type:
            st.markdown("⏳ **Expected Recovery:** <span class='highlight'>2–4 months</span>", unsafe_allow_html=True)
            st.write("""
            **Medication Plan:**
            - Excision or Mohs surgery for lesion removal  
            - Topical creams (Imiquimod, Fluorouracil)  
            - Laser therapy for smaller lesions  
            - Regular dermatology follow-ups
            """)

        elif "Actinic keratoses" in cancer_type:
            st.markdown("⏳ **Expected Recovery:** <span class='highlight'>1–3 months</span>", unsafe_allow_html=True)
            st.write("""
            **Medication Plan:**
            - Cryotherapy (freezing the lesion)  
            - Topical chemotherapy  
            - Strict sun avoidance and sunscreen  
            """)

        elif "Benign" in cancer_type or "Vascular" in cancer_type:
            st.markdown("⏳ **Expected Recovery:** <span class='highlight'>2–4 weeks</span>", unsafe_allow_html=True)
            st.write("""
            **Medication Plan:**
            - Usually harmless; monitor for visual changes  
            - If irritated, minor surgical removal or laser  
            - Maintain healthy skin hygiene  
            """)

        elif "Dermatofibroma" in cancer_type:
            st.markdown("⏳ **Expected Recovery:** <span class='highlight'>3–5 weeks</span>", unsafe_allow_html=True)
            st.write("""
            **Medication Plan:**
            - Generally non-cancerous, removal if bothersome  
            - Use mild creams for irritation relief  
            - Avoid friction and protect the area  
            """)
