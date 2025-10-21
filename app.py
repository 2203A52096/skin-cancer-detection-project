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
/* Page Background */
body {
    background-color: #FFFDF5;
    color: #1E293B;
}

/* Main title */
.main-title {
    text-align: center;
    color: #D6336C;
    font-weight: 900;
    font-size: 50px;
    margin: 10px 0;
}

/* Tagline */
.tagline {
    text-align: center;
    color: #475569;
    font-style: italic;
    font-size: 22px;
    margin-top: 0px;
}

/* Section headers */
h2, h3 {
    color: #0F172A;
    font-weight: 700;
    padding-top: 10px;
}

/* Highlight important words as colorful badges */
.highlight {
    display: inline-block;
    padding: 0.25em 0.6em;
    border-radius: 0.5em;
    background: linear-gradient(90deg, #FFD166, #EF476F);
    color: white;
    font-weight: 700;
    margin: 2px;
}

/* Confidence output styling */
.confidence {
    color: #118AB2;
    font-weight: 700;
    font-size: 20px;
    background-color: #E0F7FA;
    padding: 0.3em 0.6em;
    border-radius: 0.4em;
    display: inline-block;
    margin-top: 5px;
}

/* Add padding to sections */
section .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Prediction button gradient */
div[data-testid="stForm"] button {
    width: 100%;
    height: 3em;
    border-radius: 12px;
    font-weight: bold;
    color: white;
    border: none;
    background: linear-gradient(90deg, #06D6A0, #118AB2);
}
div[data-testid="stForm"] button:hover {
    background: linear-gradient(90deg, #118AB2, #06D6A0);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = "skincancercnn.h5"

    if not os.path.exists(model_path):
        st.info("❌ Model file not found! Place 'skincancercnn.h5' in the same folder.", icon="⚠️")
        return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully!", icon="🎉")
        return model
    except Exception as e:
        st.warning(f"⚠️ Error loading model: {e}")
        return None

# Load model
model = load_model()

# ---------------- SIDEBAR NAVIGATION ----------------
page = st.sidebar.selectbox(
    "Navigate Pages",
    ["🏠 Home", "🔬 Prediction", "💊 Solution"],
    format_func=lambda x: x,
    key="nav",
    help="Select the page you want to view"
)

# ---------------- HOME PAGE ----------------
if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>🩺 Safe Skin</h1>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>AI-powered Skin Cancer Detection & Support System</p>", unsafe_allow_html=True)
    st.write("---")

    st.markdown("<h2>🌟 Features</h2>", unsafe_allow_html=True)
    st.markdown("""
    - Upload <span class='highlight'>dermoscopic skin images</span> for AI-based lesion classification.  
    - Get <span class='highlight'>instant predictions</span> with confidence scores.  
    - Access <span class='highlight'>treatment suggestions</span> and <span class='highlight'>estimated recovery times</span>.  
    - Designed for <span class='highlight'>medical professionals</span> and <span class='highlight'>self-screening users</span>.
    """, unsafe_allow_html=True)

    st.markdown("<h2>🎯 Goals</h2>", unsafe_allow_html=True)
    st.markdown("""
    - Promote <span class='highlight'>early detection</span> of skin cancer.  
    - Empower dermatologists with <span class='highlight'>AI-assisted diagnostics</span>.  
    - Provide <span class='highlight'>accessible care</span> in rural and remote areas.
    """, unsafe_allow_html=True)

    st.markdown("<h2>🚀 Advantages</h2>", unsafe_allow_html=True)
    st.markdown("""
    - Improves <span class='highlight'>accuracy</span> and <span class='highlight'>speed</span> of diagnosis.  
    - Reduces <span class='highlight'>human error</span> in lesion interpretation.  
    - Integrates with <span class='highlight'>tele-dermatology</span> and <span class='highlight'>mobile apps</span>.  
    - Includes <span class='highlight'>Grad-CAM visualizations</span> for AI decision transparency.
    """, unsafe_allow_html=True)

# ---------------- PREDICTION PAGE ----------------
elif page == "🔬 Prediction":
    st.markdown("<h2>🔬 Skin Lesion Prediction</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload a dermoscopic image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='📸 Uploaded Image', use_column_width=True)

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

            st.markdown(f"### ✅ Prediction: <span class='highlight'>{predicted_class}</span>", unsafe_allow_html=True)
            st.markdown(f"<p class='confidence'>Confidence Score: {confidence*100:.2f}%</p>", unsafe_allow_html=True)

# ---------------- SOLUTION PAGE ----------------
elif page == "💊 Solution":
    st.markdown("<h2>💊 Treatment & Recovery Plan</h2>", unsafe_allow_html=True)
    st.write("Select the **type of skin cancer** to view recommended recovery duration and medication plan:")

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

        # Plan mapping
        plans = {
            "Melanoma": ("4–6 months", """
- Surgical removal of affected tissue
- Targeted therapy or immunotherapy
- Regular skin checkups (3–6 months)
- Balanced diet, hydration, sun protection
"""),
            "Basal cell carcinoma": ("2–4 months", """
- Excision or Mohs surgery
- Topical creams (Imiquimod, Fluorouracil)
- Laser therapy for small lesions
- Regular follow-up
"""),
            "Actinic keratoses": ("1–3 months", """
- Cryotherapy
- Topical chemotherapy
- Sun protection and sunscreen
"""),
            "Benign": ("2–4 weeks", """
- Usually harmless, monitor for changes
- Minor removal if irritated
- Maintain good skin hygiene
"""),
            "Vascular lesions": ("2–4 weeks", """
- Typically harmless
- Laser therapy if needed
- Monitor for changes
"""),
            "Dermatofibroma": ("3–5 weeks", """
- Usually harmless, removal if bothersome
- Soothing creams for irritation
- Protect from friction
""")
        }

        # Determine plan
        recovery, med_plan = None, None
        for key in plans.keys():
            if key in cancer_type:
                recovery, med_plan = plans[key]
                break

        if recovery and med_plan:
            st.markdown(f"⏳ **Expected Recovery:** <span class='highlight'>{recovery}</span>", unsafe_allow_html=True)
            st.markdown("**Medication Plan:**")
            st.write(med_plan)
