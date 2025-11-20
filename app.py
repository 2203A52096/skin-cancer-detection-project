import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Safe Skin", page_icon="🩺", layout="wide")

# ---------------- CUSTOM STYLES ----------------
st.markdown("""
<style>
/* Page Background Gradient */
body {
    background: linear-gradient(to right, #f9f9f9, #e0f7fa);
    color: #1E293B;
}

/* Main title - dark */
.main-title {
    text-align: center;
    font-weight: 900;
    font-size: 50px;
    margin: 10px 0;
    color: #1E293B;
}

/* Tagline - dark */
.tagline {
    text-align: center;
    font-size: 22px;
    font-style: italic;
    margin-bottom: 30px;
    color: #1E293B;
}

/* Section headers - dark */
h2 {
    font-weight: 700;
    color: #1E293B;
    padding-top: 10px;
}

/* Highlighted words */
.highlight {
    display: inline-block;
    padding: 0.3em 0.7em;
    border-radius: 0.5em;
    background: linear-gradient(90deg, #FFD93D, #6BCB77);
    color: #1E293B;
    font-weight: 700;
    margin: 2px 2px;
}

/* Card sections */
.card {
    background-color: #ffffffcc;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Buttons Gradient */
div[data-testid="stForm"] button {
    width: 100%;
    height: 3em;
    border-radius: 12px;
    font-weight: bold;
    color: white;
    border: none;
    background: linear-gradient(90deg, #06D6A0, #118AB2);
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource

from tensorflow.keras.models import load_model

model=load_model("skincancercnn.h5")

if model is None:
    st.warning("❌ Model file not found or failed to load. Place 'skincancercnn.h5' in the app folder.", icon="⚠️")

# ---------------- NAVIGATION ----------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Home"

st.sidebar.markdown("## 📂 Navigation")

# Use radio for single selection
pages = ["🏠 Home", "🔬 Prediction", "💊 Solution"]
# make sure index is valid
default_index = pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0
page = st.sidebar.radio("", pages, index=default_index)

# Save selection to session_state
st.session_state.current_page = page

# CSS to make sidebar radio buttons light green
st.markdown("""
    <style>
    .stRadio > div > label > div {
        background-color: #C6F6D5; /* light green */
        border-radius: 10px;
        padding: 8px;
        margin-bottom: 5px;
        font-weight: bold;
        color: #1E293B;
    }
    .stRadio > div > label > div:hover {
        background-color: #A3E4B0; /* slightly darker on hover */
    }
    </style>
""", unsafe_allow_html=True)


# ---------------- HOME PAGE ----------------
if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>🩺 Safe Skin</h1>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>AI-powered Skin Cancer Detection & Support System</p>", unsafe_allow_html=True)

    st.markdown('<div class="card"><h2>🌟 Features</h2><ul>'
                '<li>Upload <span class="highlight">dermoscopic images</span> for AI-based classification.</li>'
                '<li>Get <span class="highlight">instant predictions</span>.</li>'
                '<li>Access <span class="highlight">treatment suggestions</span> & <span class="highlight">recovery times</span>.</li>'
                '<li>Designed for <span class="highlight">medical professionals</span> & <span class="highlight">self-screening users</span>.</li>'
                '</ul></div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h2>🎯 Goals</h2><ul>'
                '<li>Promote <span class="highlight">early detection</span>.</li>'
                '<li>Empower dermatologists with <span class="highlight">AI-assisted diagnostics</span>.</li>'
                '<li>Provide <span class="highlight">accessible care</span> in remote areas.</li>'
                '</ul></div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h2>🚀 Advantages</h2><ul>'
                '<li>Improves <span class="highlight">accuracy</span> & <span class="highlight">speed</span> of diagnosis.</li>'
                '<li>Reduces <span class="highlight">human error</span>.</li>'
                '<li>Integrates with <span class="highlight">tele-dermatology</span> & <span class="highlight">mobile apps</span>.</li>'
                '<li>Includes <span class="highlight">Grad-CAM visualizations</span>.</li>'
                '</ul></div>', unsafe_allow_html=True)

# ---------------- PREDICTION PAGE ----------------
elif page == "🔬 Prediction":
    st.markdown("<h2>🔬 Skin Lesion Prediction</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload a dermoscopic image", type=["jpg", "jpeg", "png"])

    class_names = [
        "Melanocytic nevi (Benign)",
        "Melanoma (Malignant)",
        "Benign keratosis (Benign)",
        "Basal cell carcinoma (Malignant)",
        "Actinic keratoses (Precancerous)",
        "Vascular lesions (Benign)",
        "Dermatofibroma (Benign)"
    ]

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Unable to open image: {e}")
            image = None

        if image is not None:
            # Convert to RGB to avoid issues with RGBA or palette images
            image_rgb = image.convert("RGB")
            st.image(image_rgb, caption='📸 Uploaded Image', use_column_width=True)

            # Disable predict button if model not loaded
            if model is None:
                st.error("Model not available. Put 'skincancercnn.h5' in the app folder and restart the app.")
            else:
                if st.button("🔍 Predict"):
                    st.info("🧠 Analyzing your image...")

                    # Preprocess image safely
                    img = image_rgb.resize((224, 224))
                    img_array = np.array(img).astype("float32") / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Get predictions and ensure probabilities
                    raw_preds = model.predict(img_array)[0]
                    # If model outputs logits, convert via softmax
                    if not np.isclose(np.sum(raw_preds), 1.0):
                        preds = tf.nn.softmax(raw_preds).numpy()
                    else:
                        preds = raw_preds

                    # Defensive: ensure preds length matches class_names
                    if preds.shape[0] != len(class_names):
                        st.error(f"Model output length ({preds.shape[0]}) does not match number of classes ({len(class_names)}).")
                    else:
                        predicted_class = class_names[np.argmax(preds)]
                        st.markdown(f"### ✅ Prediction: <span class='highlight'>{predicted_class}</span>", unsafe_allow_html=True)

                        fig = go.Figure(go.Bar(
                            x=(preds * 100),
                            y=class_names,
                            orientation='h',
                            marker=dict(
                                color=['#FFDAB9', '#FFB6C1', '#F0E68C', '#B0E0E6', '#90EE90', '#D8BFD8', '#FFDEAD'],
                                line=dict(color='rgb(248, 248, 249)', width=1)
                            )
                        ))
                        fig.update_layout(
                            title='Class Probabilities (%)',
                            xaxis=dict(title='Probability (%)'),
                            yaxis=dict(autorange="reversed"),
                            height=450,
                            plot_bgcolor='#f9f9f9',
                            paper_bgcolor='#f9f9f9'
                        )
                        st.plotly_chart(fig, use_container_width=True)

# ---------------- SOLUTION PAGE ----------------
elif page == "💊 Solution":
    st.markdown("<h2>💊 Treatment & Recovery Plan</h2>", unsafe_allow_html=True)
    st.write("Select the **type of skin cancer** to view recommended recovery duration and medication plan:")

    cancer_type = st.selectbox(
        "Select Skin Cancer Type",
        class_names
    )

    if cancer_type:
        st.markdown(f"### 🧠 Selected: <span class='highlight'>{cancer_type}</span>", unsafe_allow_html=True)
        st.write("---")

        plans = {
            "Melanoma": ("4–6 months", ["Surgical removal", "Targeted therapy/immunotherapy", "Regular checkups", "Sun protection & diet"]),
            "Basal cell carcinoma": ("2–4 months", ["Excision or Mohs surgery", "Topical creams", "Laser therapy", "Regular follow-up"]),
            "Actinic keratoses": ("1–3 months", ["Cryotherapy", "Topical chemotherapy", "Sun protection"]),
            "Benign": ("2–4 weeks", ["Monitor changes", "Minor removal if irritated", "Maintain hygiene"]),
            "Vascular lesions": ("2–4 weeks", ["Monitor changes", "Laser therapy if needed"]),
            "Dermatofibroma": ("3–5 weeks", ["Monitor changes", "Removal if bothersome", "Use soothing creams"])
        }

        recovery, meds = None, None
        for key in plans.keys():
            if key in cancer_type:
                recovery, meds = plans[key]
                break

        if recovery and meds:
            st.markdown(f'<div class="card"><h3>⏳ Expected Recovery: <span class="highlight">{recovery}</span></h3>'
                        '<ul>' + ''.join([f'<li>{m}</li>' for m in meds]) + '</ul></div>', unsafe_allow_html=True)
        else:
            st.info("No specific plan found for the selected type. Please consult a dermatologist for a tailored plan.")
