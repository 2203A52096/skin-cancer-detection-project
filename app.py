import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Safe Skin", page_icon="🩺", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = "skincancercnn.h5"  # ensure this file is in the same folder as app.py

    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please make sure 'skincancercnn.h5' is in the same folder as app.py")
        return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None


# Load model once
model = load_model()

# ---------------- SIDEBAR NAVIGATION ----------------
page = st.sidebar.selectbox(
    "🧭 Navigate",
    ["🏠 Home", "🔬 Prediction", "💊 Solution"]
)

# ---------------- HOME PAGE ----------------
if page == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: #D6336C;'>🩺 <b>Safe Skin</b></h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #555;'>AI-powered Skin Cancer Detection & Support System</h3>", unsafe_allow_html=True)
    st.write("---")

    st.subheader("🌟 Features")
    st.write("""
    - Upload dermoscopic skin images for **AI-based classification**.
    - Get **instant predictions** of skin lesion type with **confidence scores**.
    - Access **possible treatment plans** and **recovery insights**.
    - Designed for both **dermatologists** and **patients** for decision support.
    """)

    st.subheader("🎯 Goals")
    st.write("""
    - Enable **early detection** of skin cancer using deep learning.
    - Support **dermatologists** with AI-assisted decision tools.
    - Provide **accessible screening** for people in remote areas.
    """)

    st.subheader("💡 Advantages")
    st.write("""
    - Improves **accuracy** and **speed** of diagnosis.  
    - Reduces **human error** in lesion interpretation.  
    - Can be integrated into **tele-dermatology and mobile apps**.  
    - Offers **visual explanation (Grad-CAM)** for better trust in AI decisions.
    """)

# ---------------- PREDICTION PAGE ----------------
elif page == "🔬 Prediction":
    st.title("🔬 Skin Lesion Prediction")

    uploaded_file = st.file_uploader("📤 Upload a dermoscopic image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("🔍 Predict"):
            st.write("🧠 Analyzing the image...")

            # Preprocess image
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
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

            st.success(f"### 🧾 Prediction: {predicted_class}")
            st.info(f"**Confidence Score:** {confidence * 100:.2f}%")

# ---------------- SOLUTION PAGE ----------------
elif page == "💊 Solution":
    st.title("💊 Treatment & Recovery Plan")

    cancer_type = st.selectbox(
        "Select the type of skin cancer:",
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
        st.subheader(f"🧠 Selected: {cancer_type}")
        st.write("---")

        # Possible recovery time and treatment plans
        if "Melanoma" in cancer_type:
            st.write("⏳ **Expected Recovery:** 4–6 months (depends on stage and spread)")
            st.write("""
            **Treatment Plan:**
            - Surgical removal of the lesion.  
            - Targeted therapy or immunotherapy for advanced stages.  
            - Regular skin checkups every 3–6 months.  
            - Maintain hydration, sun protection, and balanced nutrition.
            """)

        elif "Basal cell carcinoma" in cancer_type:
            st.write("⏳ **Expected Recovery:** 2–4 months")
            st.write("""
            **Treatment Plan:**
            - Excision or Mohs surgery for lesion removal.  
            - Topical creams (Imiquimod or Fluorouracil).  
            - Laser therapy for small lesions.  
            - Regular follow-up to monitor recurrence.
            """)

        elif "Actinic keratoses" in cancer_type:
            st.write("⏳ **Expected Recovery:** 1–3 months")
            st.write("""
            **Treatment Plan:**
            - Cryotherapy (freezing the lesion).  
            - Topical chemotherapy.  
            - Avoid direct sun exposure.  
            - Use high SPF sunscreen daily.
            """)

        elif "Benign" in cancer_type or "Vascular" in cancer_type:
            st.write("⏳ **Expected Recovery:** 2–4 weeks")
            st.write("""
            **Treatment Plan:**
            - Usually non-cancerous and may not need treatment.  
            - Monitor for changes in size, color, or shape.  
            - If irritated, minor surgical removal or laser therapy.  
            - Maintain good skin hygiene.
            """)

        elif "Dermatofibroma" in cancer_type:
            st.write("⏳ **Expected Recovery:** 3–5 weeks")
            st.write("""
            **Treatment Plan:**
            - Typically harmless; removal only if bothersome.  
            - Use soothing creams for irritation.  
            - Protect the area from friction or trauma.
            """)

        else:
            st.info("Please select a valid lesion type to view treatment suggestions.")
