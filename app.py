import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Page config
st.set_page_config(page_title="Safe Skin", page_icon="🩺", layout="wide")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "skin_cancer_model.h5")
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# --- Sidebar Navigation ---
page = st.sidebar.selectbox("Navigate", ["Home", "Summary", "Prediction"])

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.title("🩺 Safe Skin: AI-based Skin Cancer Detection")

    st.subheader("Project Overview")
    st.write("""
**Safe Skin** is a deep learning system designed to detect and classify skin lesions from dermoscopic images.  
Early detection of skin cancer significantly improves patient survival. This system assists dermatologists and enables self-screening in areas with limited access to medical expertise.
""")

    st.subheader("Project Goal")
    st.write("""
- Provide **accurate and fast detection** of skin cancer.
- Assist dermatologists in making **informed decisions**.
- Enable **self-screening** for individuals in remote or resource-limited areas.
""")

    st.subheader("Advantages")
    st.write("""
- **Early diagnosis** improves survival chances.
- Reduces **human error** in lesion classification.
- Can be integrated into **mobile applications** for tele-dermatology.
- **Visual explanation** via Grad-CAM highlights important regions.
""")

# ---------------- SUMMARY PAGE ----------------
elif page == "Summary":
    st.title("📊 Summary of Safe Skin System")

    st.subheader("Dataset & Classes")
    st.write("""
The system uses the HAM10000 dataset (subset) with JPEG images and CSV metadata.  

**Classes:**
- Melanocytic nevi (nv) – Benign
- Melanoma (mel) – Malignant
- Benign keratosis (bkl) – Benign
- Basal cell carcinoma (bcc) – Malignant
- Actinic keratoses (akiec) – Precancerous
- Vascular lesions (vasc) – Benign
- Dermatofibroma (df) – Benign
""")

    st.subheader("Workflow")
    st.write("""
1. **Input:** Dermoscopic image uploaded by the user.
2. **Preprocessing:** Hair removal, color normalization, resizing, data augmentation.
3. **Prediction:** CNN or Transfer Learning model (ResNet50, EfficientNet) predicts the lesion class.
4. **Visualization:** Grad-CAM heatmap highlights important regions influencing the prediction.
""")

    st.subheader("Implementation Pipeline")
    st.write("""
- **Data Preprocessing:** Load images, remove hair, resize, normalize, augment.  
- **Model Development:** Custom CNN or Transfer Learning with dropout to prevent overfitting.  
- **Training & Evaluation:** Optimizers: Adam/SGD, Loss: Categorical Cross-Entropy, Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.  
- **Explainability:** Grad-CAM to visualize lesion regions influencing predictions.
""")

# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":
    st.title("🔬 Safe Skin Prediction")

    # File uploader
    uploaded_file = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("🔍 Predict"):
            st.write("Analyzing image...")

            # Preprocess image
            img = image.resize((224, 224))  # adjust to your model input
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)
            class_names = ["Melanocytic nevi", "Melanoma", "Benign keratosis", 
                           "Basal cell carcinoma", "Actinic keratoses", "Vascular lesions", "Dermatofibroma"]
            predicted_class = class_names[np.argmax(preds)]
            confidence = np.max(preds)

            st.success(f"### Prediction: {predicted_class}")
            st.write(f"**Confidence:** {confidence*100:.2f}%")
