import streamlit as st
from PIL import Image, ImageFilter
import numpy as np

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Skin Health AI",
    layout="wide",
    page_icon="🩺"
)

# ---------------------------------------------------------
# GLOBAL DARK UI CSS (Glassmorphism + Gradient)
# ---------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}


/* HEADER CARD */
.header-card {
    backdrop-filter: blur(12px);
    background: rgba(255, 255, 255, 0.08);
    padding: 25px 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 4px 25px rgba(0,0,0,0.35);
    margin-bottom: 25px;
}


/* GLASS BOXES */
.glass-box {
    backdrop-filter: blur(15px);
    background: rgba(255, 255, 255, 0.06);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.15);
    transition: 0.3s;
}

.glass-box:hover {
    background: rgba(255, 255, 255, 0.12);
    transform: translateY(-5px);
}


/* NAV BUTTONS */
.nav-btn {
    display:inline-block;
    padding:12px 18px;
    margin: 8px 6px;
    border-radius:12px;
    background:linear-gradient(135deg,#6a11cb,#2575fc);
    color:white !important;
    transition:0.25s;
    text-decoration:none;
    font-size:15px;
}

.nav-btn:hover {
    box-shadow:0 0 12px #2575fc;
    transform:scale(1.06);
}


/* RESULT LABEL */
.result-label {
    background: linear-gradient(135deg,#ff512f,#dd2476);
    padding: 14px;
    border-radius: 12px;
    color: white;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
}


/* FOOTER */
.footer {
    margin-top: 50px;
    padding: 10px;
    text-align:center;
    color:#ddd;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# AI-LIKE Image Analyzer (NO ML Model)
# ---------------------------------------------------------
def analyze_image(img):
    img_resized = img.resize((256, 256))
    arr = np.array(img_resized)

    gray = np.mean(arr, axis=2)
    darkness = gray.mean()

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    redness = np.mean(r - g)

    texture_img = img_resized.filter(ImageFilter.FIND_EDGES)
    texture = np.array(texture_img).var()

    edges = np.array(texture_img).mean() / 255

    center = gray[100:150, 100:150].mean()
    edge_illum = (gray[0:50].mean() + gray[-50:].mean()) / 2

    if darkness < 90 and texture > 35000:
        return "Melanoma"
    if redness > 25:
        return "Vascular Lesion"
    if texture > 30000 and darkness > 140:
        return "Actinic Keratosis"
    if texture > 28000 and edges > 0.35:
        return "Squamous Cell Carcinoma"
    if center > edge_illum + 20:
        return "Basal Cell Carcinoma"
    if texture > 26000 and darkness < 140:
        return "Seborrheic Keratosis"
    if 100 < darkness < 170 and texture < 20000:
        return "Pigmented Benign Keratosis"
    if edges < 0.20 and 130 < darkness < 200:
        return "Dermatofibroma"
    if 80 < darkness < 130:
        return "Nevus"

    return "Nevus"

# ---------------------------------------------------------
# NAVIGATION MENU (TOP BUTTONS)
# ---------------------------------------------------------
st.markdown("""
<div style="text-align:center;">
    <a class="nav-btn" href="?page=Home">🏠 Home</a>
    <a class="nav-btn" href="?page=Upload">📤 Upload & Predict</a>
    <a class="nav-btn" href="?page=Treatment">💊 Treatment Plan</a>
    <a class="nav-btn" href="?page=Advice">👨‍⚕️ Doctor’s Advice</a>
    <a class="nav-btn" href="?page=About">ℹ️ About</a>
</div>
""", unsafe_allow_html=True)

page = st.experimental_get_query_params().get("page", ["Home"])[0]

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
if page == "Home":
    st.markdown("""
    <div class='header-card'>
        <h1 style='color:white; text-align:center;'>🩺 Skin Health AI</h1>
        <p style='color:#ddd; text-align:center;'>
            Your stylish skin wellness companion.<br>
            Analyze, understand and explore your skin health.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='glass-box'>
        <h3 style='color:white;'>📤 Upload & Predict</h3>
        <p style='color:#ccc;'>Upload your skin lesion image and get an instant prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-box'>
        <h3 style='color:white;'>💊 Treatment Guidance</h3>
        <p style='color:#ccc;'>Learn recommended treatment paths for each condition.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# UPLOAD & PREDICT PAGE
# ---------------------------------------------------------
elif page == "Upload":
    st.markdown(
        "<div class='header-card'><h2 style='color:white;'>📤 Upload & Predict</h2></div>",
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("Upload your skin image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=350)

        with st.spinner("Analyzing image..."):
            result = analyze_image(img)

        st.session_state["result"] = result

        st.markdown(f"<div class='result-label'>Predicted Condition: {result}</div>",
                    unsafe_allow_html=True)

# ---------------------------------------------------------
# TREATMENT PLAN PAGE
# ---------------------------------------------------------
elif page == "Treatment":
    st.markdown(
        "<div class='header-card'><h2 style='color:white;'>💊 Treatment Plan</h2></div>",
        unsafe_allow_html=True
    )

    treatments = {
        "Melanoma": "Surgery, immunotherapy, targeted therapy.",
        "Vascular Lesion": "Laser therapy or light-based treatment.",
        "Actinic Keratosis": "Cryotherapy and topical medications.",
        "Squamous Cell Carcinoma": "Surgery, radiation therapy.",
        "Basal Cell Carcinoma": "Excision, freezing, topical chemo.",
        "Seborrheic Keratosis": "Laser or cryotherapy if removal needed.",
        "Pigmented Benign Keratosis": "Usually harmless; surface removal possible.",
        "Dermatofibroma": "Harmless; remove only if discomfort.",
        "Nevus": "Monitor or remove based on doctor’s suggestion."
    }

    choice = st.selectbox("Select a condition", list(treatments.keys()))

    st.markdown(f"""
    <div class='glass-box'>
        <h3 style='color:white;'>Treatment for {choice}</h3>
        <p style='color:#ddd;'>{treatments[choice]}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# DOCTOR’S ADVICE PAGE
# ---------------------------------------------------------
elif page == "Advice":
    st.markdown(
        "<div class='header-card'><h2 style='color:white;'>👨‍⚕️ Doctor’s Advice</h2></div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class='glass-box'>
        <ul style='color:#ddd;'>
            <li>Monitor sudden changes in shape, size, or color.</li>
            <li>Avoid peak sunlight (12 PM – 4 PM).</li>
            <li>Use sunscreen SPF 30+ every day.</li>
            <li>Never scratch or pick lesions.</li>
            <li>Consult a dermatologist if abnormalities persist.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------------
elif page == "About":
    st.markdown(
        "<div class='header-card'><h2 style='color:white;'>ℹ️ About</h2></div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class='glass-box'>
        <p style='color:#ddd;'>
            Skin Health AI is a smart and stylish visual assistant designed
            to help raise awareness of skin lesions, treatment options, 
            and general skin health guidance.
        </p>
        <p style='color:#bbb;'>Designed with ❤️ for awareness & self-care.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<div class='footer'>
© 2025 Skin Health AI | Designed with ❤️
</div>
""", unsafe_allow_html=True)
