import streamlit as st
from PIL import Image, ImageFilter
import numpy as np

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Skin Health AI",
    layout="wide",
    page_icon="🩺"
)

# -----------------------------------------
# DARK MODE - GLASS UI CSS
# -----------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}

.header-card {
    backdrop-filter: blur(12px);
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 20px;
    text-align: center;
}

.glass-box {
    backdrop-filter: blur(12px);
    background: rgba(255,255,255,0.06);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.12);
    transition: 0.3s ease;
}

.glass-box:hover {
    background: rgba(255,255,255,0.12);
    transform: translateY(-5px);
}

.nav-btn {
    padding: 10px 18px;
    margin: 5px;
    border-radius: 10px;
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    border: none;
    color: white;
    cursor: pointer;
    font-size: 15px;
    transition: 0.25s;
}

.nav-btn:hover {
    transform: scale(1.05);
    box-shadow: 0 0 10px #2575fc;
}

.result-label {
    background: linear-gradient(135deg,#ff512f,#dd2476);
    color:white;
    padding:15px;
    border-radius:12px;
    font-size:20px;
    text-align:center;
    font-weight:600;
}

.footer {
    margin-top: 40px;
    text-align: center;
    color: #ddd;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# INITIALIZE PAGE STATE
# -----------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# -----------------------------------------
# PAGE SWITCHER FUNCTION
# -----------------------------------------
def switch_page(p):
    st.session_state.page = p

# -----------------------------------------
# SIMPLE ANALYZER (FAKE MODEL)
# -----------------------------------------
def analyze_image(img):
    img_resized = img.resize((256, 256))
    arr = np.array(img_resized)

    gray = np.mean(arr, axis=2)
    darkness = gray.mean()

    r, g = arr[:,:,0], arr[:,:,1]
    redness = np.mean(r - g)

    texture_img = img_resized.filter(ImageFilter.FIND_EDGES)
    texture = np.array(texture_img).var()

    if darkness < 90 and texture > 35000:
        return "Melanoma"
    if redness > 25:
        return "Vascular Lesion"
    if texture > 30000:
        return "Actinic Keratosis"

    return "Nevus"

# -----------------------------------------
# SIDE NAVIGATION (Same Window)
# -----------------------------------------
st.sidebar.title("🧭 Navigation")

st.sidebar.button("🏠 Home", on_click=lambda: switch_page("Home"))
st.sidebar.button("📤 Upload & Predict", on_click=lambda: switch_page("Upload"))
st.sidebar.button("🩺 Treatment Plan", on_click=lambda: switch_page("Treatment"))
st.sidebar.button("💡 Doctor's Advice", on_click=lambda: switch_page("Advice"))
st.sidebar.button("ℹ️ About", on_click=lambda: switch_page("About"))


# =========================================================
#                     PAGE CONTENT
# =========================================================

# -----------------------------------------
# HOME PAGE
# -----------------------------------------
if st.session_state.page == "Home":
    st.markdown("""
    <div class='header-card'>
        <h1 style='color:white;'>🩺 Skin Health AI</h1>
        <p style='color:#ddd;'>Your personal skin wellness companion.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='glass-box'>
            <h3 style='color:white;'>📤 Upload & Predict</h3>
            <p style='color:#ccc;'>Upload a skin image to get a quick analysis based on texture, color, and patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-box'>
            <h3 style='color:white;'>🩺 Treatment & Care</h3>
            <p style='color:#ccc;'>Explore customized care suggestions and treatment paths.</p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------
# UPLOAD PAGE
# -----------------------------------------
elif st.session_state.page == "Upload":
    st.markdown("<div class='header-card'><h2 style='color:white;'>📤 Upload Image</h2></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)

        with st.spinner("Analyzing..."):
            result = analyze_image(img)

        st.session_state["result"] = result

        st.markdown(f"<div class='result-label'>{result}</div>", unsafe_allow_html=True)


# -----------------------------------------
# TREATMENT PAGE
# -----------------------------------------
elif st.session_state.page == "Treatment":
    st.markdown("<div class='header-card'><h2 style='color:white;'>🩺 Treatment Plan</h2></div>", unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.warning("Please upload an image first.")
    else:
        cond = st.session_state["result"]

        treatment = {
            "Melanoma": "Seek urgent dermatology support; early removal is crucial.",
            "Vascular Lesion": "Laser therapy + anti-redness creams may help.",
            "Actinic Keratosis": "Cryotherapy or topical medication is common.",
            "Nevus": "Benign; just monitor for changes."
        }

        st.markdown(f"""
        <div class='glass-box'>
            <h3 style='color:white;'>Recommended Treatment for {cond}</h3>
            <p style='color:#ccc;'>{treatment.get(cond)}</p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------
# DOCTOR ADVICE PAGE
# -----------------------------------------
elif st.session_state.page == "Advice":
    st.markdown("<div class='header-card'><h2 style='color:white;'>💡 Doctor's Advice</h2></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-box'>
        <h4 style='color:white;'>General Dermatology Tips</h4>
        <ul style='color:#ccc;'>
            <li>Use SPF 30 sunscreen daily.</li>
            <li>Avoid peak sunlight (11 AM – 4 PM).</li>
            <li>Moisturize regularly to protect skin barrier.</li>
            <li>Monitor moles for change in shape, size, or color.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------
# ABOUT PAGE
# -----------------------------------------
elif st.session_state.page == "About":
    st.markdown("<div class='header-card'><h2 style='color:white;'>ℹ️ About This App</h2></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-box'>
        <p style='color:#ddd;'>
        Skin Health AI is created to help users understand the appearance of skin abnormalities.
        It analyzes visible patterns and provides early awareness—not a medical diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------
# FOOTER
# -----------------------------------------
st.markdown("""
<div class='footer'>
© 2025 Skin Health AI | Built with ❤️
</div>
""", unsafe_allow_html=True)
