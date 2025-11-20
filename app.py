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
# GLOBAL CSS
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
# SIMPLE "AI-LIKE" ANALYZER (No Model Used)
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
# NAVIGATION
# ---------------------------------------------------------
st.markdown("""
<div style="text-align:center;">
    <a class="nav-btn" href="?page=Home">🏠 Home</a>
    <a class="nav-btn" href="?page=Upload">📤 Upload</a>
    <a class="nav-btn" href="?page=Results">📊 Results</a>
    <a class="nav-btn" href="?page=Care">💡 Recommendations</a>
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
            Welcome to your smart skin wellness assistant.<br>
            Analyze, visualize, and get insights for better skin health.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='glass-box'>
        <h3 style='color:white;'>📤 Upload & Analyze</h3>
        <p style='color:#ccc;'>Upload your skin image and let the system evaluate the patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-box'>
        <h3 style='color:white;'>📊 Visual Results</h3>
        <p style='color:#ccc;'>View a detailed interpretation of your scan.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# UPLOAD PAGE
# ---------------------------------------------------------
elif page == "Upload":
    st.markdown("<div class='header-card'><h2 style='color:white;'>📤 Upload Skin Image</h2></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)

        with st.spinner("Analyzing..."):
            result = analyze_image(img)

        st.session_state["result"] = result

        st.markdown(f"<div class='result-label'>{result}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# RESULTS PAGE
# ---------------------------------------------------------
elif page == "Results":
    st.markdown("<div class='header-card'><h2 style='color:white;'>📊 Results</h2></div>", unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.warning("Please upload an image first.")
    else:
        st.markdown(f"""
        <div class='glass-box'>
            <h3 style='color:white;'>🧠 Predicted Condition</h3>
            <p class='result-label'>{st.session_state['result']}</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# RECOMMENDATIONS PAGE
# ---------------------------------------------------------
elif page == "Care":
    st.markdown("<div class='header-card'><h2 style='color:white;'>💡 Recommendations</h2></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-box'>
    <h3 style='color:white;'>🌿 Care Tips</h3>
    <ul style='color:#ddd;'>
        <li>Apply broad-spectrum sunscreen daily.</li>
        <li>Avoid excessive sunlight exposure.</li>
        <li>Keep the skin moisturized.</li>
        <li>Monitor changes in shape, color, and size.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------------
elif page == "About":
    st.markdown("<div class='header-card'><h2 style='color:white;'>ℹ️ About</h2></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-box'>
        <p style='color:#ddd;'>
        Skin Health AI is designed to help users evaluate skin conditions visually
        and raise awareness about early detection and wellness.
        </p>
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
