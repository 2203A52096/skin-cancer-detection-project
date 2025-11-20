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
st.markdown("""
<style>

/* Navigation buttons - gradient */
[data-testid="stSidebar"] button {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 10px 18px !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    margin-bottom: 10px !important;
    box-shadow: 0 0 10px rgba(32, 32, 60, 0.5) !important;
    transition: all 0.2s ease-in-out !important;
}

/* Hover effect - brighter gradient */
[data-testid="stSidebar"] button:hover {
    background: linear-gradient(135deg, #1a173d, #3e3980, #2f3054) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 0 15px rgba(70, 70, 120, 0.7) !important;
}

/* Active effect - pressed */
[data-testid="stSidebar"] button:active {
    transform: scale(0.97) !important;
    background: linear-gradient(135deg, #0d0b20, #26224e, #1e1e35) !important;
}
</style>
""", unsafe_allow_html=True)



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
        <h1 style='color:white; text-align:center;'>🩺 Skin Health AI</h1>
        <p style='color:#ddd; text-align:center; font-size:17px;'>
            Welcome to your all-in-one intelligent skin health companion.<br>
            Navigate through our smart tools to analyze, understand, and care for your skin.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-box'>
        <h2 style='color:white;'>📌 What You Can Do Here</h2>
        <p style='color:#ccc; font-size:16px;'>
        Skin Health AI provides a complete workflow to help you understand your skin condition.<br><br>
        Below is a quick overview of every page:
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Four Sections in a Grid ---
    col1, col2 = st.columns(2)

    # Upload Page Intro
    with col1:
        st.markdown("""
        <div class='glass-box'>
            <h3 style='color:white;'>📤 Upload & Predict</h3>
            <p style='color:#ccc;'>
                Upload your skin image and let AI analyze texture, color patterns, and lesion features 
                to predict possible skin conditions.
                <br><br>Includes:
                <ul style='color:#ccc;'>
                    <li>Instant prediction</li>
                    <li>Smart image analysis</li>
                    <li>Secure offline processing</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Treatment Page Intro
    with col2:
        st.markdown("""
        <div class='glass-box'>
            <h3 style='color:white;'>🩺 Treatment Plan</h3>
            <p style='color:#ccc;'>
                Once your skin condition is predicted, this page provides detailed treatment guidance 
                based on authentic dermatology references.
                <br><br>Includes:
                <ul style='color:#ccc;'>
                    <li>Condition-specific treatment steps</li>
                    <li>When to consult a dermatologist</li>
                    <li>Precaution and care guidance</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Advice
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class='glass-box'>
            <h3 style='color:white;'>💡 Doctor's Advice</h3>
            <p style='color:#ccc;'>
                Helpful dermatology-backed recommendations to maintain healthy skin.
                <br><br>Includes:
                <ul style='color:#ccc;'>
                    <li>General skin care rules</li>
                    <li>Prevention tips</li>
                    <li>Do's and Don’ts</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # About Page Intro
    with col4:
        st.markdown("""
        <div class='glass-box'>
            <h3 style='color:white;'>ℹ️ About This App</h3>
            <p style='color:#ccc;'>
                Learn about the purpose of this project, how the AI works, developers behind it, 
                and our mission to make skin health accessible.
                <br><br>Includes:
                <ul style='color:#ccc;'>
                    <li>Project goals</li>
                    <li>Team & guidance</li>
                    <li>Technology stack</li>
                </ul>
            </p>
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

    st.markdown("""
    <div class='glass-box'>
        <p style='color:#ccc; font-size:16px;'>
            Select the diagnosed or predicted skin condition below to view recommended
            treatment options and medical guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 🔽 Dropdown List of All Skin Conditions
    condition_list = [
        "Melanoma",
        "Vascular Lesion",
        "Actinic Keratosis",
        "Squamous Cell Carcinoma",
        "Basal Cell Carcinoma",
        "Seborrheic Keratosis",
        "Pigmented Benign Keratosis",
        "Dermatofibroma",
        "Nevus"
    ]

    selected_cond = st.selectbox("Select Skin Condition:", condition_list)

    # Treatment Plans Dictionary
    treatment = {
        "Melanoma": """
        <ul>
            <li>Immediate dermatology consultation is recommended.</li>
            <li>Surgical removal of the lesion at the earliest.</li>
            <li>Possible lymph node evaluation depending on depth.</li>
            <li>Regular follow-up every 3–6 months.</li>
        </ul>
        """,

        "Vascular Lesion": """
        <ul>
            <li>Laser treatment is highly effective.</li>
            <li>Topical anti-redness creams may reduce irritation.</li>
            <li>Cold compress can help minimize swelling.</li>
        </ul>
        """,

        "Actinic Keratosis": """
        <ul>
            <li>Cryotherapy (freezing the lesion) is common.</li>
            <li>Topical medications like 5-FU or imiquimod.</li>
            <li>Strict sun protection and SPF 50+ sunscreen daily.</li>
        </ul>
        """,

        "Squamous Cell Carcinoma": """
        <ul>
            <li>Surgical excision is the primary treatment.</li>
            <li>Mohs surgery for sensitive areas like the face.</li>
            <li>Follow-up every 3–6 months is recommended.</li>
        </ul>
        """,

        "Basal Cell Carcinoma": """
        <ul>
            <li>Outpatient surgical removal.</li>
            <li>Topical treatments for superficial types.</li>
            <li>Radiotherapy for large or difficult areas.</li>
        </ul>
        """,

        "Seborrheic Keratosis": """
        <ul>
            <li>Usually harmless; treatment is optional.</li>
            <li>Cryotherapy or laser removal for cosmetic concerns.</li>
            <li>Avoid scratching or picking.</li>
        </ul>
        """,

        "Pigmented Benign Keratosis": """
        <ul>
            <li>Generally harmless; no treatment required.</li>
            <li>Laser or cryotherapy optional for appearance.</li>
            <li>Monitor for sudden changes.</li>
        </ul>
        """,

        "Dermatofibroma": """
        <ul>
            <li>Benign and stable; treatment not required.</li>
            <li>Minor surgery optional if painful or irritating.</li>
        </ul>
        """,

        "Nevus": """
        <ul>
            <li>Benign mole; record size and monitor changes.</li>
            <li>Dermatology check-up every year recommended.</li>
            <li>Removal only if irregularity or irritation develops.</li>
        </ul>
        """
    }

    # 🔘 Button to Show Treatment Plan
    if st.button("Show Treatment Plan"):
        st.markdown(f"""
        <div class='glass-box'>
            <h3 style='color:white;'>Recommended Treatment for {selected_cond}</h3>
            <p style='color:#ccc; font-size:16px;'>{treatment[selected_cond]}</p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------
# DOCTOR ADVICE PAGE
# -----------------------------------------
elif st.session_state.page == "Advice":
    st.markdown("<div class='header-card'><h2 style='color:white;'>💡 Doctor's Advice</h2></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-box'>
        <h4 style='color:white;'>💊 Essential Dermatology Care Tips</h4>
        <p style='color:#ccc;'>Follow these dermatologist-approved guidelines to maintain healthy and protected skin:</p>

        <ul style='color:#ccc; line-height:1.7; font-size:16px;'>
            <li>Use a broad-spectrum sunscreen (SPF 30 or higher) every day.</li>
            <li>Reapply sunscreen every 2–3 hours when outdoors.</li>
            <li>Avoid peak sunlight exposure between 11 AM and 4 PM.</li>
            <li>Moisturize twice daily to support the skin barrier.</li>
            <li>Stay hydrated — drink at least 2–3 liters of water daily.</li>
            <li>Avoid touching or picking at lesions, rashes, or acne.</li>
            <li>Use gentle, fragrance-free cleansers to reduce irritation.</li>
            <li>Exfoliate only 1–2 times per week — excessive scrubbing damages skin.</li>
            <li>Wear protective clothing, hats, and sunglasses when outside.</li>
            <li>Monitor moles for changes in symmetry, border, color, or size.</li>
            <li>Avoid tanning beds — they significantly increase skin cancer risk.</li>
            <li>Do patch tests before trying new skin care products.</li>
            <li>Eat antioxidant-rich foods (berries, nuts, green vegetables).</li>
            <li>Reduce stress — it can trigger acne, eczema, and psoriasis.</li>
            <li>Sleep at least 7–8 hours daily to support skin recovery.</li>
            <li>Use retinol or vitamin C serums at night for skin repair (if tolerated).</li>
            <li>Keep your phone, pillowcase, and makeup brushes clean.</li>
            <li>Avoid heavy makeup on irritated or inflamed skin.</li>
            <li>Visit a dermatologist once a year for a full skin check.</li>
            <li>Seek medical help if a spot bleeds, grows rapidly, or becomes painful.</li>
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
        <h3 style='color:white;'>🌟 Mission</h3>
        <p style='color:#ccc;'>
            Our mission is to make skin health insights accessible to everyone by providing a simple, 
            visually-intelligent tool that helps users understand their skin's condition early and clearly.
        </p>

        <h3 style='color:white;'>🎯 Goal</h3>
        <p style='color:#ccc;'>
            The goal of Skin Health AI is to empower people with quick, informative skin analysis 
            and guidance, helping them take the right steps toward better skin care and early awareness.
        </p>

        <h3 style='color:white;'>❗ Problem Statement</h3>
        <p style='color:#ccc;'>
            Many individuals delay seeking medical help for skin abnormalities due to lack of 
            awareness, hesitation, or not knowing whether a spot or lesion is concerning. 
            This delay can lead to late detection of serious conditions.
        </p>

        <h3 style='color:white;'>💡 What This App Does</h3>
        <ul style='color:#ccc; line-height:1.7;'>
            <li>Analyzes the uploaded skin image based on texture, color, and intensity patterns.</li>
            <li>Provides possible skin condition identification.</li>
            <li>Offers treatment guidance based on selected condition.</li>
            <li>Provides dermatologist-style general advice for daily skin care.</li>
            <li>Helps users stay informed and aware of their skin health.</li>
        </ul>

        <h3 style='color:white;'>🧑‍⚕️ Why This Matters</h3>
        <p style='color:#ccc;'>
            Early awareness is key in preventing severe outcomes in many skin conditions. 
            This app encourages proactive skin monitoring and helps users understand when 
            they should seek professional medical evaluation.
        </p>

        <h3 style='color:white;'>⚠️ Disclaimer</h3>
        <p style='color:#ccc;'>
            This tool is intended for awareness and educational purposes only. 
            It is <b>not</b> a medical diagnostic system. Always consult a certified dermatologist 
            for clinical evaluation and treatment.
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
