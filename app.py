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
    <div class='header-card' style='margin-top:30px; margin-bottom:40px;'>
        <h1 style='color:white; text-align:center; font-size:42px;'>🩺 Skin Health AI</h1>
        <p style='color:#ddd; text-align:center; font-size:18px; margin-top:10px;'>
            Your intelligent assistant for early skin health awareness.<br>
            Explore tools designed to analyze images, understand conditions, and get dermatology-backed advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -------- Intro Card -------- #
    st.markdown("""
    <div class='glass-box' style='padding:25px; margin-bottom:40px;'>
        <h2 style='color:white; margin-bottom:10px;'>📌 What You Can Do Here</h2>
        <p style='color:#ccc; font-size:17px;'>
        Skin Health AI guides you through a complete flow — from uploading your image to receiving predictions, 
        treatment suggestions, and expert tips to maintain skin wellness.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create extra vertical spacing
    st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)

    # ======================
    #       GRID START
    # ======================

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class='glass-box' style='padding:25px; min-height:330px; margin-bottom:35px;'>
            <h3 style='color:white;'>📤 Upload & Predict</h3>
            <p style='color:#ccc; font-size:15px;'>
                Upload your skin image and let AI analyze texture, color variation, and lesion patterns 
                to predict possible skin conditions.
                <br><br><b>Includes:</b>
                <ul style='color:#ccc; line-height:1.6;'>
                    <li>Instant prediction</li>
                    <li>Smart image analysis</li>
                    <li>Secure offline processing</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-box' style='padding:25px; min-height:330px; margin-bottom:35px;'>
            <h3 style='color:white;'>🩺 Treatment Plan</h3>
            <p style='color:#ccc; font-size:15px;'>
                Access dermatology-based treatment recommendations tailored to the predicted skin condition.
                <br><br><b>Includes:</b>
                <ul style='color:#ccc; line-height:1.6;'>
                    <li>Condition-specific treatment steps</li>
                    <li>When to consult a dermatologist</li>
                    <li>Precautions & care routines</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Add large vertical gap before next grid
    st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown("""
        <div class='glass-box' style='padding:25px; min-height:330px; margin-bottom:35px;'>
            <h3 style='color:white;'>💡 Doctor's Advice</h3>
            <p style='color:#ccc; font-size:15px;'>
                Evidence-based dermatology tips to help you maintain healthy, safe, and glowing skin.
                <br><br><b>Includes:</b>
                <ul style='color:#ccc; line-height:1.6;'>
                    <li>Daily care recommendations</li>
                    <li>Prevention techniques</li>
                    <li>Skincare do’s & don’ts</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='glass-box' style='padding:25px; min-height:330px; margin-bottom:35px;'>
            <h3 style='color:white;'>ℹ️ About This App</h3>
            <p style='color:#ccc; font-size:15px;'>
                Learn about the mission, purpose, technology, and team behind Skin Health AI — built 
                to promote early awareness and accessible digital dermatology.
                <br><br><b>Includes:</b>
                <ul style='color:#ccc; line-height:1.6;'>
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

    st.markdown("""
    <div class='header-card' style='margin-bottom:25px;'>
        <h2 style='color:white;'>🩺 Treatment Plan</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-box' style='padding:20px; margin-bottom:20px;'>
        <p style='color:#ccc; font-size:16px;'>
            Select the predicted or known skin condition to view recommended
            treatment options and medical guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dropdown list
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

    # Treatment dictionary (with color styling fixed)
    treatment = {
        "Melanoma": """
        <ul style='color:#ccc; line-height:1.6;'>
            <li>Immediate dermatology consultation is recommended.</li>
            <li>Surgical removal of the lesion as early as possible.</li>
            <li>Lymph node evaluation depending on melanoma depth.</li>
            <li>Regular follow-up every 3–6 months.</li>
        </ul>
        """,

        "Vascular Lesion": """
        <ul style='color:#ccc; line-height:1.6;'>
            <li>Laser treatment is the most effective option.</li>
            <li>Anti-redness topical creams can reduce irritation.</li>
            <li>Cold compress may help with swelling.</li>
        </ul>
        """,

        "Actinic Keratosis": """
        <ul style='color:#ccc; line-height:1.6;'>
            <li>Cryotherapy (freezing the lesion).</li>
            <li>Topical medications like 5-FU or imiquimod.</li>
            <li>Daily use of SPF 50+ sunscreen.</li>
        </ul>
        """,

        "Squamous Cell Carcinoma": """
        <ul style='color:#ccc; line-height:1.6;'>
            <li>Surgical removal is the primary treatment.</li>
            <li>Mohs surgery for sensitive or facial areas.</li>
            <li>Follow-up every 3–6 months is recommended.</li>
        </ul>
        """,

        "Basal Cell Carcinoma": """
        <ul style='color:#ccc; line-height:1.6;'>
            <li>Outpatient surgical excision.</li>
            <li>Topical therapy for superficial types.</li>
            <li>Radiation therapy for large or difficult areas.</li>
        </ul>
        """,

        "Seborrheic Keratosis": """
        <ul style='color:#ccc; line-height:1.6;'>
            <li>Harmless; treatment optional.</li>
            <li>Cryotherapy or laser removal for cosmetic reasons.</li>
            <li>Avoid scratching or picking.</li>
        </ul>
        """,

        "Pigmented Benign Keratosis": """
        <ul style='color:#ccc; line-height:1.6;'>
            <li>Benign; no treatment needed.</li>
            <li>Cryotherapy or laser optional for cosmetic improvement.</li>
            <li>Monitor for unusual changes.</li>
        </ul>
        """,

        "Dermatofibroma": """
        <ul style='color:#ccc; line-height:1.6;'>
            <li>Harmless and stable; no treatment needed.</li>
            <li>Minor surgery only if painful or irritating.</li>
        </ul>
        """,

        "Nevus": """
        <ul style='color:#ccc; line-height:1.6;'>
            <li>Benign mole; monitor size and color over time.</li>
            <li>Yearly dermatology check-up recommended.</li>
            <li>Removal only if changing or irritating.</li>
        </ul>
        """
    }

    # Button to show plan
    if st.button("Show Treatment Plan"):
        st.markdown(f"""
        <div class='glass-box' style='padding:25px; margin-top:20px;'>
            <h3 style='color:white;'>Recommended Treatment for {selected_cond}</h3>
            <p style='color:#ccc; font-size:16px;'>{treatment[selected_cond]}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------
# DOCTOR ADVICE PAGE
# -----------------------------------------
elif st.session_state.page == "Advice":

    # Header
    st.markdown(
        "<div class='header-card'><h2 style='color:white;'>💡 Doctor's Advice</h2></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
            /* Match global font */
            .advice-box * {
                font-family: inherit !important;
            }

            /* Individual advice item styling */
            .advice-item {
                background: rgba(255, 255, 255, 0.06);
                padding: 12px 18px;
                margin-bottom: 10px;
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.15);
                color: #ddd;
                font-size: 17px;
                line-height: 1.6;
            }
        </style>

        <div class='advice-box' 
             style='padding:45px 50px; border-radius:15px;'>

            <h4 style='color:white; margin-bottom:18px; font-size:22px;'>
                💊 Essential Dermatology Care Tips
            </h4>

            <p style='color:#ccc; margin-bottom:25px; font-size:17px; line-height:1.9;'>
                Follow these dermatologist-approved guidelines to maintain healthy, glowing,
                and well-protected skin:
            </p>

            <div class='advice-item'>Use a broad-spectrum sunscreen (SPF 30 or higher) every day.</div>
            <div class='advice-item'>Reapply sunscreen every 2–3 hours when outdoors.</div>
            <div class='advice-item'>Avoid peak sunlight exposure between 11 AM and 4 PM.</div>
            <div class='advice-item'>Moisturize twice daily to support the skin barrier.</div>
            <div class='advice-item'>Stay hydrated — drink at least 2–3 liters of water daily.</div>
            <div class='advice-item'>Avoid touching or picking at lesions, rashes, or acne.</div>
            <div class='advice-item'>Use gentle, fragrance-free cleansers to reduce irritation.</div>
            <div class='advice-item'>Exfoliate only 1–2 times per week — excessive scrubbing damages skin.</div>
            <div class='advice-item'>Wear protective clothing, hats, and sunglasses when outside.</div>
            <div class='advice-item'>Monitor moles for changes in symmetry, border, color, or size.</div>
            <div class='advice-item'>Avoid tanning beds — they significantly increase skin cancer risk.</div>
            <div class='advice-item'>Do patch tests before trying new skin care products.</div>
            <div class='advice-item'>Eat antioxidant-rich foods (berries, nuts, green vegetables).</div>
            <div class='advice-item'>Reduce stress — it can trigger acne, eczema, and psoriasis.</div>
            <div class='advice-item'>Sleep at least 7–8 hours daily to support skin recovery.</div>
            <div class='advice-item'>Use retinol or vitamin C serums at night for skin repair (if tolerated).</div>
            <div class='advice-item'>Keep your phone, pillowcase, and makeup brushes clean.</div>
            <div class='advice-item'>Avoid heavy makeup on irritated or inflamed skin.</div>
            <div class='advice-item'>Visit a dermatologist once a year for a full skin check.</div>
            <div class='advice-item'>Seek medical help if a spot bleeds, grows rapidly, or becomes painful.</div>

        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------
# ABOUT PAGE
# -----------------------------------------
elif st.session_state.page == "About":
    st.markdown("<div class='header-card'><h2 style='color:white;'>ℹ️ About This App</h2></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-box' style='padding:25px 30px;'>

        <!-- Mission -->
        <div style='margin-bottom:25px;'>
            <h3 style='color:white; margin-bottom:8px;'>🌟 Mission</h3>
            <p style='color:#ccc; line-height:1.8;'>
                Our mission is to make skin health insights accessible to everyone by providing a simple,
                visually-intelligent tool that helps users understand their skin's condition early and clearly.
            </p>
        </div>

        <!-- Goal -->
        <div style='margin-bottom:25px;'>
            <h3 style='color:white; margin-bottom:8px;'>🎯 Goal</h3>
            <p style='color:#ccc; line-height:1.8;'>
                The goal of Skin Health AI is to empower people with quick, informative skin analysis
                and guidance, helping them take the right steps toward better skin care and early awareness.
            </p>
        </div>

        <!-- Problem Statement -->
        <div style='margin-bottom:25px;'>
            <h3 style='color:white; margin-bottom:8px;'>❗ Problem Statement</h3>
            <p style='color:#ccc; line-height:1.8;'>
                Many individuals delay seeking medical help for skin abnormalities due to lack of
                awareness, hesitation, or not knowing whether a spot or lesion is concerning.
                This delay can lead to late detection of serious conditions.
            </p>
        </div>

        <!-- What This App Does -->
        <div style='margin-bottom:25px;'>
            <h3 style='color:white; margin-bottom:12px;'>💡 What This App Does</h3>
            <ul style='color:#ccc; line-height:1.9; font-size:16px; padding-left:20px;'>
                <li>Analyzes the uploaded skin image based on texture, color, and intensity patterns.</li>
                <li>Provides possible skin condition identification.</li>
                <li>Offers treatment guidance based on selected condition.</li>
                <li>Provides dermatologist-style general advice for daily skin care.</li>
                <li>Helps users stay informed and aware of their skin health.</li>
            </ul>
        </div>

        <!-- Why This Matters -->
        <div style='margin-bottom:25px;'>
            <h3 style='color:white; margin-bottom:8px;'>🧑‍⚕️ Why This Matters</h3>
            <p style='color:#ccc; line-height:1.8;'>
                Early awareness is key in preventing severe outcomes in many skin conditions.
                This app encourages proactive skin monitoring and helps users understand when
                they should seek professional medical evaluation.
            </p>
        </div>

        <!-- Disclaimer -->
        <div style='margin-bottom:10px;'>
            <h3 style='color:white; margin-bottom:8px;'>⚠️ Disclaimer</h3>
            <p style='color:#ccc; line-height:1.8;'>
                This tool is intended for awareness and educational purposes only.
                It is <b>not</b> a medical diagnostic system. Always consult a certified dermatologist
                for clinical evaluation and treatment.
            </p>
        </div>

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
