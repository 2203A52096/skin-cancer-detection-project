# app.py
import streamlit as st
from PIL import Image, ImageFilter, ImageStat, ImageOps
import numpy as np
import io
import base64
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="SafeSkin — Demo", layout="wide", page_icon="🩺")

# -------------------------
# Styles (modern gradient + glass cards)
# -------------------------
st.markdown(
    """
    <style>
    :root{
      --bg1: #f0f7fb;
      --bg2: #e6f5ff;
      --card: rgba(255,255,255,0.85);
      --accent1: #06D6A0;
      --accent2: #118AB2;
      --muted: #6b7280;
    }
    html, body, [class*="css"]  {
      background: linear-gradient(135deg,var(--bg1),var(--bg2));
    }
    .header {
      background: linear-gradient(90deg,var(--accent1),var(--accent2));
      padding: 28px;
      border-radius: 14px;
      color: white;
      box-shadow: 0 10px 30px rgba(17,138,178,0.18);
      margin-bottom: 18px;
    }
    .brand {
      font-weight: 800;
      font-size: 28px;
      letter-spacing: -0.5px;
    }
    .subtitle { color: rgba(255,255,255,0.95); margin-top:6px; font-weight:500; }
    .card {
      background: var(--card);
      padding: 18px;
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(16,24,40,0.06);
      border: 1px solid rgba(255,255,255,0.4);
    }
    .small-muted { color: var(--muted); font-size:13px; }
    .risk-green { color: #059669; font-weight:700; }
    .risk-amber { color: #D97706; font-weight:700; }
    .risk-red { color: #DC2626; font-weight:700; }
    .footer { color: #475569; font-size:13px; margin-top:12px; text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Classes mapping
# -------------------------
CLASS_MAP = {
    0: 'Pigmented Benign Keratosis',
    1: 'Melanoma',
    2: 'Vascular Lesion',
    3: 'Actinic Keratosis',
    4: 'Squamous Cell Carcinoma',
    5: 'Basal Cell Carcinoma',
    6: 'Seborrheic Keratosis',
    7: 'Dermatofibroma',
    8: 'Nevus'
}

# -------------------------
# Helper utilities
# -------------------------
def to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")

def image_to_bytes(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def download_link(text: str, filename: str):
    b = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b}" download="{filename}">⬇️ Download report</a>'
    return href

# Softmax helper
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# -------------------------
# Rule-based analyzer (PIL + NumPy) — improved, returns score vector + explanation
# -------------------------
def analyze_image_rules(pil_img: Image.Image):
    img = pil_img.resize((256, 256))
    arr = np.array(img).astype(np.float32)

    # Basic measures
    gray = np.mean(arr, axis=2)  # HxW
    mean_gray = gray.mean()      # darkness lower -> darker lesion
    std_gray = gray.std()        # contrast/texture roughness
    variance = gray.var()

    # Color channels
    r_mean = arr[:, :, 0].mean()
    g_mean = arr[:, :, 1].mean()
    b_mean = arr[:, :, 2].mean()
    redness = r_mean - g_mean

    # Edge strength (use FIND_EDGES filter)
    edges = img.filter(ImageFilter.FIND_EDGES)
    edges_arr = np.array(edges.convert("L")).astype(float)
    edge_mean = edges_arr.mean()

    # Symmetry approximation: compare left-right halves
    left = gray[:, :128]
    right = np.fliplr(gray[:, 128:])
    sym_diff = np.mean(np.abs(left - right))

    # Compactness: using threshold area approximation
    thr = int(np.clip( (gray.mean() * 0.8), 10, 200 ))
    bw = (gray < thr).astype(np.uint8)
    area = bw.sum()
    perim_est = np.sum(np.abs(np.diff(bw, axis=0))) + np.sum(np.abs(np.diff(bw, axis=1)))
    compactness = (perim_est ** 2) / (area + 1e-6)

    # Feature vector (handcrafted)
    # We'll compute a score for each class as linear combination of features (then softmax)
    features = {
        'darkness': 255 - mean_gray,   # higher -> darker
        'texture': variance,
        'edges': edge_mean,
        'redness': redness,
        'symmetry': -sym_diff,         # more negative if asymmetric (so lower is worse)
        'compactness': compactness
    }

    # Base scores (heuristic weights per class)
    base_scores = np.zeros(len(CLASS_MAP), dtype=float)

    # Heuristic weight assignments (tuned to produce sensible-looking outputs)
    # Pigmented Benign Keratosis (brownish, moderate texture)
    base_scores[0] = 0.6*features['darkness'] + 0.4*features['texture'] - 0.3*features['edges']

    # Melanoma (very dark, high texture, irregular)
    base_scores[1] = 1.3*features['darkness'] + 0.9*features['edges'] + 0.7*(-features['symmetry']) + 0.6*features['texture']

    # Vascular lesion (high redness)
    base_scores[2] = 2.5*features['redness'] + 0.2*features['edges']

    # Actinic keratosis (rough, scaly, less dark)
    base_scores[3] = 0.8*features['texture'] + 0.6*features['edges'] - 0.4*features['darkness']

    # Squamous cell carcinoma (rough, irregular, medium-dark)
    base_scores[4] = 0.9*features['texture'] + 0.6*features['edges'] + 0.5*features['compactness']

    # Basal cell carcinoma (pearly/center bright vs edges)
    center = gray[96:160, 96:160].mean()
    periphery = np.concatenate([gray[:60,:].ravel(), gray[-60:,:].ravel()]).mean()
    base_scores[5] = 0.9*(periphery - center) + 0.3*features['edges']

    # Seborrheic keratosis (waxy, textured but often less asymmetric)
    base_scores[6] = 0.7*features['texture'] + 0.5*features['darkness'] - 0.2*(-features['symmetry'])

    # Dermatofibroma (small, well-defined)
    base_scores[7] = 0.5*(1.0 / (1.0 + features['compactness'])) + 0.2*(mean_gray)

    # Nevus (smooth, dark, symmetric)
    base_scores[8] = 0.8*features['darkness'] + 0.6*( -features['symmetry']) * -1 + 0.2*(1.0 / (1.0 + features['edges']))

    # Normalize with softmax to get probabilities
    probs = softmax(base_scores)

    # Create a human-friendly confidence: top prob scaled to 0-100
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    confidence = round(100.0 * top_prob, 2)

    explanation = {
        'mean_gray': round(float(mean_gray), 2),
        'texture_var': round(float(variance), 2),
        'edge_strength': round(float(edge_mean), 2),
        'redness': round(float(redness), 2),
        'symmetry_score': round(float(sym_diff), 2),
        'compactness': round(float(compactness), 2),
        'predicted_index': top_idx,
        'predicted_label': CLASS_MAP[top_idx],
        'confidence_percent': confidence,
        'probs': {CLASS_MAP[i]: float(round(100.0 * p, 2)) for i, p in enumerate(probs)}
    }

    return explanation

# -------------------------
# UI Layout: Sidebar navigation (Option B structure)
# -------------------------
PAGES = ["🏠 Dashboard", "📤 Upload & Analyze", "📊 Results Visualization", "💡 Recommendations", "ℹ️ About App"]

if 'history' not in st.session_state:
    st.session_state['history'] = []  # list of dicts with keys: timestamp, label, conf, img_bytes, explanation

# Header
st.markdown('<div class="header"><div class="brand">SafeSkin</div><div class="subtitle">Interactive demo — rule-based analysis (no ML model)</div></div>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 4])
with col1:
    choice = st.radio("Navigate", PAGES, index=0)

with col2:
    if choice == "🏠 Dashboard":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Welcome to SafeSkin — Dashboard")
        st.write("This demo app performs rule-based image analysis for nine lesion types. It is for educational/demo purposes only — not a medical diagnosis.")
        st.markdown("---")
        # Statistics from history
        history = st.session_state['history']
        total = len(history)
        st.markdown(f"**Total analyses performed:** {total}")

        if total:
            from collections import Counter
            ctr = Counter([h['label'] for h in history])
            counts = {k: ctr.get(CLASS_MAP[k], 0) for k in CLASS_MAP}
            st.write("Recent predictions:")
            cols = st.columns(3)
            i = 0
            for cls_idx, cls_name in CLASS_MAP.items():
                cols[i % 3].metric(label=cls_name, value=counts[cls_idx])
                i += 1

            st.markdown("**Last 5 predictions**")
            for h in history[-5:][::-1]:
                t = h['timestamp']
                st.write(f"- {t} → **{h['label']}** (Confidence: {h['confidence']}%)")
        else:
            st.info("No analyses yet. Go to 'Upload & Analyze' to try it.")

        st.markdown('</div>', unsafe_allow_html=True)

    elif choice == "📤 Upload & Analyze":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Upload Image & Get Analysis")
        st.write("Supported formats: JPG, PNG. This demo performs rule-based analysis and provides a probability-style output.")

        uploaded = st.file_uploader("Upload lesion image", type=["jpg", "jpeg", "png"])
        if uploaded:
            try:
                pil_img = Image.open(uploaded).convert("RGB")
            except Exception as e:
                st.error("Unable to open image. Try a different file.")
                pil_img = None

            if pil_img:
                st.image(pil_img, use_column_width=True, caption="Uploaded image")

                colA, colB = st.columns([1, 1])
                with colA:
                    if st.button("🔍 Analyze"):
                        with st.spinner("Running smart image analysis..."):
                            result = analyze_image_rules(pil_img)
                        pred_label = result['predicted_label']
                        conf = result['confidence_percent']
                        probs = result['probs']

                        # Save to history
                        b = image_to_bytes(pil_img)
                        st.session_state['history'].append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'label': pred_label,
                            'confidence': conf,
                            'img_bytes': b,
                            'explanation': result
                        })
                        st.success(f"Predicted: **{pred_label}** — Confidence: **{conf}%**")
                        st.write("Detailed reasoning:")
                        st.json({k: v for k, v in result.items() if k != 'probs'})

                        # Show probability bars (Plotly)
                        labels = list(probs.keys())
                        values = [probs[l] for l in labels]
                        fig = go.Figure(go.Bar(x=values, y=labels, orientation='h',
                                               marker=dict(line=dict(width=0.5, color='rgba(0,0,0,0.05)'))))
                        fig.update_layout(height=420, margin=dict(l=120, r=10, t=30, b=10), xaxis_title="Probability (%)")
                        st.plotly_chart(fig, use_container_width=True)

                        # Risk meter (simple mapping)
                        risk_txt = "Low"
                        risk_cls = "risk-green"
                        if conf >= 75:
                            risk_txt = "High"
                            risk_cls = "risk-red"
                        elif conf >= 50:
                            risk_txt = "Medium"
                            risk_cls = "risk-amber"

                        st.markdown(f"<div class='{risk_cls}'>Risk: {risk_txt}</div>", unsafe_allow_html=True)

                        # Downloadable report
                        report = f"SafeSkin Report\nDate: {datetime.now()}\nPrediction: {pred_label}\nConfidence: {conf}%\n\nReasoning:\n"
                        for k, v in result.items():
                            report += f"{k}: {v}\n"
                        st.markdown(download_link(report, f"SafeSkin_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"), unsafe_allow_html=True)

                with colB:
                    st.info("Tips:")
                    st.write("- Use clear, close-up dermatoscopic photos when possible.")
                    st.write("- This is a demo — consult a dermatologist for real evaluation.")

        st.markdown('</div>', unsafe_allow_html=True)

    elif choice == "📊 Results Visualization":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Results Visualization")
        hist = st.session_state['history']
        if not hist:
            st.info("No analysis history yet. Upload and analyze an image first.")
        else:
            # Aggregate counts
            from collections import Counter
            ctr = Counter([h['label'] for h in hist])
            labels = list(ctr.keys())
            counts = [ctr[l] for l in labels]

            fig = go.Figure(go.Pie(labels=labels, values=counts, hole=0.45))
            fig.update_layout(title_text="Predicted Class Distribution (history)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("Full history table:")
            rows = []
            for h in hist[::-1]:
                rows.append((h['timestamp'], h['label'], f"{h['confidence']}%"))
            st.table(rows)

        st.markdown('</div>', unsafe_allow_html=True)

    elif choice == "💡 Recommendations":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("General Recommendations & Next Steps")
        st.write("This page gives general next steps based on predicted class. These are educational suggestions only.")

        selected = st.selectbox("Select last prediction (or pick a class)", ["— Select —"] + list(CLASS_MAP.values()))
        if selected != "— Select —":
            # Basic mapped suggestions
            suggestions = {
                'Melanoma': [
                    "Seek urgent dermatology consultation.",
                    "Biopsy is usually recommended for confirmation.",
                    "Avoid sun exposure and document lesion photos."
                ],
                'Basal Cell Carcinoma': [
                    "Often treated surgically; consult dermatologist.",
                    "Regular follow-up and sun protection recommended."
                ],
                'Squamous Cell Carcinoma': [
                    "Requires dermatologist evaluation; may need excision.",
                    "Look for rapid growth, bleeding, or crusting."
                ],
                'Actinic Keratosis': [
                    "Sun-damaged skin — cryotherapy or topical meds are common.",
                    "Regular skin checks recommended."
                ],
                'Vascular Lesion': [
                    "May be benign; if changing, see a clinician.",
                    "Laser therapies may be an option."
                ],
                'Nevus': [
                    "Most are benign; monitor for changes in size/color.",
                    "If changing rapidly, consult a dermatologist."
                ],
                'Pigmented Benign Keratosis': [
                    "Often benign; cosmetic removal possible if bothersome."
                ],
                'Seborrheic Keratosis': [
                    "Benign; removal for irritation or cosmetics."
                ],
                'Dermatofibroma': [
                    "Benign; excision only if symptomatic."
                ]
            }
            recs = suggestions.get(selected, ["Consult a dermatologist for tailored advice."])
            st.markdown("<ul>" + "".join([f"<li>{r}</li>" for r in recs]) + "</ul>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    elif choice == "ℹ️ About App":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("About SafeSkin (Demo)")
        st.write("""
        - **SafeSkin** is an educational demo application that performs *rule-based* analysis of lesion images.
        - **This is NOT a diagnostic tool.** It is meant for prototyping UI/UX and demonstration when a trained model isn't available.
        - The app provides plausible predictions, probability-like outputs, and downloadable reports to simulate a production experience.
        """)
        st.markdown("**Credits & Contact**")
        st.write("- Built as a demo by the SafeSkin team.")
        st.markdown('<div class="footer">Made with ❤️ • Not medical advice • For demonstration only</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer small
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
