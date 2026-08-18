import os
import hashlib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 1. PAGE CONFIG

st.set_page_config(
    page_title="CropDisease Prediction AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2.  CSS

st.markdown(
    """
<style>
:root {
    --bg-1: #ffffff;
    --bg-2: #f8fafc;
    --bg-3: #f0fdf4;
    --card: rgba(255,255,255,0.95);
    --border: rgba(226, 232, 240, 0.8);
    --accent: #059669;
    --accent-soft: rgba(5, 150, 105, 0.08);
    --text: #1e293b;
    --muted: #64748b;
    --danger: #dc2626;
    --danger-soft: rgba(220, 38, 38, 0.08);
}
.stApp {
    background: linear-gradient(180deg, var(--bg-3) 0%, var(--bg-1) 25%, var(--bg-2) 100%);
    color: var(--text);
}

/* Clean header layout without background crop/cuts */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    z-index: 100 !important;
}

#MainMenu, footer {
    display: none !important;
}

/* Sidebar toggle button placement */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    color: #2E7D32 !important;
    top: 10px !important;
    left: 10px !important;
}

/* Laptop spacing */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 0rem !important;
    max-width: 1200px !important;
}

/* Mobile spacing */
@media (max-width: 768px) {
    .block-container {
        padding-top: 0.5rem !important;
    }
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, #f8fafc);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text); }
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small { color: var(--muted); }

/* Cards */
.card {
    background: var(--card);
    padding: 22px 24px;
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: 0 2px 12px rgba(0,0,0,0.03);
}

/* Metrics */
.metric {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: left;
    box-shadow: 0 2px 12px rgba(0,0,0,0.03);
    transition: all 0.2s ease;
}
.metric:hover {
    border-color: var(--accent);
    box-shadow: 0 4px 20px rgba(5, 150, 105, 0.08);
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
}
.metric-label {
    color: var(--muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Result */
.result {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 28px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.04);
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-ok {
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid rgba(5,150,105,0.2);
}
.badge-bad {
    background: var(--danger-soft);
    color: var(--danger);
    border: 1px solid rgba(220,38,38,0.2);
}
.badge-demo {
    background: rgba(0,0,0,0.04);
    color: var(--muted);
    border: 1px solid var(--border);
    margin-left: 8px;
}
.diagnosis {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 14px 0 6px 0;
    letter-spacing: -0.01em;
    color: var(--text);
}
.confidence {
    font-size: 3rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.02em;
    line-height: 1;
    margin-top: 10px;
}
.confidence-label {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 4px;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
    border: none;
    padding: 10px 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(5, 150, 105, 0.25);
}
.stDownloadButton > button {
    border-radius: 12px;
    font-weight: 600;
    background: white;
    color: var(--accent);
    border: 1px solid var(--border);
    padding: 10px 20px;
}
.stDownloadButton > button:hover {
    border-color: var(--accent);
    background: var(--accent-soft);
}

/* Progress */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #059669, #34d399);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {
    background: var(--card);
    border-radius: 10px;
    padding: 8px 16px;
    border: 1px solid var(--border);
    color: var(--text);
}
.stTabs [aria-selected="true"] {
    border-color: rgba(5,150,105,0.4);
    color: var(--accent);
    background: var(--accent-soft);
}


/* Class list item */
.class-item {
    padding: 6px 12px;
    border-radius: 8px;
    background: rgba(5,150,105,0.04);
    margin-bottom: 6px;
    font-size: 0.9rem;
    color: var(--text);
    border: 1px solid rgba(5,150,105,0.08);
}

/* Arch info card */
.arch-info-card {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border: 1px solid rgba(5,150,105,0.15);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.arch-info-title {
    font-weight: 700;
    color: #059669;
    font-size: 0.95rem;
    margin-bottom: 4px;
}
.arch-info-desc {
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.5;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 2px dashed #2E7D32 !important;
    border-radius: 20px !important;
    background-color: #FAFCFA !important;
    box-shadow: 0 10px 25px -5px rgba(46, 125, 50, 0.12) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #1B5E20 !important;
    box-shadow: 0 14px 30px -5px rgba(46, 125, 50, 0.22) !important;
}

/* Center Tabs Inside Box */
[data-testid="stTabs"] {
    display: flex !important;
    justify-content: center !important;
    max-width: 280px !important;
    margin: 10px auto !important;
}

/* Center Upload Button Inside Box */
[data-testid="stTabPanel"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
}

[data-testid="stFileUploaderDropzone"] {
    border: none !important;
    background: transparent !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 8px 0 !important;
}

/*  Upload Button */
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 10px 38px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    box-shadow: 0 4px 14px rgba(46, 125, 50, 0.25) !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg, #1B5E20 0%, #0D3B10 100%) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# 3. MODEL + SETTINGS

MODEL_PATH = os.environ.get("CROPGUARD_MODEL_PATH", "model/crop_disease_model_cnn.keras")
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]


# 4. LOAD MODEL

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.session_state["_model_error"] = str(e)
        return None

model = load_model()
DEMO_MODE = model is None


# 5. HELPERS

def format_class_name(name: str) -> str:
    return name.replace("___", " — ").replace("_", " ").replace(",", "").title()


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = ImageOps.fit(img, IMG_SIZE, method=Image.Resampling.LANCZOS)
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)


def _mock_predict(img: Image.Image) -> np.ndarray:
    img_small = img.convert("RGB").resize((32, 32))
    digest = hashlib.sha256(np.array(img_small).tobytes()).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    logits = rng.normal(loc=0.0, scale=1.0, size=len(CLASS_NAMES))
    peak = seed % len(CLASS_NAMES)
    logits[peak] += rng.uniform(3.5, 6.0)
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def predict_disease(img: Image.Image):
    if DEMO_MODE:
        preds = _mock_predict(img)
    else:
        preds = model.predict(preprocess_image(img), verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx] * 100), preds


# 8. HERO

demo_badge = (
    '<span class="badge badge-demo">Demo Mode</span>' if DEMO_MODE else ""
)

st.markdown(
    f"""
    <div style="text-align: center; padding: 10px 0px 15px 0px;">
        <h1 style="color: #2E7D32; font-size: 3rem; margin-bottom: 5px;">
            🌿 Crop Disease Prediction AI {demo_badge}
        </h1>
        <p style="font-size: 1.2rem; color: #555555;">
            Upload a leaf image. Get an instant diagnosis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
# SIDEBAR - ABOUT & SUPPORTED CROPS

with st.sidebar:
    st.markdown("## 🌿 AI Model")
    st.write("This Ai Model is trained to accurate predict different crops and their diseases with approx 94% validation accuracy ")
    
    st.markdown("---")
    
    st.markdown("### 🌾 Available Crops List")
    
    # Extract unique crop names from CLASS_NAMES
    try:
        raw_crops = [c.split('_')[0].replace('_', ' ').title() for c in CLASS_NAMES]
        unique_crops = sorted(list(set(raw_crops)))
    except NameError:
        unique_crops = ["Apple", "Blueberry", "Cherry", "Corn", "Grape", "Peach", "Pepper", "Potato", "Strawberry", "Tomato"]

    # Display Crop List
    for crop in unique_crops:
        st.markdown(f"- 🌱 *{crop}*")
        
# 10. IMAGE INPUT

with st.container(border=True):
    # Header & Icon Inside Box
    st.markdown("""
        <div style="text-align: center;">
            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#2E7D32" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M16 16l-4-4-4 4"/>
                <path d="M12 12v9"/>
                <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
            </svg>
            <h3 style="margin: 10px 0 4px 0; color: #111827; font-weight: 700; font-size: 22px;">Upload Plant Image</h3>
            <p style="color: #6B7280; margin: 0 0 10px 0; font-size: 14px;">Drag & drop image here or capture using camera</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs Inside Box
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload Plant Image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

    with tab2:
        camera_file = st.camera_input(
            "Take Photo",
            label_visibility="collapsed"
        )

    active_file = uploaded_file or camera_file


# 11. PREDICTION
# =========================================================
if active_file is not None:
    image = Image.open(active_file).convert("RGB")
    st.write("")
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.image(image, use_container_width=True)
        st.caption(f"{image.width} × {image.height} px")

    with right:
        if st.button("Analyze Leaf", use_container_width=True, key="analyze-btn"):
            with st.spinner("Analyzing..."):
                disease, confidence, predictions = predict_disease(image)

            formatted = format_class_name(disease)
            is_healthy = "healthy" in disease.lower()

            badge = (
                '<span class="badge badge-ok">Healthy</span>'
                if is_healthy
                else '<span class="badge badge-bad">Disease Detected</span>'
            )

            st.markdown(
                f"""
                <div class="result" data-testid="result-card">
                    {badge}
                    <div class="diagnosis">{formatted}</div>
                    <div class="confidence">{confidence:.1f}%</div>
                    <div class="confidence-label">Model confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

           
