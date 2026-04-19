import streamlit as st
import io
import os
import numpy as np
from PIL import Image

# ── Load secrets into env ─────────────────────────────────────────────────────
if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config( page_title="PixelForge Studio", page_icon="🎨", layout="wide", initial_sidebar_state="collapsed", )

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

:root {
    --bg:       #0f1115;
    --bg-card:  #161a22;
    --bg-card2: #1d2330;
    --accent:   #5b6cff;
    --accent2:  #8b93a7;
    --text:     #e8eaed;
    --muted:    #9aa0aa;
    --border:   rgba(255,255,255,0.06);
    --fs:       1rem;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: var(--fs) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
p, li, label, input, textarea, select {
    font-size: var(--fs) !important;
}
.stApp {
    background: radial-gradient(ellipse at 15% 15%, rgba(124,106,247,0.07) 0%, transparent 55%),
                radial-gradient(ellipse at 85% 85%, rgba(232,121,249,0.05) 0%, transparent 55%),
                var(--bg) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}

/* Hero */
.hero { text-align:center; padding:2.8rem 0 2rem; }
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 5.5rem !important;
    font-weight: 800;
    background: linear-gradient(135deg, #7c6af7 0%, #e879f9 55%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -2px;
    line-height: 1.1;
    display: block !important;
    width: 100% !important;
}
.hero p {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size: 1.05rem !important;
    margin-top: 0.55rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius:12px !important; padding:5px !important;
    border:1px solid var(--border) !important; gap:3px !important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important; border-radius:9px !important;
    color:var(--muted) !important; font-size:1rem !important;
    font-weight:500 !important; padding:8px 20px !important;
    border:none !important; transition:all 0.18s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,var(--accent),var(--accent2)) !important;
    color:#fff !important; box-shadow:0 2px 12px rgba(124,106,247,0.3) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top:1.4rem !important; }

/* Cards */
.card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:14px; padding:1.3rem 1.4rem; margin-bottom:0.9rem;
    transition:border-color 0.2s;
}
.card:hover { border-color:rgba(124,106,247,0.38); }
.result-card {
    background:var(--bg-card2); border:1px solid var(--border);
    border-radius:14px; padding:1.2rem;
}

/* Chip */
.chip {
    display:inline-block; font-size:0.72rem !important; font-weight:700;
    letter-spacing:1.8px; text-transform:uppercase;
    color:var(--accent); margin-bottom:0.55rem;
    -webkit-text-fill-color: var(--accent) !important;
}

/* Main buttons */
.stButton > button {
    background: linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%) !important;
    color:#fff !important; border:none !important; border-radius:9px !important;
    font-family:'DM Sans',sans-serif !important; font-size:1rem !important;
    font-weight:600 !important; padding:0.55rem 1.4rem !important;
    transition:all 0.18s !important;
    box-shadow:0 3px 12px rgba(124,106,247,0.28) !important;
}
.stButton > button:hover {
    transform:translateY(-1px) !important;
    box-shadow:0 5px 18px rgba(124,106,247,0.42) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg,#059669 0%,#0891b2 100%) !important;
    color:#fff !important; border:none !important; border-radius:9px !important;
    font-family:'DM Sans',sans-serif !important; font-size:1rem !important;
    font-weight:600 !important; width:100% !important; padding:0.55rem !important;
    box-shadow:0 3px 12px rgba(5,150,105,0.25) !important;
}

/* Inputs */
.stSelectbox [data-baseweb="select"] > div,
.stTextArea textarea,
.stTextInput input,
.stNumberInput input {
    background:var(--bg-card2) !important; border-color:var(--border) !important;
    border-radius:9px !important; color:var(--text) !important;
    font-family:'DM Sans',sans-serif !important; font-size:1rem !important;
}

/* Labels */
.stTextArea label, .stTextInput label,
.stSelectbox label, .stSlider label,
.stRadio label, .stCheckbox label,
.stNumberInput label, .stFileUploader label {
    font-size: 1rem !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    font-weight: 500 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background:var(--bg-card) !important;
    border:2px dashed var(--border) !important; border-radius:12px !important;
}
[data-testid="stFileUploader"]:hover { border-color:var(--accent) !important; }

/* Slider */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background:var(--accent) !important; border-color:var(--accent) !important;
}

/* Metric */
[data-testid="stMetric"] {
    background:var(--bg-card2) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; padding:0.7rem 0.9rem !important;
}
[data-testid="stMetricLabel"] {
    color:var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size:0.82rem !important;
}
[data-testid="stMetricValue"] {
    color:var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    font-size:1.1rem !important;
}

/* Alerts */
.stSuccess { background:rgba(5,150,105,0.1)  !important; border-color:#059669 !important; border-radius:9px !important; }
.stInfo    { background:rgba(56,189,248,0.08) !important; border-color:#38bdf8 !important; border-radius:9px !important; }
.stWarning { background:rgba(245,158,11,0.09) !important; border-color:#f59e0b !important; border-radius:9px !important; }
.stError   { background:rgba(239,68,68,0.09)  !important; border-color:#ef4444 !important; border-radius:9px !important; }

/* Expander */
.streamlit-expanderHeader {
    background:var(--bg-card2) !important; border-radius:9px !important;
    font-size:1rem !important; color:var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}

/* Misc */
hr { border-color:var(--border) !important; margin:0.8rem 0 !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--accent); border-radius:3px; }
.stSpinner > div > div { border-top-color:var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    save_fmt = "JPEG" if fmt.upper() == "JPG" else fmt.upper()
    if save_fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    img.save(buf, format=save_fmt)
    return buf.getvalue()

def show_pair(original, result, r_label="Result"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<span class="chip">Original</span>', unsafe_allow_html=True)
        st.image(original, use_container_width=True)
    with c2:
        st.markdown(f'<span class="chip">{r_label}</span>', unsafe_allow_html=True)
        st.image(result, use_container_width=True)

def dl_btn(img: Image.Image, filename: str, fmt: str = "PNG"):
    mime = "image/jpeg" if fmt.upper() in ("JPEG","JPG") else f"image/{fmt.lower()}"
    st.download_button(
        "⬇  Download",
        data=pil_to_bytes(img, fmt),
        file_name=filename,
        mime=mime,
        use_container_width=True,
    )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>✦ PixelForge Studio</h1>
    <p>Image processing & AI generation workspace</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🖼  Image Studio", "✨  AI Generator"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE STUDIO
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader(
        "Upload an image — JPG, PNG, BMP, WEBP, TIFF",
        type=["jpg","jpeg","png","bmp","webp","tiff","gif"],
    )

    if uploaded:
        original_image = Image.open(uploaded)
        if "uploaded_name" not in st.session_state or st.session_state["uploaded_name"] != uploaded.name:
            st.session_state["uploaded_name"] = uploaded.name
        for key in ["cropped_img", "crop_source", "crop_left", "crop_right", "crop_top", "crop_bottom"]:
            st.session_state.pop(key, None)
        w, h = original_image.size

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Width",  f"{w}px")
        m2.metric("Height", f"{h}px")
        m3.metric("Mode",   original_image.mode)
        m4.metric("Format", uploaded.name.rsplit(".", 1)[-1].upper())

        st.markdown("---")
        s1, s2, s3, s4, s5 = st.tabs(["🔄 Convert", "🎨 Filters", "✏ Sketch", "📝 OCR", "✂ Crop"])

        # ── CONVERT ──────────────────────────────────────────────────────────
        with s1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                target_fmt = st.selectbox("Convert to", ["PNG","JPEG","BMP","WEBP","TIFF","ICO"])
            with col2:
                quality = st.slider("JPEG Quality", 10, 100, 90) if target_fmt == "JPEG" else 95
            if st.button("Convert", key="convert_btn"):
                from utils.format_convert import convert_image
                with st.spinner("Converting…"):
                    converted = convert_image(original_image.copy(), target_fmt)
                st.success(f"Converted to {target_fmt}")
                show_pair(original_image, converted, target_fmt)
                ext = "jpg" if target_fmt == "JPEG" else target_fmt.lower()
                dl_btn(converted, f"converted.{ext}", target_fmt)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── FILTERS ──────────────────────────────────────────────────────────
        with s2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            FILTERS = {
                "Grayscale":"grayscale", "Gaussian Blur":"blur",
                "Invert":"invert",       "Sepia":"sepia",
                "Edge Detect":"edge",    "Sharpen":"sharpen",
                "Emboss":"emboss",       "Brightness":"brightness",
            }
            col1, col2 = st.columns(2)
            with col1:
                filter_label = st.selectbox("Filter", list(FILTERS.keys()))
            filter_type = FILTERS[filter_label]
            with col2:
                intensity = 15
                if filter_type in ("blur","brightness"):
                    intensity = st.slider("Intensity", 1, 51, 15, step=2)
            if st.button("Apply", key="filter_btn"):
                from utils.image_filters import apply_filter
                with st.spinner("Applying…"):
                    result = apply_filter(original_image.copy(), filter_type, intensity=intensity)
                show_pair(original_image, result, filter_label)
                dl_btn(result, f"{filter_type}.png", "PNG")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── SKETCH ───────────────────────────────────────────────────────────
        with s3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                sketch_mode = st.radio("Style", ["Classic B&W","Coloured"], horizontal=True)
            with col2:
                blur_strength = st.slider("Stroke Softness", 5, 51, 21, step=2)
            if st.button("Generate Sketch", key="sketch_btn"):
                import cv2
                with st.spinner("Sketching…"):
                    img_np  = np.array(original_image.convert("RGB"))
                    gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    inv     = cv2.bitwise_not(gray)
                    blurred = cv2.GaussianBlur(inv, (blur_strength, blur_strength), 0)
                    sketch  = cv2.divide(gray, 255 - blurred, scale=256)
                    if sketch_mode == "Coloured":
                        base   = Image.fromarray(sketch).convert("RGB")
                        colour = original_image.convert("RGB").resize(base.size)
                        result = Image.blend(base, colour, alpha=0.3)
                    else:
                        result = Image.fromarray(sketch)
                show_pair(original_image, result, "Sketch")
                dl_btn(result, "sketch.png", "PNG")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── OCR ──────────────────────────────────────────────────────────────
        with s4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            preprocess = st.checkbox("Preprocess before reading", value=True)
            if st.button("Extract Text", key="ocr_btn"):
                try:
                    import pytesseract, cv2
                    with st.spinner("Reading…"):
                        img_np = np.array(original_image.convert("RGB"))
                        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                        if preprocess:
                            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                        text = pytesseract.image_to_string(gray)
                    if text.strip():
                        st.success(f"{len(text.split())} words extracted")
                        st.text_area("Extracted text", text, height=240)
                        st.download_button("⬇  Download .txt", data=text,
                            file_name="extracted.txt", mime="text/plain", use_container_width=True)
                    else:
                        st.warning("No text detected. Try a clearer image.")
                except Exception as e:
                    st.error(f"OCR failed: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── CROP ─────────────────────────────────────────────────────────────
        with s5:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(original_image, caption=f"{w} × {h}", use_container_width=True)
            st.markdown("---")
        
            c1, c2, c3, c4 = st.columns(4)
            left   = c1.number_input("Left",   min_value=0, max_value=w-1, value=0, step=1)
            right  = c2.number_input("Right",  min_value=1, max_value=w,   value=w, step=1)
            top    = c3.number_input("Top",    min_value=0, max_value=h-1, value=0, step=1)
            bottom = c4.number_input("Bottom", min_value=1, max_value=h,   value=h, step=1)
        
            if st.button("Crop", key="crop_btn"):
                if left >= right or top >= bottom:
                    st.error("Invalid bounds.")
                else:
                    with st.spinner("Cropping…"):
                        cropped = original_image.convert("RGB").crop((int(left), int(top), int(right), int(bottom)))
                    st.session_state["cropped_img"]  = cropped
                    st.session_state["crop_source"]  = original_image.copy()
        
            if "cropped_img" in st.session_state:
                cw, ch = st.session_state["cropped_img"].size
                st.success(f"Cropped to {cw} × {ch}px")
                show_pair(st.session_state["crop_source"], st.session_state["cropped_img"], "Cropped")
                dl_btn(st.session_state["cropped_img"], "cropped.png", "PNG")
        
            st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI GENERATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab2:

    EXAMPLES = [
        "A misty Japanese forest temple at dawn, golden light, cinematic",
        "Astronaut riding a horse on Mars, photorealistic, epic scale",
        "Portrait of a cyberpunk samurai, neon rain, highly detailed",
        "Underwater city of Atlantis, bioluminescent creatures, fantasy art",
        "A cozy library inside an ancient tree, warm lighting, magical realism",
        "Aerial view of a futuristic city at night, glowing rivers, dramatic fog",
    ]

    STYLE_SUFFIXES = {
        "None":           "",
        "Photorealistic": ", photorealistic, 8k, RAW photo, highly detailed",
        "Oil Painting":   ", oil painting, thick brushstrokes, canvas texture",
        "Watercolor":     ", watercolor illustration, soft edges, paper texture",
        "Anime":          ", anime art style, vibrant colors, Studio Ghibli",
        "Concept Art":    ", concept art, matte painting, ArtStation trending",
        "Digital Art":    ", digital art, sharp edges, vibrant",
        "Sketch":         ", pencil sketch, charcoal drawing, hand-drawn",
    }

    if "prompt_seed" not in st.session_state:
        st.session_state["prompt_seed"] = ""

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        prompt = st.text_area(
            "Prompt",
            value=st.session_state["prompt_seed"],
            placeholder="Describe the image you want to create…",
            height=120,
        )

        style_preset = st.selectbox("Style", list(STYLE_SUFFIXES.keys()))

        if st.button("🪄  Generate", key="gen_btn", use_container_width=True):
            if not prompt.strip():
                st.warning("Enter a prompt first.")
            else:
                final_prompt = prompt + STYLE_SUFFIXES.get(style_preset, "")
                from utils.text_to_image import generate_image_from_text
                with st.spinner("Generating… this may take 20–40 seconds"):
                    img, err = generate_image_from_text(final_prompt)
                if err:
                    st.error(f"Failed: {err}")
                else:
                    st.session_state["gen_img"]    = img
                    st.session_state["gen_prompt"] = prompt

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<span class="chip" style="margin-top:0.6rem;display:block;">💡 Suggestions — click to use</span>', unsafe_allow_html=True)
        for ex in EXAMPLES:
            if st.button(ex, key=f"sug_{ex[:28]}"):
                st.session_state["prompt_seed"] = ex
                st.rerun()

    with col_right:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if "gen_img" in st.session_state:
            gen_img = st.session_state["gen_img"]
            st.markdown('<span class="chip">Result</span>', unsafe_allow_html=True)
            st.image(gen_img, use_container_width=True)
            st.caption(st.session_state.get("gen_prompt", ""))
            st.markdown("---")
            dl_btn(gen_img, "generated.png",  "PNG")
            c1, c2 = st.columns(2)
            with c1: dl_btn(gen_img, "generated.jpg",  "JPEG")
            with c2: dl_btn(gen_img, "generated.webp", "WEBP")
        else:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;min-height:360px;text-align:center;">
                <div style="font-size:3.5rem;margin-bottom:0.8rem;opacity:0.3">✨</div>
                <div style="font-size:1rem;color:#b0b0cc;font-weight:500;-webkit-text-fill-color:#b0b0cc;">Your image will appear here</div>
                <div style="font-size:0.9rem;margin-top:0.3rem;color:#7878a0;-webkit-text-fill-color:#7878a0;">Write a prompt and hit Generate</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.82rem;padding:0.4rem 0 1rem;color:#44445a;-webkit-text-fill-color:#44445a;">
    ✦ PixelForge Studio · Streamlit · OpenCV · HuggingFace
</div>
""", unsafe_allow_html=True)
