# ✦ PixelForge Studio

A beautiful Streamlit-based image processing & AI generation application.

## Features

### 🖼 Module 1 — Image Studio
| Tool | Description |
|------|-------------|
| **Format Conversion** | Convert between JPG, PNG, BMP, WEBP, TIFF, ICO |
| **Image Filters** | Grayscale, Blur, Invert, Sepia, Edge Detect, Sharpen, Emboss, Brightness |
| **Pencil Sketch** | Classic B&W and Coloured sketch styles with adjustable blur |
| **OCR Text Extraction** | Extract text from images using Tesseract with preprocessing |
| **Crop Tool** | Free-form crop with pixel inputs + 1:1, 16:9, 4:3, 3:2 presets |

### ✨ Module 2 — AI Image Generator
- Text → Image using Stable Diffusion XL via HuggingFace Inference API
- Style presets (Photorealistic, Anime, Oil Painting, Watercolor, etc.)
- Negative prompt support
- Download in PNG, JPEG, WEBP

---

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (for text extraction)
```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. HuggingFace API Token
- Get a free token at https://huggingface.co/settings/tokens
- Enter it in the sidebar when the app opens
- Required only for the AI Image Generator module

---

## Project Structure
```
streamlit_app/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── README.md
└── utils/
    ├── __init__.py
    ├── format_convert.py   # PIL-based format conversion
    ├── image_filters.py    # OpenCV filters
    ├── ocr_utils.py        # Tesseract OCR
    └── text_to_image.py    # HuggingFace SDXL API
```
