import cv2
import numpy as np
from PIL import Image


def apply_filter(image, filter_type, **kwargs):
    """
    image: PIL Image
    filter_type: string
    kwargs: optional parameters like intensity
    """
    img = np.array(image)

    # Ensure uint8 format
    img = img.astype(np.uint8)

    # ---------- FILTERS ----------

    if filter_type == "grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    elif filter_type == "blur":
        intensity = int(kwargs.get('intensity', 15))
        intensity = max(3, min(intensity, 51))

        if intensity % 2 == 0:
            intensity += 1

        img = cv2.GaussianBlur(img, (intensity, intensity), 0)

    elif filter_type == "invert":
        img = cv2.bitwise_not(img)

    elif filter_type == "sepia":
    # Ensure correct dtype
        img = img.astype(np.uint8)
    
        # Handle grayscale → convert to RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
        # Handle RGBA (4 channels) → convert to RGB
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
        # Final safety check
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Invalid image shape for sepia: {img.shape}")
    
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ], dtype=np.float32)
    
        img = cv2.transform(img, kernel)
        img = np.clip(img, 0, 255).astype(np.uint8)

    elif filter_type == "edge":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.Canny(gray, 100, 200)

    elif filter_type == "sharpen":
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        img = cv2.filter2D(img, -1, kernel)

    elif filter_type == "emboss":
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [0,   1, 2]])

        img = cv2.filter2D(img, -1, kernel)

    elif filter_type == "brightness":
        intensity = int(kwargs.get('intensity', 15))
        intensity = max(-50, min(intensity, 50))

        img = cv2.convertScaleAbs(img, alpha=1, beta=intensity)

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # ---------- RETURN IMAGE ----------

    # If grayscale or edge → 2D image
    if len(img.shape) == 2:
        return Image.fromarray(img)

    return Image.fromarray(img)


def pencil_sketch(image):
    """
    Pencil sketch effect
    """
    img = np.array(image)
    img = img.astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    return Image.fromarray(sketch)
