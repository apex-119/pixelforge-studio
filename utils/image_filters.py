import cv2
import numpy as np
from PIL import Image

def apply_filter(image, filter_type, **kwargs):
    img = np.array(image)

    # RGB → BGR
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if filter_type == "grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif filter_type == "blur":
        intensity = int(kwargs.get('intensity', 15))
        intensity = max(3, min(intensity, 51))
        if intensity % 2 == 0:
            intensity += 1
        img = cv2.GaussianBlur(img, (intensity, intensity), 0)

    elif filter_type == "invert":
        img = cv2.bitwise_not(img)

    elif filter_type == "sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        img = cv2.transform(img, kernel)
        img = np.clip(img, 0, 255).astype(np.uint8)

    elif filter_type == "edge":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Canny(gray, 100, 200)

    elif filter_type == "sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

    elif filter_type == "emboss":
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        img = cv2.filter2D(img, -1, kernel)

    elif filter_type == "brightness":
        intensity = int(kwargs.get('intensity', 15))
        img = cv2.convertScaleAbs(img, alpha=1, beta=intensity)

    # Convert back
    if len(img.shape) == 2:
        return Image.fromarray(img)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pencil_sketch(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return Image.fromarray(sketch)
