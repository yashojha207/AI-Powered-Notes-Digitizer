import cv2
import numpy as np
from PIL import Image

# Load image from file paths or uploaded file objects from Streamlit
def load_image(path):
    try:
        if hasattr(path, "read"):
            # Read bytes from uploaded file
            arr = np.frombuffer(path.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(path)
        return img
    except Exception as e:
        print("Error loading image:", e)
        return None

# Convert to grayscale
def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize image while keeping aspect ratio
def resize_keep_aspect(img, height=800):
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = height / float(h)
    new_w = int(w * scale)
    resized = cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)
    return resized

# Denoise grayscale image
def denoise(img_gray):
    return cv2.fastNlMeansDenoising(img_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

# Binarize grayscale image to improve OCR
def binarize(img_gray):
    return cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )

# Deskew image — only small angle corrections to avoid flipping
def deskew(img_gray):
    coords = np.column_stack(np.where(img_gray < 255))
    if coords.size == 0:
        return img_gray

    angle = cv2.minAreaRect(coords)[-1]

    # Limit to small angles, avoid 90° flips
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > 25:
        # too large → ignore
        return img_gray

    (h, w) = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img_gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated
