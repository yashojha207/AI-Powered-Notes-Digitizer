import streamlit as st
from PIL import Image
import io
from gpt_ocr import run_gpt_ocr

from preprocess import load_image, to_grayscale, deskew, denoise, binarize, resize_keep_aspect
from cleanup import fix_ocr_errors

st.set_page_config(page_title="AI-Powered Notes Digitizer", layout="centered")

st.title("AI-Powered Notes Digitizer")
st.markdown("Upload a photo of your handwritten notes. The app will digitize the text for you.")

uploaded_file = st.file_uploader("Upload an image or PDF page", type=["jpg", "jpeg", "png", "pdf"])

def file_to_pil(file_obj):
    if file_obj is None:
        return None

    filename = file_obj.name.lower()
    if filename.endswith("pdf"):
        try:
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(file_obj.read(), dpi=200)
            return pages[0].convert("RGB")
        except Exception as e:
            st.error("PDF support requires pdf2image + poppler installed. Please upload an image instead.")
            return None
    else:
        return Image.open(file_obj).convert("RGB")

if uploaded_file:
    img_cv = load_image(uploaded_file)

    if img_cv is None:
        st.error("Could not read image. Make sure it is a valid image file (png/jpg).")
        st.stop()

    # manual image orientation
    st.subheader("Image Orientation")

    rotation = st.selectbox(
        "Rotate image",
        [0, 90, 180, 270],
        index=0
    )

    import cv2

    def rotate_image(img, angle):
        if angle == 0:
            return img
        elif angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img_cv = rotate_image(img_cv, rotation)

    st.image(
        Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)),
        caption="Rotated Image",
        use_column_width=True
    )
    

    st.subheader("Preprocessing options")
    st.write("Preview and tweak preprocessing to improve OCR accuracy")

    preview = st.checkbox("Show preprocessed preview", value=True)
    desired_height = st.slider("Resize height (pixels)", 400, 1600, 900)
    do_deskew = st.checkbox("Deskew image", value=True)
    do_denoise = st.checkbox("Denoise", value=True)
    do_binarize = st.checkbox("Binarize (threshold)", value=True)

    img_rs = resize_keep_aspect(img_cv, height=desired_height)
    img_gray = to_grayscale(img_rs)

    if do_deskew:
        img_gray = deskew(img_gray)

    if do_denoise:
        img_gray = denoise(img_gray)

    if do_binarize:
        img_proc = binarize(img_gray)
    else:
        img_proc = img_gray

    if preview:
        st.image(Image.fromarray(img_proc), caption="Preprocessed image (for OCR)", use_column_width=True)

    pil_img = Image.fromarray(img_proc).convert("RGB")

    if st.button("Run OCR"):
        with st.spinner("Running OCR..."):
            try:
                raw_text = run_gpt_ocr(pil_img)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                raw_text = ""

            cleaned = fix_ocr_errors(raw_text)

        st.subheader("Digitized Text (editable)")
        text_area = st.text_area("You can edit or copy this text:", cleaned, height=300)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Copy to clipboard (browser)"):
                st.write("Select all text in the box and copy (Ctrl+C / Cmd+C).")

        with col2:
            if st.button("Download .txt"):
                b = io.BytesIO()
                b.write(text_area.encode("utf-8"))
                b.seek(0)
                st.download_button(
                    "Download digitized text",
                    data=b,
                    file_name="digitized_notes.txt",
                    mime="text/plain"
                )

else:
    st.info("No file uploaded yet. Try uploading a clear photo of a single-page note.")
