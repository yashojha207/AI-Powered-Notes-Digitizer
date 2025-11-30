import os
from typing import Optional
from PIL import Image
import pytesseract
import torch

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    _TROCR_AVAILABLE = True
except Exception:
    _TROCR_AVAILABLE = False

# Global vars for loaded model
_processor = None
_model = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# loads TrOCR model and proccessor once and caches them

def load_trocr(local_path: Optional[str] = None):
    global _processor, _model
    if not _TROCR_AVAILABLE:
        raise RuntimeError("transformers or torch not available in this environment")
    if _model is not None and _processor is not None:
        return _processor, _model
    
    #model_name = "microsoft/trocr-base-handwritten"
    model_name = "microsoft/trocr-large-handwritten"

    if local_path and os.path.isdir(local_path):
        model_name = local_path

    _processor = TrOCRProcessor.from_pretrained(model_name)
    _model = VisionEncoderDecoderModel.from_pretrained(model_name)
    _model.to(_DEVICE)
    return _processor, _model

def trocr_ocr(pil_image: Image.Image, chunk_lines: bool = False) -> str:
    global _processor, _model
    if _processor is None or _model is None:
        load_trocr()
    
    pixel_values = _processor(images=pil_image, return_tensors="pt").pixel_values.to(_DEVICE)
    generated_ids = _model.generate(pixel_values, max_length=512)
    text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

# make sure tesseract is installed on system
def tesseract_ocr(pil_image: Image.Image, lang: str = 'eng') -> str:
    return pytesseract.image_to_string(pil_image, lang=lang)

# tries TrOCR, otherwise pytesseract
def ocr_image(pil_image: Image.Image, prefer_trocr: bool = True) -> str:
    if prefer_trocr and _TROCR_AVAILABLE:
        try:
            return trocr_ocr(pil_image)
        except Exception as e:
            print(f"TrOCR failed with: {e}. Falling back to Tesseract")
    # fallback
    return tesseract_ocr(pil_image)
