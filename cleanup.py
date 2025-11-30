import re

COMMON_REPLACEMENTS = {
    '': '',
    '': '',
    '': '',
    '': '',
    '': ''
}

def simple_replace(text: str) -> str:
    for k, v in COMMON_REPLACEMENTS.items():
        text = text.replace(k, v)
    return text

def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[  \t]{2,}', ' ', text)
    return text.strip()

def fix_ocr_errors(text: str) -> str:
    text = simple_replace(text)
    text = normalize_whitespace(text)
    
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    return text