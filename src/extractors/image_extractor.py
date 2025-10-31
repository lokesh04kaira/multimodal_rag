import os
import re
from typing import List, Tuple

from PIL import Image, ImageOps, ImageFilter
import numpy as np
import pytesseract

from src.utils import clean_text

if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
if os.getenv("TESSDATA_PREFIX"):
    os.environ["TESSDATA_PREFIX"] = os.getenv("TESSDATA_PREFIX")

def _prep(img: Image.Image) -> Image.Image:
    """
    Robust preprocessing:
    - Convert to L (grayscale)
    - Auto-contrast
    - Slight denoise
    - Adaptive threshold (local mean)
    """
    g = img.convert("L")
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.MedianFilter(size=3))

    arr = np.asarray(g, dtype=np.uint8)

    k = 15
    pad = k // 2
    pad_arr = np.pad(arr, pad, mode="edge").astype(np.float32)

    cumsum = pad_arr.cumsum(axis=0).cumsum(axis=1)
    h, w = arr.shape
    sums = (
        cumsum[k:, k:]
        - cumsum[:-k, k:]
        - cumsum[k:, :-k]
        + cumsum[:-k, :-k]
    )
    means = (sums / (k * k)).astype(np.float32)
    means = means[:h, :w]

    bin_img = (arr > (means - 5)).astype(np.uint8) * 255
    return Image.fromarray(bin_img, mode="L")

def _tesseract_pass(img: Image.Image, psm: int) -> str:
    cfg = f"--oem 3 --psm {psm}"
    try:
        return pytesseract.image_to_string(img, lang="eng", config=cfg) or ""
    except Exception:
        return ""

def _try_rotations(img: Image.Image) -> List[Tuple[int, Image.Image]]:
    """Return images rotated by 0, 90, 180, 270 degrees."""
    rots = [0, 90, 180, 270]
    return [(r, img.rotate(r, expand=True)) for r in rots]

def _merge_texts(parts: List[str]) -> str:
    joined = "\n".join(p for p in parts if p and p.strip())
    lines = [l.strip() for l in joined.splitlines()]
    seen, out = set(), []
    for l in lines:
        norm = " ".join(l.split())
        if norm and norm.lower() not in seen:
            out.append(norm)
            seen.add(norm.lower())
    return "\n".join(out).strip()

def extract_image(path: str) -> str:
    """
    OCR an image file. Tries multiple PSMs and rotations; falls back to EasyOCR.
    Returns cleaned multi-line text (keeps digits/symbols).
    """
    img = Image.open(path)
    prepped = _prep(img)

    w, h = prepped.size
    aspect = w / max(1, h)
    likely_single_line = aspect > 6.0

    psm_candidates = [7, 6, 3] if likely_single_line else [6, 3, 4, 7]

    tess_results: List[str] = []
    for deg, rotated in _try_rotations(prepped):
        for psm in psm_candidates:
            txt = _tesseract_pass(rotated, psm)
            if txt and txt.strip():
                tess_results.append(txt)

    merged = _merge_texts(tess_results)

    if not merged:
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            easy_lines = reader.readtext(path, detail=0)
            merged = _merge_texts(easy_lines)
        except Exception:
            merged = ""

    final = clean_text(merged)

    return final
