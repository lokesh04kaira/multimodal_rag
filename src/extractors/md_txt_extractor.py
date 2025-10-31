from pathlib import Path
from src.utils import clean_text

def extract_md(path: str) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            txt = p.read_text(encoding="utf-16", errors="ignore")
        except Exception:
            txt = ""
    return clean_text(txt)

def extract_txt(path: str) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            txt = p.read_text(encoding="utf-16", errors="ignore")
        except Exception:
            txt = ""
    return clean_text(txt)
