from pathlib import Path
from src.utils import clean_text

def extract_pdf(path: str) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    text = ""
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(str(p)) or ""
    except Exception:
        try:
            import PyPDF2
            with open(p, "rb") as f:
                r = PyPDF2.PdfReader(f)
                parts = []
                for i in range(len(r.pages)):
                    try:
                        parts.append(r.pages[i].extract_text() or "")
                    except Exception:
                        parts.append("")
                text = "\n".join(parts)
        except Exception:
            text = ""
    return clean_text(text)
