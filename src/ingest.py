from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# --- text / doc extractors ---
from src.extractors.pdf_extractor import extract_pdf
from src.extractors.docx_extractor import extract_docx
from src.extractors.pptx_extractor import extract_pptx
from src.extractors.md_txt_extractor import extract_md, extract_txt
from src.extractors.image_extractor import extract_image

# --- AV (audio/video) + YouTube (optional) ---
HAVE_AV = True
try:
    from src.extractors.av_extractor import extract_audio, extract_video
except Exception:
    HAVE_AV = False

HAVE_YT = True
try:
    from src.extractors.youtube_extractor import extract_youtube
except Exception:
    HAVE_YT = False

from src.indexer import add_document

# Supported extensions (case-insensitive)
SUPPORTED = {
    ".pdf", ".docx", ".pptx", ".ppt", ".md", ".txt",
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
    ".mp3", ".wav", ".m4a",
    ".mp4", ".mov", ".mkv"
}

# Map extension → modality type for consistent metadata and filtering
EXT_TYPE = {
    ".pdf": "text", ".docx": "text", ".pptx": "text", ".ppt": "text",
    ".md": "text", ".txt": "text",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".bmp": "image", ".tif": "image", ".tiff": "image",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio",
    ".mp4": "video", ".mov": "video", ".mkv": "video",
}

def _ext(path: Union[str, Path]) -> str:
    return Path(path).suffix.lower()

def _type_for_ext(ext: str) -> Optional[str]:
    return EXT_TYPE.get(ext)

def extract_any(path: str) -> str:
    """Detect file type by extension and extract text; returns '' on unsupported/disabled features."""
    p = Path(path)
    ext = p.suffix.lower()
    try:
        if ext == ".pdf":
            return extract_pdf(str(p)) or ""
        if ext == ".docx":
            return extract_docx(str(p)) or ""
        if ext in (".pptx", ".ppt"):
            return extract_pptx(str(p)) or ""
        if ext == ".md":
            return extract_md(str(p)) or ""
        if ext == ".txt":
            return extract_txt(str(p)) or ""
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            return extract_image(str(p)) or ""

        # Audio
        if ext in (".mp3", ".wav", ".m4a"):
            if not HAVE_AV:
                return ""
            return extract_audio(str(p)) or ""

        # Video
        if ext in (".mp4", ".mov", ".mkv"):
            if not HAVE_AV:
                return ""
            return extract_video(str(p)) or ""
    except Exception as e:
        # Let caller decide how to report; just surface as empty
        return f""

    # Unsupported extension
    return ""

def ingest_path(path: str):
    """
    Ingest a file or a folder.
    - Directory: returns dict with totals and per-file results.
    - File: returns per-file result dict.
    """
    p = Path(path)

    if p.is_dir():
        results: List[Dict] = []
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in SUPPORTED:
                results.append(ingest_path(str(f)))
        total_chars = sum(r.get("chars", 0) for r in results)
        ingested = sum(1 for r in results if not r.get("skipped"))
        skipped = [r for r in results if r.get("skipped")]
        return {
            "path": str(p),
            "files_scanned": len(results),
            "files_ingested": ingested,
            "total_chars": total_chars,
            "skipped_count": len(skipped),
            "results": results,
        }

    # File case
    ext = _ext(p)
    if ext not in SUPPORTED or not p.is_file():
        return {"path": str(p), "chars": 0, "skipped": "unsupported or not a file"}

    try:
        text = (extract_any(str(p)) or "").strip()
    except Exception as e:
        return {"path": str(p), "chars": 0, "skipped": f"extract failed: {e}"}

    if not text:
        # Nothing useful extracted (e.g., empty OCR/transcript)
        return {"path": str(p), "chars": 0, "skipped": "no text extracted"}

    meta = {
        "source": "file",
        "path": str(p),
        "name": p.name,
        "ext": ext,
        "type": _type_for_ext(ext) or "text",
    }

    stats = add_document(doc_id=str(p), text=text, meta=meta)
    return {"path": str(p), "chars": len(text), "added_chunks": stats.get("added", 0) if isinstance(stats, dict) else None}

def ingest_youtube(url: str):
    """Download YT audio and ingest the transcript as a doc."""
    if not HAVE_YT:
        return {"youtube": url, "chars": 0, "skipped": "yt modules not available"}

    try:
        text = (extract_youtube(url) or "").strip()
    except Exception as e:
        return {"youtube": url, "chars": 0, "skipped": f"extract failed: {e}"}

    if not text:
        return {"youtube": url, "chars": 0, "skipped": "no text extracted"}

    meta = {"source": "youtube", "url": url, "type": "audio", "ext": ".yt"}
    stats = add_document(doc_id=url, text=text, meta=meta)
    return {"youtube": url, "chars": len(text), "added_chunks": stats.get("added", 0) if isinstance(stats, dict) else None}
