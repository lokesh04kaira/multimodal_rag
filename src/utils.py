import os
from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _resolve_dir(env_key: str, default_rel: str) -> Path:
    v = os.getenv(env_key, "").strip()
    p = Path(v) if v else (PROJECT_ROOT / default_rel)
    return p.resolve()

UPLOAD_DIR: Path = _resolve_dir("UPLOAD_DIR", "data/uploads")
CHROMA_DIR: Path = _resolve_dir("CHROMA_DIR", "data/chroma")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

def safe_filename(name: str) -> str:
    """
    Minimal filename sanitizer: strips path components and replaces separators.
    """
    name = os.path.basename(name or "")
    return name.replace("\\", "_").replace("/", "_").strip() or "file"

def clean_text(s: str) -> str:
    return "\n".join(line.strip() for line in (s or "").splitlines() if line.strip())
