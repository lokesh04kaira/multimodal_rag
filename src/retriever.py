from typing import Dict, List, Tuple, Optional

from src.indexer import search

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
AUD_EXT = (".mp3", ".wav", ".m4a")
VID_EXT = (".mp4", ".mov", ".mkv")

def _is_image(meta: Dict) -> bool:
    t = str(meta.get("type", "")).lower()
    if t == "image":
        return True
    return str(meta.get("path", "")).lower().endswith(IMG_EXT)

def _is_audio(meta: Dict) -> bool:
    t = str(meta.get("type", "")).lower()
    if t == "audio":
        return True
    return str(meta.get("path", "")).lower().endswith(AUD_EXT)

def _is_video(meta: Dict) -> bool:
    t = str(meta.get("type", "")).lower()
    if t == "video":
        return True
    return str(meta.get("path", "")).lower().endswith(VID_EXT)

def _keep(meta: Dict, where: Optional[Dict]) -> bool:
    if not where:
        return True
    if "source" in where and meta.get("source") != where["source"]:
        return False
    t = where.get("type")
    if t == "image" and not _is_image(meta):
        return False
    if t == "audio" and not _is_audio(meta):
        return False
    if t == "video" and not _is_video(meta):
        return False
    pc = where.get("path_contains")
    if pc and pc.lower() not in str(meta.get("path", "")).lower():
        return False
    uc = where.get("url_contains")
    if uc and uc.lower() not in str(meta.get("url", "")).lower():
        return False
    return True

def _where_pushdown(where: Optional[Dict]) -> Optional[Dict]:
    """
    Convert UI filters to Chroma-compatible exact-match filters where possible.
    (Chroma doesn't support 'contains', so we only forward exact keys.)
    """
    if not where:
        return None
    out: Dict = {}
    if "source" in where:
        out["source"] = where["source"]
    if "type" in where:
        out["type"] = where["type"]
    return out or None

def ask(question: str, top_k: int = 6, where: Optional[Dict] = None) -> Dict:
    top_k = max(1, int(top_k))

    pushdown = _where_pushdown(where)
    raw_hits: List[Tuple[str, Dict]] = search(question, top_k=top_k * 3, where=pushdown)

    hits = []
    for d, m in raw_hits:
        m = m or {}
        if _keep(m, where):
            hits.append((d, m))
            if len(hits) >= top_k:
                break

    context = "\n\n".join(d for d, _ in hits) if hits else ""

    from src.llm import chat_rag
    answer = chat_rag(question, context) if context else "No relevant context found."

    return {"answer": answer, "contexts": hits}
