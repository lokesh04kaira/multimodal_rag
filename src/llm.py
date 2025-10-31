import os
import re
from typing import List, Optional

def _env(k: str, d: str = "") -> str:
    return os.getenv(k, d)

# ---------- Lazy Gemini init (REST) ----------
_genai = None
def _get_genai():
    global _genai
    if _genai is not None:
        return _genai
    try:
        import google.generativeai as genai
        key = _env("GOOGLE_API_KEY")
        if not key:
            print("GEMINI INIT: missing GOOGLE_API_KEY")
            return None
        genai.configure(api_key=key, transport="rest")
        _genai = genai
        return _genai
    except Exception as e:
        print("GEMINI INIT ERROR:", e)
        return None

# ---------- Embeddings ----------
def _embed_gemini(texts: List[str]) -> Optional[List[List[float]]]:
    genai = _get_genai()
    if genai is None:
        return None
    try:
        # Prefer batch API if present
        if hasattr(genai, "batch_embed_contents"):
            out = genai.batch_embed_contents(
                model=_env("MODEL_GEMINI", "text-embedding-004"),
                contents=[{"parts": [t or ""]} for t in texts],
                task_type="retrieval_document",
            )
            # SDK may return dict or an object with .embeddings
            if isinstance(out, dict) and "embeddings" in out:
                return [e["values"] for e in out["embeddings"]]
            if hasattr(out, "embeddings"):
                return [getattr(e, "values", e) for e in out.embeddings]

        # Fallback: call per item
        embs: List[List[float]] = []
        for t in texts:
            res = genai.embed_content(
                model=_env("MODEL_GEMINI", "text-embedding-004"),
                content={"parts": [t or ""]},
                task_type="retrieval_document",
            )
            if isinstance(res, dict) and "embedding" in res and "values" in res["embedding"]:
                embs.append(res["embedding"]["values"])
            elif hasattr(res, "embedding"):
                val = getattr(res.embedding, "values", None)
                if val:
                    embs.append(val)
        return embs if embs else None
    except Exception as e:
        print("GEMINI EMBED ERROR:", e)
        return None

_local_sbert = None

def embed_texts(texts: List[str]) -> List[List[float]]:
    provider = _env("EMBEDDING_PROVIDER", "gemini").lower()
    texts = [t if isinstance(t, str) else "" for t in texts]

    if provider == "gemini":
        embs = _embed_gemini(texts)
        if not embs:
            raise RuntimeError("Gemini embedding failed (no fallback). Check key/model/network.")
        return embs

    # Local provider (384-dim) with lazy model cache
    global _local_sbert
    if _local_sbert is None:
        from sentence_transformers import SentenceTransformer
        _local_sbert = SentenceTransformer("all-MiniLM-L6-v2")
    arr = _local_sbert.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return arr.tolist()

# ---------- Chat ----------
def _answer_locally(prompt: str, context: str) -> str:
    if not context.strip():
        return "No relevant context found."
    sents = re.split(r'(?<=[.!?])\s+', context.strip())

    # DEDUPE sentences (case/space-insensitive)
    seen, dsents = set(), []
    for s in sents:
        norm = " ".join(s.lower().split())
        if norm and norm not in seen:
            dsents.append(s)
            seen.add(norm)
    sents = dsents

    keys = [w.lower() for w in re.findall(r"[a-zA-Z]+", prompt) if len(w) >= 4]
    if not keys:
        return " ".join(sents[:2])[:600]
    scores = []
    for s in sents:
        sl = s.lower()
        score = sum(sl.count(k) for k in keys)
        scores.append((score, s))
    scores.sort(reverse=True, key=lambda x: x[0])
    best = [s for sc, s in scores[:3] if sc > 0] or sents[:2]
    return " ".join(best)[:800]

def _system_prompt() -> str:
    return (
        "You are a helpful assistant. Answer the user using ONLY the provided context. "
        "Synthesize and summarize—do not copy verbatim unless necessary. "
        "If the answer is not in the context, say you don't know. "
        "Prefer concise, direct answers. If the user asks for entities (e.g., animals mentioned), "
        "list them explicitly."
    )

def chat_rag(prompt: str, context: str) -> str:
    provider = _env("CHAT_PROVIDER", "local").lower()

    # Local extractive fallback
    if provider != "gemini":
        return _answer_locally(prompt, context)

    # Gemini reasoning over retrieved context
    genai = _get_genai()
    if genai is None:
        return _answer_locally(prompt, context)

    try:
        model_name = _env("CHAT_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)
        full = f"{_system_prompt()}\n\nContext:\n{context}\n\nQuestion: {prompt}"
        resp = model.generate_content(full)
        return getattr(resp, "text", "") or _answer_locally(prompt, context)
    except Exception as e:
        print("GEMINI CALL FAILED:", e)
        return _answer_locally(prompt, context)
