import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=False)

import chromadb
from src.llm import embed_texts

CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
COLL_NAME = os.getenv("COLLECTION_NAME", f"multimodal-{PROVIDER}")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(COLL_NAME)
print(f"[Chroma] path={CHROMA_DIR} collection={COLL_NAME}")

def chunk(text: str) -> List[str]:
    text = text or ""
    if not text:
        return []
    out, i, n = [], 0, len(text)
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    while i < n:
        out.append(text[i:i + CHUNK_SIZE])
        i += step
    return out

def add_document(doc_id: str, text: str, meta: Dict) -> Dict[str, int]:
    """
    Adds a document by chunking + embedding. Uses upsert to avoid duplicate-ID errors.
    Returns simple stats dict.
    """
    text = (text or "").strip()
    if not text:
        print(f"[Index] Skipping empty doc: {doc_id}")
        return {"chunks": 0, "added": 0}

    chunks = chunk(text)
    if not chunks:
        print(f"[Index] No chunks after processing: {doc_id}")
        return {"chunks": 0, "added": 0}

    embeds = embed_texts(chunks) or []
    if not embeds or len(embeds) != len(chunks):
        print(f"[Index] Embedding failure: got {len(embeds)} for {len(chunks)} chunks")
        return {"chunks": len(chunks), "added": 0}

    print(f"[Embeddings] provider={PROVIDER} dim={len(embeds[0])} n={len(embeds)}")

    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [{**(meta or {}), "chunk": i} for i in range(len(chunks))]

    # upsert prevents duplicate-id exceptions on re-ingest
    collection.upsert(documents=chunks, embeddings=embeds, ids=ids, metadatas=metadatas)
    return {"chunks": len(chunks), "added": len(chunks)}

def search(query: str, top_k: int = 6, where: Optional[Dict] = None) -> List[Tuple[str, Dict]]:
    """
    Return list of (document_text, metadata) using our own query embeddings.
    Optional `where` supports Chroma metadata filtering, e.g. {"type":"audio"}.
    """
    q_embs = embed_texts([query]) or []
    if not q_embs:
        return []
    q_emb = q_embs[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=max(1, int(top_k)),
        where=where or {}
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return list(zip(docs, metas))

def delete_by_prefix(doc_id_prefix: str) -> int:
    """
    Delete all chunks whose id starts with <doc_id_prefix>- .
    Returns count deleted (best-effort).
    """
    
    try:
        
        res = collection.get(include=["metadatas"])
        ids = res.get("ids", [])
        target_ids = [i for i in ids if i.startswith(f"{doc_id_prefix}-")]
        if target_ids:
            collection.delete(ids=target_ids)
        return len(target_ids)
    except Exception as e:
        print(f"[Delete] Failed for prefix {doc_id_prefix}: {e}")
        return 0
