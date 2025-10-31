
import os, sys
from pathlib import Path

APP_DIR = os.path.dirname(os.path.abspath(__file__))         
PROJECT_ROOT = os.path.dirname(APP_DIR)                      
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------------------------------------------------

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
AUD_EXT = (".mp3", ".wav", ".m4a")
VID_EXT = (".mp4", ".mov", ".mkv")

def _is_image(meta: dict) -> bool:
    return str(meta.get("path", "")).lower().endswith(IMG_EXT)

def _is_audio(meta: dict) -> bool:
    return str(meta.get("path", "")).lower().endswith(AUD_EXT)

def _is_video(meta: dict) -> bool:
    return str(meta.get("path", "")).lower().endswith(VID_EXT)

from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st

from src.ingest import ingest_path, extract_any, ingest_youtube
from src.retriever import ask
from src.utils import UPLOAD_DIR

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# ---------------- UI ----------------
st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("📚 Multimodal Data Processing System")

with st.sidebar:
    st.header("Upload / Ingest")
    up = st.file_uploader(
        "Upload files (txt, md, pdf, docx, pptx, jpg, png, mp3, mp4, etc.)",
        accept_multiple_files=True
    )
    if up:
        for f in up:
            safe_name = os.path.basename(f.name).replace("\\", "_").replace("/", "_")
            dest = Path(UPLOAD_DIR) / safe_name

            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            st.caption(f"Saved to: {dest}")

            with st.spinner(f"Previewing extraction for {safe_name}..."):
                try:
                    preview = extract_any(str(dest))
                except Exception as e:
                    preview = f"[Error extracting: {e}]"
                st.write("Extracted preview:", (str(preview)[:200] if preview else "[empty]"))

            with st.spinner(f"Indexing {safe_name}..."):
                try:
                    ingest_stats = ingest_path(str(dest))
                    st.success(f"Ingested: {ingest_stats}")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.divider()
    st.subheader("Ingest YouTube (audio only)")
    yt_url = st.text_input("Paste a YouTube URL")
    if st.button("Ingest YouTube") and yt_url:
        with st.spinner("Downloading & transcribing..."):
            try:
                stats = ingest_youtube(yt_url.strip())
                st.success(f"Ingested: {stats}")
            except Exception as e:
                st.error(f"YouTube ingest failed: {e}")

st.divider()
st.subheader("Scope")

scope = st.selectbox(
    "Limit search to…",
    [
        "All",
        "Only audio",
        "Only video",
        "Only YouTube links",
        "Only images",
        "Only this filename…",
    ],
)

selected_file = None
if scope == "Only this filename…":
    files = sorted([p.name for p in Path(UPLOAD_DIR).glob("*")])
    selected_file = st.selectbox("Choose a file", files) if files else None

st.subheader("Answering model")
ans_mode = st.radio("Use model for answering:", ["Gemini", "Local"], index=0)
os.environ["CHAT_PROVIDER"] = "gemini" if ans_mode == "Gemini" else "local"

st.divider()
q = st.text_input("Ask a question about your knowledge base")
ask_clicked = st.button("Ask")

if ask_clicked and q:
    where = None
    if scope == "Only YouTube links":
        where = {"source": "youtube"}
    elif scope == "Only images":
        where = {"type": "image"}
    elif scope == "Only audio":
        where = {"type": "audio"}
    elif scope == "Only video":
        where = {"type": "video"}
    elif scope == "Only this filename…" and selected_file:
        where = {"path_contains": selected_file}

    with st.spinner("Thinking..."):
        try:
            res = ask(q, where=where)  
        except Exception as e:
            res = {"answer": f"[Error generating answer: {e}]", "contexts": []}

    ctxs = res.get("contexts", []) or []
    if scope == "Only images":
        ctxs = [(d, m) for (d, m) in ctxs if isinstance(m, dict) and _is_image(m)]
    elif scope == "Only audio":
        ctxs = [(d, m) for (d, m) in ctxs if isinstance(m, dict) and _is_audio(m)]
    elif scope == "Only video":
        ctxs = [(d, m) for (d, m) in ctxs if isinstance(m, dict) and _is_video(m)]
    elif scope == "Only this filename…" and selected_file:
        ctxs = [(d, m) for (d, m) in ctxs if isinstance(m, dict) and selected_file.lower() in str(m.get("path", "")).lower()]

    if ctxs:
        filtered_context = "\n\n".join(str(d) for d, _ in ctxs)
        from src.llm import chat_rag
        try:
            res["answer"] = chat_rag(q, filtered_context)
        except Exception as e:
            res["answer"] = f"[Error generating answer: {e}]"
    else:
        res["answer"] = "No relevant context found."

    # Render
    st.subheader("Answer")
    st.write(res.get("answer", ""))

    with st.expander("Show retrieved chunks"):
        if not ctxs:
            st.write("No context retrieved.")
        else:
            for i, (doc, meta) in enumerate(ctxs, 1):
                src = (meta or {}).get("url") or (meta or {}).get("path", "")
                st.markdown(f"**Chunk {i}** — `{src}`")
                st.write(doc)
