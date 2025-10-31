# 📚 Multimodal Data Processing System (Multimodal RAG)

> An intelligent Retrieval-Augmented Generation (RAG) system that processes multimodal input files — text, documents, images, audio, video, and YouTube — and answers user queries in natural language.

---

## 🧩 Overview

This project builds a **Multimodal RAG (Retrieval-Augmented Generation)** system capable of:
- Extracting and processing data from diverse file formats
- Converting it into embeddings stored in a local vector database
- Allowing users to query the knowledge base in **natural language**
- Generating **context-aware answers** using Gemini or a local LLM

---

## 🚀 Features

✅ Supports multiple data types:
- **Text:** `.pdf`, `.docx`, `.pptx`, `.md`, `.txt`
- **Image:** `.png`, `.jpg`, `.jpeg`
- **Audio/Video:** `.mp3`, `.wav`, `.mp4`, `.mkv`
- **YouTube:** Direct URL ingestion and transcription

✅ Capabilities:
- Automatic text extraction (OCR, Speech-to-text, Captions)
- Chunking and vector embedding
- Semantic search using **ChromaDB**
- Question-answering using **Gemini API** or **Local Model**
- Web UI via **Streamlit**
- Command-line interface for automation

---

## ⚙️ Tech Stack

| Component | Technology Used |
|------------|------------------|
| Frontend | Streamlit |
| Backend | Python |
| Vector DB | ChromaDB |
| LLM | Google Gemini (Free version) |
| Local Embedding Fallback | Sentence Transformers |
| OCR | Tesseract + EasyOCR |
| Audio/Video | FFmpeg + Whisper / Faster-Whisper |
| YouTube | yt-dlp + YouTube Transcript API |
| Config | dotenv (.env) |

---

## 🏗️ Project Structure

multimodal-rag/
│
├── app.py # Streamlit app for UI
├── requirements.txt
├── .env.example # Environment variable template
├── src/
│ ├── cli.py # CLI interface
│ ├── ingest.py # Ingestion logic
│ ├── retriever.py # Query + retrieval
│ ├── indexer.py # Vector database indexing
│ ├── llm.py # LLM + Embedding integration
│ ├── utils.py # Utility helpers
│ ├── extractors/ # File-specific data extractors
│ │ ├── pdf_extractor.py
│ │ ├── docx_extractor.py
│ │ ├── pptx_extractor.py
│ │ ├── md_txt_extractor.py
│ │ ├── image_extractor.py
│ │ ├── av_extractor.py
│ │ ├── youtube_extractor.py
│ │ └── init.py
│ └── init.py
└── data/
├── uploads/ # Uploaded files
└── chroma/ # Vector embeddings


---

## 🧠 Working Principle

1. **Ingestion**  
   User uploads files or a YouTube link → content extracted → cleaned → embedded.

2. **Indexing**  
   Text chunks stored in ChromaDB with metadata (type, path, etc.).

3. **Query**  
   User enters a natural-language question → system embeds it → retrieves similar chunks.

4. **Answer Generation**  
   Gemini (or local model) processes context → generates concise, accurate response.

---

## 🧰 Installation


### 1️⃣ Clone Repository
```bash
git clone https://github.com/lokesh04kaira/multimodal_rag.git
cd multimodal_rag

### 2️⃣ Create Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate     # on Windows
# or
source .venv/bin/activate    # on macOS/Linux

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Configure Environment Variables
GOOGLE_API_KEY=your_api_key_here
CHAT_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
UPLOAD_DIR=./data/uploads
CHROMA_DIR=./data/chroma
WHISPER_MODEL=base
YT_LANGS=en,en-US,en-GB
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe


👨‍💻 Author
Developer: Lokesh Kaira
Tech Stack: Python · Streamlit · ChromaDB · Gemini API · Whisper · OCR

