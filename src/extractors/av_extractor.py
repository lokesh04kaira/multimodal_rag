import os, tempfile, subprocess
from pathlib import Path
from typing import Optional
from src.utils import clean_text
import re

def _ffmpeg_bin() -> str:
    return os.getenv("FFMPEG_BIN", "ffmpeg")

def _to_wav(in_path: str) -> str:
    if not Path(in_path).exists():
        raise FileNotFoundError(in_path)
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    cmd = [
        _ffmpeg_bin(), "-y", "-i", in_path,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out_path
    except Exception as e:
        try: os.remove(out_path)
        except Exception: pass
        raise RuntimeError(f"ffmpeg failed: {e}") from e

def _dedupe_sentences(text: str) -> str:
    sents = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    out, seen = [], set()
    for s in sents:
        norm = " ".join(s.lower().split())
        if norm and norm not in seen:
            out.append(s.strip())
            seen.add(norm)
    return " ".join(out)

def _postproc(text: str) -> str:
    t = clean_text(text or "")
    t = _dedupe_sentences(t)
    return t.strip()

def _transcribe_wav(wav_path: str) -> str:
    model_size = os.getenv("WHISPER_MODEL", "base").strip()

    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device=os.getenv("WHISPER_DEVICE", "cpu"), compute_type=os.getenv("WHISPER_COMPUTE", "int8"))
        segments, _info = model.transcribe(
            wav_path,
            vad_filter=True,
            beam_size=5,
            temperature=0.0,
        )
        text = " ".join((seg.text or "").strip() for seg in segments)
        return _postproc(text)
    except Exception:
        pass

    try:
        import whisper
        model = whisper.load_model(model_size)
        res = model.transcribe(
            wav_path,
            temperature=0.0,
            beam_size=5,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        return _postproc(res.get("text", ""))
    except Exception:
        return ""  

def extract_audio(path: str) -> str:
    wav = _to_wav(path)
    try:
        return _transcribe_wav(wav)
    finally:
        try: os.remove(wav)
        except Exception: pass

def extract_video(path: str) -> str:
    wav = _to_wav(path)  
    try:
        return _transcribe_wav(wav)
    finally:
        try: os.remove(wav)
        except Exception: pass
