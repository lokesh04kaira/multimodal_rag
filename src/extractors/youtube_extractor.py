import os
import tempfile
from pathlib import Path

from yt_dlp import YoutubeDL
from src.extractors.av_extractor import _to_wav, _transcribe_wav
from src.utils import clean_text


def _video_id(url: str) -> str:
    """Resolve the video ID via yt-dlp metadata without downloading."""
    opts = {"quiet": True, "skip_download": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        vid = info.get("id")
        if not vid:
            raise ValueError("Could not resolve YouTube video ID")
        return vid


def _captions_text(video_id: str) -> str | None:
    """
    Try to fetch captions via youtube-transcript-api.
    Preference order:
      1) Manually-created captions in preferred languages
      2) Auto-generated captions in preferred languages
      3) Any available captions (generated or manual)
    """
    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            TranscriptsDisabled,
            NoTranscriptFound,
            NoTranscriptAvailable,
        )

        langs = [x.strip() for x in os.getenv("YT_LANGS", "en,en-US,en-GB").split(",") if x.strip()]
        tlist = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            t = tlist.find_manually_created_transcript(langs)
            items = t.fetch()
            text = " ".join(ch.get("text", "").strip() for ch in items if ch.get("text"))
            return clean_text(text) if text.strip() else None
        except Exception:
            pass

        try:
            t = tlist.find_generated_transcript(langs)
            items = t.fetch()
            text = " ".join(ch.get("text", "").strip() for ch in items if ch.get("text"))
            return clean_text(text) if text.strip() else None
        except Exception:
            pass

        for t in tlist:
            try:
                items = t.fetch()
                text = " ".join(ch.get("text", "").strip() for ch in items if ch.get("text"))
                if text.strip():
                    return clean_text(text)
            except Exception:
                continue

        return None
    except (TranscriptsDisabled, NoTranscriptFound, NoTranscriptAvailable):
        return None
    except Exception:
        return None


def extract_youtube(url: str) -> str:
    """
    Extract transcript for a YouTube URL.
    - Prefer official captions if available.
    - Otherwise download bestaudio and run ASR.
    - Returns '' on failure.
    """
    try:
        vid = _video_id(url)
        cap = _captions_text(vid)
        if cap and len(cap) > 30:
            return cap
    except Exception:
        pass

    cookies = os.getenv("YT_COOKIES") or os.getenv("YT_COOKIES_PATH") 
    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "outtmpl": None,  
    }
    if cookies and Path(cookies).exists():
        ydl_opts["cookiefile"] = cookies

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts["outtmpl"] = str(Path(tmpdir) / "%(id)s.%(ext)s")
        downloaded_path = None
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_path = Path(ydl.prepare_filename(info))
        except Exception:
            return ""

        if not downloaded_path or not downloaded_path.exists():
            return ""

        try:
            wav = _to_wav(str(downloaded_path))
            try:
                return _transcribe_wav(wav)
            finally:
                try:
                    os.remove(wav)
                except Exception:
                    pass
        except Exception:
            return ""
