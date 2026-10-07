"""Audio analysis. Heavy dependencies (librosa) are imported lazily so the rest of the tool works
without the ``audio`` extra installed."""
from __future__ import annotations

import wave
from pathlib import Path


def require_librosa():
    try:
        import librosa
    except ImportError as e:
        raise RuntimeError("audio analysis needs the audio extra: pip install -e .[audio]") from e
    return librosa


def duration(path: str | Path) -> float | None:
    """Length in seconds, or None if it can't be read without librosa."""
    try:
        import soundfile
        return float(soundfile.info(str(path)).duration)
    except Exception:
        pass
    if str(path).lower().endswith(".wav"):
        try:
            with wave.open(str(path)) as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            pass
    try:
        return float(require_librosa().get_duration(path=str(path)))
    except Exception:
        return None
