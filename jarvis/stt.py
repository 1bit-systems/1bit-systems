import os
import sys
import tempfile
import threading

_whisper_model = None
_whisper_lock = threading.Lock()
JARVIS_VENV = os.environ.get("JARVIS_VENV", os.path.expanduser("~/jarvis-env"))


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
                sys.path.insert(0, os.path.join(JARVIS_VENV, f"lib/{py_ver}/site-packages"))
                from faster_whisper import WhisperModel
                _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_audio(wav_bytes):
    try:
        model = _get_whisper()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name
        segments, info = model.transcribe(tmp, beam_size=1)
        text = " ".join(seg.text for seg in segments)
        os.unlink(tmp)
        return text.strip() or "[silence]"
    except Exception as e:
        return f"[transcription error: {e}]"
