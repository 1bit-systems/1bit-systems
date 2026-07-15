import io
import os
import subprocess
import wave

PIPER_VOICES_DIR = os.environ.get("PIPER_VOICES_DIR", os.path.expanduser("~/piper-voices"))
JARVIS_VENV = os.environ.get("JARVIS_VENV", os.path.expanduser("~/jarvis-env"))


def synthesize_speech(text, voice="en_US-lessac-medium"):
    model_path = os.path.join(PIPER_VOICES_DIR, f"{voice}.onnx")
    if not os.path.exists(model_path):
        return None
    piper_bin = os.path.join(JARVIS_VENV, "bin/piper")
    if not os.path.exists(piper_bin):
        piper_bin = "piper"
    try:
        proc = subprocess.run(
            [piper_bin, "--model", model_path, "--output-raw"],
            input=text.encode("utf-8"), capture_output=True, timeout=30,
        )
        if proc.returncode != 0:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(proc.stdout)
        return buf.getvalue()
    except:
        return None
