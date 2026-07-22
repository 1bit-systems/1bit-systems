"""Local speaker playback for jarvis's TTS output.

Lets the server mirror synthesized speech out through a physical speaker
attached to the box jarvis runs on (the "voice demo centerpiece" use case),
independent of whatever the calling client does with the returned audio
bytes. Entirely dormant (no-op, no error) until a non-onboard playback
device is actually present -- ships now, starts working the moment a USB
speaker is plugged in, no further code changes needed.
"""
import re
import subprocess
import threading

_ONBOARD_HINTS = ("hd-audio generic", "hdmi")


def list_playback_devices():
    """Parse `aplay -l` into a list of {card, device, name, is_onboard}."""
    try:
        out = subprocess.run(["aplay", "-l"], capture_output=True, timeout=5, text=True).stdout
    except Exception:
        return []
    devices = []
    for line in out.splitlines():
        m = re.match(r"card (\d+): \S+ \[(.*?)\], device (\d+): (.*?) \[", line)
        if not m:
            continue
        card, name, dev, dev_name = m.groups()
        onboard = any(h in name.lower() for h in _ONBOARD_HINTS)
        devices.append({
            "card": int(card), "device": int(dev),
            "name": name, "device_name": dev_name,
            "is_onboard": onboard,
            "alsa_id": f"plughw:CARD={card},DEV={dev}",
        })
    return devices


def find_external_speaker():
    """First non-onboard playback device (i.e. a plugged-in USB speaker), or None."""
    for d in list_playback_devices():
        if not d["is_onboard"]:
            return d
    return None


def play_wav_local(wav_bytes, device=None, blocking=False):
    """Play WAV bytes on a local ALSA device. No-op if none is available.

    Fire-and-forget by default (runs in a background thread) so it never
    adds latency to the HTTP response that carries the same audio back to
    the calling client.
    """
    dev = device or find_external_speaker()
    if not dev:
        return False

    def _run():
        try:
            subprocess.run(
                ["aplay", "-q", "-D", dev["alsa_id"], "-"],
                input=wav_bytes, capture_output=True, timeout=60,
            )
        except Exception:
            pass

    if blocking:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()
    return True
