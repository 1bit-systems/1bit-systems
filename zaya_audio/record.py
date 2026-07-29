#! /usr/bin/env python3
"""Voice recording CLI tool for zaya_audio.

Records audio from a microphone, splits into speech segments using
energy-based VAD, normalises loudness, and prepares the data for
codec training or speaker embedding extraction.

Two recording backends:
1. **pyaudio** (preferred) — cross-platform PortAudio binding.
2. **arecord** (fallback) — ALSA ``arecord`` subprocess on Linux.

Usage
-----
.. code-block:: bash

    # Record for 30 minutes with default settings
    python -m zaya_audio.record --duration 1800 --speaker-name alice

    # Record for 5 minutes with custom output directory
    python -m zaya_audio.record --duration 300 --speaker-name bob \\
        --output_dir ./my_voice_samples
"""

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("record")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# VAD defaults
VAD_ENERGY_THRESHOLD_DB = -30.0    # dB below which is silence
VAD_MIN_SILENCE_MS = 300           # ms of silence to end a segment
VAD_MIN_SEGMENT_MS = 3000          # minimum segment duration (3 s)
VAD_MAX_SEGMENT_MS = 10000         # maximum segment duration (10 s)
SILENCE_PADDING_MS = 200           # padding (ms) around detected speech

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SegmentInfo:
    """Information about a single recorded and processed segment."""
    filename: str              # relative path within segments/
    duration_sec: float        # duration in seconds
    sample_rate: int           # sample rate
    rms_db: float              # RMS level in dB
    snr_db: float              # estimated SNR in dB
    clipping: bool             # whether clipping was detected


# ---------------------------------------------------------------------------
# Recording backends
# ---------------------------------------------------------------------------

def _backend_available(name: str) -> bool:
    """Check if a recording backend is available."""
    if name == "pyaudio":
        try:
            import pyaudio
            pyaudio.PyAudio().terminate()
            return True
        except Exception:
            return False
    elif name == "arecord":
        return shutil.which("arecord") is not None
    return False


def select_backend() -> str:
    """Auto-select the best available recording backend."""
    if _backend_available("pyaudio"):
        return "pyaudio"
    if _backend_available("arecord"):
        return "arecord"
    raise RuntimeError(
        "No recording backend available. Install pyaudio (pip install pyaudio) "
        "or ensure arecord (alsa-utils) is on PATH."
    )


def record_pyaudio(duration: int, sample_rate: int, output_path: Path) -> None:
    """Record audio using pyaudio (PortAudio)."""
    import pyaudio as pa_lib

    chunk = 1024
    format = pa_lib.paInt16
    channels = 1

    p = pa_lib.PyAudio()
    try:
        stream = p.open(
            format=format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk,
        )
    except OSError as e:
        p.terminate()
        raise RuntimeError(
            f"Could not open audio input device: {e}\n"
            "Try specifying a different device or use arecord backend."
        ) from e

    log.info("Recording: %d s at %d Hz via pyaudio ...", duration, sample_rate)
    frames: List[bytes] = []
    total_chunks = int(sample_rate / chunk * duration)

    try:
        for i in range(total_chunks):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            # Progress indicator
            if i % max(1, total_chunks // 20) == 0:
                pct = int(100 * i / total_chunks)
                bar = "#" * (pct // 5) + "." * (20 - pct // 5)
                sys.stdout.write(f"\r  [{bar}] {pct}%")
                sys.stdout.flush()
    except KeyboardInterrupt:
        log.info("Recording interrupted by user.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        sys.stdout.write("\n")

    # Write WAV
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    log.info("Raw recording saved: %s (%.1f s)", output_path,
             len(b"".join(frames)) / sample_rate / 2)


def record_arecord(duration: int, sample_rate: int, output_path: Path) -> None:
    """Record audio using arecord (ALSA subprocess)."""
    cmd = [
        "arecord",
        "-d", str(duration),
        "-r", str(sample_rate),
        "-c", "1",
        "-f", "S16_LE",
        "-t", "wav",
        str(output_path),
    ]
    log.info("Recording: %d s at %d Hz via arecord ...", duration, sample_rate)
    log.debug("Running: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"arecord failed: {e}") from e
    except FileNotFoundError as e:
        raise RuntimeError(
            "arecord not found. Install alsa-utils: "
            "apt-get install alsa-utils (Linux)"
        ) from e

    log.info("Raw recording saved: %s", output_path)


# ---------------------------------------------------------------------------
# VAD splitting
# ---------------------------------------------------------------------------

def split_into_segments(
    wav_path: Path,
    output_dir: Path,
    sample_rate: int,
    energy_threshold_db: float = VAD_ENERGY_THRESHOLD_DB,
    min_silence_ms: int = VAD_MIN_SILENCE_MS,
    min_segment_ms: int = VAD_MIN_SEGMENT_MS,
    max_segment_ms: int = VAD_MAX_SEGMENT_MS,
    silence_padding_ms: int = SILENCE_PADDING_MS,
) -> List[SegmentInfo]:
    """Split a WAV file into speech segments using energy-based VAD.

    The algorithm:
    1. Compute short-time RMS energy (20 ms windows).
    2. Convert to dB and classify frames as speech / silence.
    3. Merge consecutive speech frames into segments.
    4. Enforce min/max segment duration constraints.
    5. Save each segment as a separate WAV file.

    Parameters
    ----------
    wav_path : Path
        Path to the input WAV file.
    output_dir : Path
        Directory where segments will be saved.
    sample_rate : int
        Sample rate of the audio.
    energy_threshold_db : float
        Threshold in dB below which a frame is considered silence.
    min_silence_ms : int
        Minimum silence duration (ms) to end a segment.
    min_segment_ms : int
        Minimum segment duration (ms). Shorter segments are discarded.
    max_segment_ms : int
        Maximum segment duration (ms). Longer segments are split.
    silence_padding_ms : int
        Padding (ms) at segment boundaries to avoid hard cuts.

    Returns
    -------
    segments : list of SegmentInfo
        Information about each saved segment.
    """
    import scipy.io.wavfile as wavfile

    sr, audio = wavfile.read(str(wav_path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # ── Frame-level energy ─────────────────────────────────────────────
    frame_ms = 20  # 20 ms windows
    frame_len = int(sr * frame_ms / 1000)
    hop_len = frame_len // 2  # 50 % overlap

    n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)
    rms_db = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        frame = audio[start:end]
        rms = max(1e-10, np.sqrt(np.mean(frame ** 2)))
        rms_db[i] = 20.0 * math.log10(rms)

    # ── Speech / silence classification ─────────────────────────────────
    is_speech = rms_db > energy_threshold_db

    # ── Merge into segments ─────────────────────────────────────────────
    min_silence_frames = min_silence_ms // frame_ms
    padding_frames = silence_padding_ms // frame_ms

    segments: List[Tuple[int, int]] = []  # (start_sample, end_sample) pairs
    in_speech = False
    seg_start = 0
    silence_count = 0

    for i in range(len(is_speech)):
        if is_speech[i] and not in_speech:
            # Start of speech
            seg_start = max(0, i * hop_len - padding_frames * hop_len)
            in_speech = True
            silence_count = 0
        elif not is_speech[i] and in_speech:
            silence_count += 1
            if silence_count >= min_silence_frames:
                # End of speech segment
                seg_end = min(
                    len(audio),
                    i * hop_len + padding_frames * hop_len
                )
                if seg_end - seg_start >= int(sr * min_segment_ms / 1000):
                    segments.append((seg_start, seg_end))
                in_speech = False
        # Continue speech

    # Handle trailing speech
    if in_speech:
        seg_end = min(len(audio),
                      len(audio) - 1)
        if seg_end - seg_start >= int(sr * min_segment_ms / 1000):
            segments.append((seg_start, seg_end))

    # ── Enforce max segment duration (split long segments) ──────────────
    max_samples = int(sr * max_segment_ms / 1000)
    split_segments: List[Tuple[int, int]] = []
    for start, end in segments:
        if end - start <= max_samples:
            split_segments.append((start, end))
        else:
            for t in range(start, end, max_samples):
                split_segments.append((t, min(t + max_samples, end)))

    segments = split_segments

    # ── Save segments ───────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_infos: List[SegmentInfo] = []
    for idx, (start, end) in enumerate(segments):
        seg_audio = audio[start:end]
        duration_sec = len(seg_audio) / sr

        # Skip very short segments (noise bursts)
        if duration_sec < (min_segment_ms / 1000):
            continue

        # RMS
        rms = max(1e-10, np.sqrt(np.mean(seg_audio ** 2)))
        rms_db_seg = 20.0 * math.log10(rms)

        # Clipping detection
        clipping = bool(np.any(np.abs(seg_audio) >= 0.99))

        # SNR estimate (simple: speech vs silence energy ratio)
        speech_mask = np.abs(seg_audio) > 0.02
        if speech_mask.sum() > 0:
            speech_power = np.mean(seg_audio[speech_mask] ** 2)
            noise_power = np.mean(seg_audio[~speech_mask] ** 2) if (~speech_mask).sum() > 0 else 1e-10
            snr_db = 10.0 * math.log10(max(speech_power, 1e-10) / max(noise_power, 1e-10))
        else:
            snr_db = 0.0

        # Write WAV
        seg_path = output_dir / f"segment_{idx:05d}.wav"
        _save_wav_int16(str(seg_path), seg_audio, sr)

        segment_infos.append(SegmentInfo(
            filename=str(seg_path.relative_to(output_dir.parent)),
            duration_sec=round(duration_sec, 3),
            sample_rate=sr,
            rms_db=round(rms_db_seg, 1),
            snr_db=round(snr_db, 1),
            clipping=clipping,
        ))

    return segment_infos


# ---------------------------------------------------------------------------
# Loudness normalisation
# ---------------------------------------------------------------------------

def normalize_loudness(segments_dir: Path, target_lufs: float = -23.0,
                       sample_rate: int = 24000) -> None:
    """Normalise loudness of all WAV files in a directory.

    Tries ``ffmpeg-normalize`` first (EBU R128 / LUFS).  Falls back to
    simple RMS normalisation to the target level.

    Parameters
    ----------
    segments_dir : Path
        Directory containing WAV segment files.
    target_lufs : float
        Target integrated loudness in LUFS (default -23 LUFS).
    sample_rate : int
        Target sample rate for output files.
    """
    wav_files = sorted(segments_dir.glob("*.wav"))
    if not wav_files:
        log.warning("No WAV files found in %s", segments_dir)
        return

    log.info("Normalising %d segments to %.1f LUFS ...", len(wav_files), target_lufs)

    # Try ffmpeg-normalize
    if shutil.which("ffmpeg-normalize") is not None:
        _normalize_ffmpeg(wav_files, segments_dir, target_lufs, sample_rate)
    elif shutil.which("ffmpeg") is not None:
        _normalize_ffmpeg_direct(wav_files, segments_dir, target_lufs, sample_rate)
    else:
        _normalize_rms(wav_files, segments_dir, target_lufs)
        log.warning(
            "ffmpeg-normalize not found — used simple RMS normalisation. "
            "Install with: pip install ffmpeg-normalize"
        )


def _normalize_ffmpeg(wav_files: List[Path], output_dir: Path,
                      target_lufs: float, sample_rate: int) -> None:
    """Normalise with ffmpeg-normalize."""
    tmp_dir = output_dir / ".normalize_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for wav in wav_files:
        rel = wav.relative_to(output_dir.parent)
        tmp_path = tmp_dir / rel.name
        cmd = [
            "ffmpeg-normalize", str(wav),
            "-o", str(tmp_path),
            "-nt", "ebu",
            "-t", str(target_lufs),
            "-f", "wav",
            "-ar", str(sample_rate),
            "-c", "1",
            "-l", "0",   # don't limit true peak
            "--dual-mono",
            "-of", "same",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.replace(tmp_path, wav)
        except subprocess.CalledProcessError as e:
            log.warning("ffmpeg-normalize failed for %s: %s", wav.name,
                        e.stderr.decode()[:200] if e.stderr else "unknown")
            if tmp_path.exists():
                tmp_path.unlink()

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _normalize_ffmpeg_direct(wav_files: List[Path], output_dir: Path,
                             target_lufs: float, sample_rate: int) -> None:
    """Normalise with ffmpeg loudnorm filter."""
    for wav in wav_files:
        tmp_path = wav.with_suffix(".tmp.wav")
        cmd = [
            "ffmpeg", "-y", "-i", str(wav),
            "-af", f"loudnorm=I={target_lufs}:LRA=7:TP=-2.0",
            "-ar", str(sample_rate),
            "-ac", "1",
            str(tmp_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.replace(tmp_path, wav)
        except subprocess.CalledProcessError as e:
            log.warning("ffmpeg loudnorm failed for %s: %s", wav.name,
                        e.stderr.decode()[:200] if e.stderr else "unknown")
            if tmp_path.exists():
                tmp_path.unlink()


def _normalize_rms(wav_files: List[Path], output_dir: Path,
                   target_lufs: float) -> None:
    """Simple RMS-based normalisation as fallback.

    LUFS is roughly correlated with RMS, so we target RMS level that
    approximates the desired LUFS.  For speech, -23 LUFS ≈ -23 dB RMS.
    """
    import scipy.io.wavfile as wavfile

    target_rms = 10 ** (target_lufs / 20.0)  # linear RMS

    for wav in wav_files:
        sr, audio = wavfile.read(str(wav))
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

        current_rms = max(1e-10, np.sqrt(np.mean(audio ** 2)))
        gain = target_rms / current_rms
        audio = audio * gain

        # Prevent clipping
        audio = np.clip(audio, -1.0, 1.0)

        _save_wav_int16(str(wav), audio, sr)


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------

def quality_check(segments: List[SegmentInfo]) -> None:
    """Print a quality report for a set of audio segments.

    Parameters
    ----------
    segments : list of SegmentInfo
        Segment metadata from ``split_into_segments``.
    """
    if not segments:
        print("[quality] No segments to check.")
        return

    durations = [s.duration_sec for s in segments]
    rms_values = [s.rms_db for s in segments]
    snr_values = [s.snr_db for s in segments]
    clipping_count = sum(1 for s in segments if s.clipping)

    print(f"[quality] {'─' * 42}")
    print(f"[quality] Segments:         {len(segments)}")
    print(f"[quality] Total duration:   {sum(durations):.1f} s "
          f"({sum(durations) / 60:.1f} min)")
    print(f"[quality] Duration range:   {min(durations):.2f} – "
          f"{max(durations):.2f} s")
    print(f"[quality] Avg RMS:          {np.mean(rms_values):.1f} dB")
    print(f"[quality] Avg SNR (est.):   {np.mean(snr_values):.1f} dB")
    print(f"[quality] Clipping:         {'YES ⚠' if clipping_count > 0 else 'None ✓'} "
          f"({clipping_count} segments)")
    print(f"[quality] {'─' * 42}")

    if clipping_count > 0:
        log.warning("Clipping detected in %d segment(s). Reduce input gain.", clipping_count)


# ---------------------------------------------------------------------------
# WAV I/O helpers
# ---------------------------------------------------------------------------

def _save_wav_int16(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Save a float array as a 16-bit PCM WAV file."""
    import scipy.io.wavfile as wavfile

    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, audio_int16)


def _load_wav_int16(path: str) -> Tuple[int, np.ndarray]:
    """Load a WAV file as a float32 array in [-1, 1]."""
    import scipy.io.wavfile as wavfile

    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    return sr, audio


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record voice audio and prepare for codec training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--duration", type=int, default=1800,
                        help="Recording duration in seconds (default 30 min)")
    parser.add_argument("--sample_rate", type=int, default=24000,
                        help="Audio sample rate in Hz")
    parser.add_argument("--output_dir", type=str, default="./voice_samples",
                        help="Output directory for voice samples")
    parser.add_argument("--speaker_name", type=str, default="unknown",
                        help="Name of the speaker being recorded")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["pyaudio", "arecord"],
                        help="Recording backend (auto-detected if omitted)")
    parser.add_argument("--energy_threshold", type=float, default=VAD_ENERGY_THRESHOLD_DB,
                        help="VAD energy threshold in dB")
    parser.add_argument("--target_lufs", type=float, default=-23.0,
                        help="Target loudness in LUFS for normalisation")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Skip loudness normalisation")
    parser.add_argument("--list_devices", action="store_true",
                        help="List available audio input devices and exit")

    return parser


def list_devices() -> None:
    """Print available audio input devices."""
    try:
        import pyaudio as pa_lib
    except ImportError:
        print("pyaudio not installed. Cannot list devices.")
        return

    p = pa_lib.PyAudio()
    try:
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get("deviceCount", 0)
        print(f"Audio devices ({num_devices} found):")
        print(f"{'Index':>6}  {'Name':<40}  {'Channels':>8}  {'SR':>8}")
        print("-" * 68)
        for i in range(num_devices):
            dev = p.get_device_info_by_index(i)
            if dev.get("maxInputChannels", 0) > 0:
                print(
                    f"{i:>6}  {dev['name']:<40}  "
                    f"{dev['maxInputChannels']:>8}  "
                    f"{dev['defaultSampleRate']:>8.0f}"
                )
    finally:
        p.terminate()


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── List devices mode ──────────────────────────────────────
    if args.list_devices:
        list_devices()
        return

    # ── Output structure ────────────────────────────────────────
    output_dir = Path(args.output_dir).expanduser().resolve()
    speaker_dir = output_dir / args.speaker_name
    segments_dir = speaker_dir / "segments"
    raw_path = speaker_dir / "raw_recording.wav"

    speaker_dir.mkdir(parents=True, exist_ok=True)

    # ── Select backend ──────────────────────────────────────────
    backend = args.backend or select_backend()
    log.info("Recording backend: %s", backend)

    # ── Record ──────────────────────────────────────────────────
    log.info("Starting recording: speaker=%s, duration=%ds, sr=%d Hz",
             args.speaker_name, args.duration, args.sample_rate)

    if backend == "pyaudio":
        record_pyaudio(args.duration, args.sample_rate, raw_path)
    elif backend == "arecord":
        record_arecord(args.duration, args.sample_rate, raw_path)

    raw_size_mb = raw_path.stat().st_size / (1024 * 1024)
    log.info("Raw recording: %.1f MB", raw_size_mb)

    # ── VAD split ───────────────────────────────────────────────
    log.info("Splitting into speech segments (VAD threshold=%.1f dB) ...",
             args.energy_threshold)
    segments = split_into_segments(
        raw_path,
        segments_dir,
        args.sample_rate,
        energy_threshold_db=args.energy_threshold,
    )
    log.info("Found %d speech segments", len(segments))

    # ── Loudness normalisation ──────────────────────────────────
    if not args.no_normalize:
        normalize_loudness(segments_dir, target_lufs=args.target_lufs,
                           sample_rate=args.sample_rate)
    else:
        log.info("Skipped loudness normalisation (--no_normalize)")

    # ── Quality check ───────────────────────────────────────────
    quality_check(segments)

    # ── Write manifest ──────────────────────────────────────────
    manifest_path = speaker_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for seg in segments:
            f.write(json.dumps(asdict(seg)) + "\n")
    log.info("Manifest written: %s (%d entries)", manifest_path, len(segments))

    # ── Summary ─────────────────────────────────────────────────
    total_duration = sum(s.duration_sec for s in segments)
    print()
    print(f"[record] {'═' * 48}")
    print(f"[record] Speaker:        {args.speaker_name}")
    print(f"[record] Segments:       {len(segments)}")
    print(f"[record] Speech total:   {total_duration:.1f} s "
          f"({total_duration / 60:.1f} min)")
    print(f"[record] Output dir:     {speaker_dir}")
    print(f"[record] {'═' * 48}")
    log.info("Done! Ready for training: python -m zaya_audio.train_codec "
             "--data_dir %s --output_dir ./training/my_codec", segments_dir)


if __name__ == "__main__":
    main()
