#!/usr/bin/env python3
"""Voice CLI — test the voice cloning pipeline end-to-end.

Provides a unified command-line interface for the Zaya Audio voice
cloning pipeline.  All voice operations (record, train, speak, stream,
chat) are available from a single entry point.

Usage:
  # Record voice samples (300 seconds = 5 minutes)
  python tools/jarvis/voice_cli.py record --duration 300 --name my_voice

  # Train codec on recorded samples
  python tools/jarvis/voice_cli.py train --data voice_samples/my_voice/

  # Synthesize speech (via Python API)
  python tools/jarvis/voice_cli.py speak --text "Hello world" --voice my_voice

  # Stream via WebSocket to jarvis_server
  python tools/jarvis/voice_cli.py stream --text "Hello" --voice my_voice

  # Interactive chat with voice output
  python tools/jarvis/voice_cli.py chat --voice my_voice
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("voice_cli")

# ── Helpers ────────────────────────────────────────────────────────────


def find_python_module(module_name: str) -> bool:
    """Check if a Python module is available."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def require_module(module_name: str, pip_name: str | None = None) -> None:
    """Ensure a module is installed, offer install hint if missing."""
    if not find_python_module(module_name):
        pkg = pip_name or module_name
        print(f"Error: '{module_name}' not installed. Install with:")
        print(f"  pip install {pkg}")
        sys.exit(1)


def resolve_voice_pack_dir(name: str) -> Path:
    """Resolve a voice pack directory by name.

    Searches:
      1. ./voice_samples/<name>
      2. ~/voice-packs/<name>
      3. VOICE_PACKS_DIR/<name>
    """
    candidates = [
        Path.cwd() / "voice_samples" / name,
        Path.home() / "voice-packs" / name,
    ]
    voice_packs_dir = os.environ.get("VOICE_PACKS_DIR")
    if voice_packs_dir:
        candidates.append(Path(voice_packs_dir) / name)

    for c in candidates:
        if c.is_dir():
            return c

    return candidates[0]  # return first candidate even if it doesn't exist


def resolve_output_file(text: str, output_dir: Path, suffix: str = ".wav") -> Path:
    """Create an output filename from text content."""
    # Take first 40 chars of text as filename
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in text)[:40].strip()
    if not safe:
        safe = "speech"
    return output_dir / f"{safe}{suffix}"


# ── Subcommand: record ────────────────────────────────────────────────


def cmd_record(args: argparse.Namespace) -> None:
    """Record voice samples using zaya_audio.record."""
    log.info("Recording voice samples...")
    log.info("  duration:  %d s (%.1f min)", args.duration, args.duration / 60)
    log.info("  speaker:   %s", args.name)
    log.info("  output:    %s", args.output_dir)

    cmd = [
        sys.executable, "-m", "zaya_audio.record",
        "--duration", str(args.duration),
        "--sample_rate", "24000",
        "--output_dir", str(args.output_dir),
        "--speaker_name", args.name,
    ]
    if args.energy_threshold is not None:
        cmd.extend(["--energy_threshold", str(args.energy_threshold)])
    if args.no_normalize:
        cmd.append("--no_normalize")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log.error("Recording failed: %s", e)
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Error: zaya_audio.record module not available.\n"
            "Make sure zaya_audio is in PYTHONPATH or install it:\n"
            "  pip install -e ."
        )
        sys.exit(1)


# ── Subcommand: train ─────────────────────────────────────────────────


def cmd_train(args: argparse.Namespace) -> None:
    """Train a codec on recorded voice samples."""
    data_dir = Path(args.data).expanduser().resolve()
    if not data_dir.is_dir():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    output_dir = Path(args.output or f"./training/{data_dir.name}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Training codec...")
    log.info("  data:      %s", data_dir)
    log.info("  output:    %s", output_dir)
    log.info("  epochs:    %d", args.epochs)
    log.info("  batch:     %d", args.batch_size)
    log.info("  lr:        %s", args.lr)

    cmd = [
        sys.executable, "-m", "zaya_audio.train_codec",
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
    ]
    if args.resume:
        cmd.extend(["--resume", args.resume])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log.error("Training failed: %s", e)
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Error: zaya_audio.train_codec module not available.\n"
            "Make sure zaya_audio is in PYTHONPATH."
        )
        sys.exit(1)


# ── Subcommand: speak ─────────────────────────────────────────────────


def cmd_speak(args: argparse.Namespace) -> None:
    """Synthesize speech from text using VoiceSynthesizer."""
    require_module("zaya_audio.synthesize", "zaya_audio")

    from zaya_audio.synthesize import VoiceSynthesizer

    if args.voice_pack:
        voice_path = Path(args.voice_pack).expanduser().resolve()
    else:
        # Look for .voice pack file
        pack_dir = resolve_voice_pack_dir(args.voice)
        voice_file = pack_dir.with_suffix(".voice")
        if voice_file.is_file():
            voice_path = voice_file
        else:
            # Check for .voice in voice_samples/
            for ext in [".voice", ".voicepack"]:
                candidate = pack_dir.parent / f"{args.voice}{ext}"
                if candidate.is_file():
                    voice_path = candidate
                    break
            else:
                log.error(
                    "Voice pack not found for '%s'. Searched: %s{.voice,.voicepack}",
                    args.voice, pack_dir.parent / args.voice,
                )
                sys.exit(1)

    log.info("Loading voice pack: %s", voice_path)
    try:
        synth = VoiceSynthesizer(
            voice_path,
            use_onnx=args.onnx,
            device=args.device,
        )
    except Exception as e:
        log.error("Failed to load voice pack: %s", e)
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.text:
        # Single text
        texts = [args.text]
    else:
        # Interactive mode
        log.info("Interactive TTS mode (Ctrl+D to exit)")
        texts = []
        try:
            while True:
                line = input("Text > ").strip()
                if line:
                    texts.append(line)
        except EOFError:
            print()

    for text in texts:
        log.info("Synthesizing: %s", text[:60])
        start = time.time()
        audio = synth.synthesize(text)
        elapsed = time.time() - start

        output_path = resolve_output_file(text, output_dir)
        from zaya_audio.utils import save_audio
        save_audio(audio, str(output_path), sr=synth.config.sample_rate)

        audio_len = audio.shape[-1] / synth.config.sample_rate
        log.info(
            "  %.1fs audio in %.2fs (RTF=%.2f) → %s",
            audio_len, elapsed, elapsed / max(audio_len, 0.01),
            output_path,
        )


# ── Subcommand: stream ────────────────────────────────────────────────


def cmd_stream(args: argparse.Namespace) -> None:
    """Stream audio via WebSocket to jarvis_server.

    Connects to the jarvis_server WebSocket endpoint at /v1/audio/stream,
    sends a text prompt, receives streaming audio frames, and plays them
    back or saves to a file.
    """
    require_module("websockets")

    import asyncio
    import struct

    async def run_ws():
        uri = f"ws://{args.host}:{args.port}/v1/audio/stream?voice={args.voice}&text={args.text}"

        log.info("Connecting to %s", uri)
        log.info("  voice: %s", args.voice)
        log.info("  text:  %s", args.text)

        async with websockets.connect(uri) as websocket:
            # Receive metadata frame
            meta_raw = await websocket.recv()
            if isinstance(meta_raw, bytes):
                meta_raw = meta_raw.decode("utf-8")
            meta = json.loads(meta_raw)
            log.info("  metadata: %s", meta)

            sample_rate = meta.get("sample_rate", 24000)
            audio_chunks = []

            # Receive audio frames
            while True:
                msg = await websocket.recv()

                if isinstance(msg, str):
                    # Text frame (metadata or end)
                    try:
                        data = json.loads(msg)
                        msg_type = data.get("type", "")
                        if msg_type == "end":
                            reason = data.get("reason", "unknown")
                            log.info("Stream complete: %s", reason)
                            break
                    except json.JSONDecodeError:
                        log.warning("Unexpected text frame: %s", msg[:60])
                else:
                    # Binary frame (audio data)
                    # Raw float32 PCM samples
                    samples = struct.unpack(f"<{len(msg) // 4}f", msg)
                    audio_chunks.append(samples)

                    if args.progress:
                        total_sec = sum(
                            len(c) for c in audio_chunks
                        ) / sample_rate
                        print(f"\r  Received {total_sec:.1f}s audio", end="")
                        sys.stdout.flush()

            if args.progress:
                print()

            # Convert to WAV and save
            if audio_chunks and args.output:
                import numpy as np

                all_samples = np.concatenate(audio_chunks)
                # Normalize to int16
                audio_int16 = (np.clip(all_samples, -1.0, 1.0) * 32767).astype(np.int16)

                import scipy.io.wavfile as wavfile
                wavfile.write(str(args.output), sample_rate, audio_int16)
                log.info("Saved %d samples (% .1fs) → %s",
                         len(all_samples), len(all_samples) / sample_rate,
                         args.output)

            if args.play and audio_chunks:
                _play_audio(audio_chunks, sample_rate)

    import websockets  # noqa: F811 (already imported in require_module)
    asyncio.run(run_ws())


def _play_audio(audio_chunks: list, sample_rate: int) -> None:
    """Play audio chunks via aplay or ffplay."""
    import numpy as np
    import tempfile

    all_samples = np.concatenate(audio_chunks)
    audio_int16 = (np.clip(all_samples, -1.0, 1.0) * 32767).astype(np.int16)

    import scipy.io.wavfile as wavfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        wavfile.write(tmp_path, sample_rate, audio_int16)

    try:
        subprocess.run(["aplay", "-q", tmp_path], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(["ffplay", "-nodisp", "-autoexit", tmp_path], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            log.warning("Could not play audio (no aplay or ffplay)")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Subcommand: chat ──────────────────────────────────────────────────


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat with voice input/output via jarvis_server."""
    require_module("aiohttp")

    import asyncio
    import aiohttp

    # Find a usable audio playback method
    def play_wav(wav_bytes: bytes) -> None:
        """Play WAV bytes via aplay (non-blocking)."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            subprocess.Popen(
                ["aplay", "-q", tmp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            log.warning("aplay not found — cannot play audio")

        # Clean up after a short delay
        import threading

        def _clean():
            time.sleep(2)
            Path(tmp_path).unlink(missing_ok=True)

        threading.Thread(target=_clean, daemon=True).start()

    async def run_chat():
        api_base = f"http://{args.host}:{args.port}"
        log.info("Chat mode — connected to %s", api_base)
        log.info("Voice: %s (Ctrl+C to exit)", args.voice)
        print()
        print("Chat with JARVIS (voice synthesis enabled)")
        print("Type your message and press Enter. Ctrl+C to quit.")
        print()

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    text = input("You > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not text:
                    continue

                # Send chat message
                payload = {
                    "messages": [{"role": "user", "content": text}],
                    "stream": False,
                }
                try:
                    async with session.post(
                        f"{api_base}/v1/chat/completions",
                        json=payload,
                    ) as resp:
                        data = await resp.json()
                        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        print(f"JARVIS > {reply}")
                except Exception as e:
                    log.error("Chat request failed: %s", e)
                    continue

                # Synthesize the reply as speech
                tts_payload = {
                    "input": reply,
                    "voice": args.voice,
                    "play_local": False,
                }
                try:
                    async with session.post(
                        f"{api_base}/v1/audio/speech",
                        json=tts_payload,
                    ) as resp:
                        if resp.status == 200:
                            wav_bytes = await resp.read()
                            play_wav(wav_bytes)
                        else:
                            log.warning("TTS failed (status %d)", resp.status)
                except Exception as e:
                    log.debug("TTS request failed: %s", e)

    asyncio.run(run_chat())


# ── Parent parser ─────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Voice CLI — test the Zaya Audio voice cloning pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record 5 minutes of voice
  voice_cli.py record --duration 300 --name alice

  # Train on recorded samples
  voice_cli.py train --data voice_samples/alice/

  # Speak via Python API
  voice_cli.py speak --text "Hello world" --voice alice

  # Stream via WebSocket to jarvis_server
  voice_cli.py stream --text "Hello" --voice alice

  # Interactive chat
  voice_cli.py chat --voice alice
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # ── record ──
    p_record = subparsers.add_parser(
        "record", help="Record voice samples for codec training"
    )
    p_record.add_argument(
        "--duration", type=int, default=300,
        help="Recording duration in seconds (default 5 min)",
    )
    p_record.add_argument(
        "--name", type=str, required=True,
        help="Speaker name (used as directory name)",
    )
    p_record.add_argument(
        "--output_dir", type=str, default="./voice_samples",
        help="Output directory for voice samples",
    )
    p_record.add_argument(
        "--energy_threshold", type=float, default=None,
        help="VAD energy threshold in dB (default: auto)",
    )
    p_record.add_argument(
        "--no_normalize", action="store_true",
        help="Skip loudness normalisation",
    )

    # ── train ──
    p_train = subparsers.add_parser(
        "train", help="Train codec on recorded voice samples"
    )
    p_train.add_argument(
        "--data", type=str, required=True,
        help="Directory containing WAV files for training",
    )
    p_train.add_argument(
        "--output", type=str, default=None,
        help="Output directory for checkpoints (default: ./training/<name>)",
    )
    p_train.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs",
    )
    p_train.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size",
    )
    p_train.add_argument(
        "--lr", type=float, default=3e-4,
        help="Peak learning rate",
    )
    p_train.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint path",
    )

    # ── speak ──
    p_speak = subparsers.add_parser(
        "speak", help="Synthesize speech from text using VoiceSynthesizer"
    )
    p_speak.add_argument(
        "--voice", type=str, required=True,
        help="Voice name or voice pack path",
    )
    p_speak.add_argument(
        "--voice_pack", type=str, default=None,
        help="Explicit path to .voice pack file",
    )
    p_speak.add_argument(
        "--text", type=str, default=None,
        help="Text to synthesize (omit for interactive mode)",
    )
    p_speak.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory for audio files",
    )
    p_speak.add_argument(
        "--onnx", action="store_true",
        help="Use ONNX Runtime for inference",
    )
    p_speak.add_argument(
        "--device", type=str, default=None,
        help="Device for PyTorch inference (e.g. 'cpu', 'cuda')",
    )

    # ── stream ──
    p_stream = subparsers.add_parser(
        "stream", help="Stream audio via WebSocket to jarvis_server"
    )
    p_stream.add_argument(
        "--voice", type=str, required=True,
        help="Voice pack name",
    )
    p_stream.add_argument(
        "--text", type=str, required=True,
        help="Text to synthesize",
    )
    p_stream.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="jarvis_server host",
    )
    p_stream.add_argument(
        "--port", type=int, default=8080,
        help="jarvis_server port",
    )
    p_stream.add_argument(
        "--output", type=str, default="./output_stream.wav",
        help="Output WAV file path",
    )
    p_stream.add_argument(
        "--play", action="store_true",
        help="Play audio after receiving",
    )
    p_stream.add_argument(
        "--progress", action="store_true",
        help="Show progress during streaming",
    )
    p_stream.add_argument(
        "--no-save", action="store_true",
        help="Don't save audio to file",
    )

    # ── chat ──
    p_chat = subparsers.add_parser(
        "chat", help="Interactive chat with voice via jarvis_server"
    )
    p_chat.add_argument(
        "--voice", type=str, required=True,
        help="Voice pack name",
    )
    p_chat.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="jarvis_server host",
    )
    p_chat.add_argument(
        "--port", type=int, default=8080,
        help="jarvis_server port",
    )

    return parser


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "record":
        cmd_record(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "speak":
        cmd_speak(args)
    elif args.command == "stream":
        cmd_stream(args)
    elif args.command == "chat":
        cmd_chat(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
