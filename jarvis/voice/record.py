#!/usr/bin/env python3
"""
Voice recording utility — Zaya Co-Host Phase 1.1

Records high-quality voice samples for cloning.
Aim for 30 minutes total across multiple recording sessions.

Usage:
  python3 jarvis/voice/record.py --name bcloud --session 1
  python3 jarvis/voice/record.py --name bcloud --session 2 --duration 600

Output:
  ./voice/samples/{name}/session-{n}_*.wav
"""

import argparse, os, subprocess, sys, time
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent.parent.parent / "voice" / "samples"
DEFAULT_DURATION = 300   # 5 minutes per session
SAMPLE_RATE = 24000      # Codec native sample rate (no downsampling needed)


def list_audio_devices():
    """List available ALSA capture devices."""
    try:
        result = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("Available audio devices:")
            for line in result.stdout.split("\n"):
                if "card" in line.lower():
                    print(f"  {line.strip()}")
        else:
            print("No ALSA devices found via arecord -l")
    except:
        print("Could not list audio devices")


def record_session(name: str, session: int, duration: int, device: str = None):
    """Record a single session of voice samples."""
    session_dir = SAMPLES_DIR / name
    session_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = session_dir / f"session-{session:02d}_{timestamp}.wav"

    script_dir = Path(__file__).parent / "prompts"
    
    print(f"\n═══ Recording Session {session}: {name} ═══\n")
    print(f"  Duration: {duration}s ({duration/60:.1f} min)")
    print(f"  Output:   {output}")
    print(f"  Format:   {SAMPLE_RATE}Hz, mono, 16-bit PCM")
    print(f"\n  READ ALOUD the provided prompts at natural pace.")
    print(f"  Speak clearly, vary your tone (excited, calm, questioning).")
    print(f"  Take pauses between sentences.\n")

    # Build arecord command
    cmd = [
        "arecord",
        "-f", "S16_LE",           # 16-bit signed little-endian
        "-r", str(SAMPLE_RATE),   # 48kHz sample rate
        "-c", "1",                # mono
        "-t", "wav",              # WAV format
        "-d", str(duration),      # duration in seconds
    ]
    if device:
        cmd.extend(["-D", device])
    cmd.append(str(output))

    # Show recording prompt topics
    prompts = {
        1: "Introductions, background, casual conversation",
        2: "Narrative reading (news, articles, stories)",
        3: "Technical reading (documentation, explanations)",
        4: "Conversational (Q&A, responses, reactions)",
        5: "Expressive (excited, concerned, thoughtful, humorous)",
        6: "Free-form (whatever you want, natural speech)",
    }
    topic = prompts.get(session, "Free-form conversation")
    print(f"  Topic: {topic}\n")

    try:
        print(f"  Recording in 3...", end="", flush=True)
        time.sleep(1)
        print("2...", end="", flush=True)
        time.sleep(1)
        print("1...", end="", flush=True)
        time.sleep(1)
        print(" GO!\n")

        proc = subprocess.run(cmd)
        if proc.returncode == 0:
            size = output.stat().st_size
            print(f"\n  ✓ Session complete: {output.name}")
            print(f"    Size: {size / 1024 / 1024:.1f} MB")
            print(f"    Duration: {duration}s\n")
            return True
        else:
            print(f"\n  ✗ Recording failed (return code {proc.returncode})\n")
            return False
    except KeyboardInterrupt:
        print("\n\n  Recording interrupted.\n")
        if output.exists():
            output.unlink()
        return False
    except Exception as e:
        print(f"\n  ✗ Error: {e}\n")
        return False


def check_total(name: str):
    """Check total recorded duration for a voice."""
    session_dir = SAMPLES_DIR / name
    if not session_dir.exists():
        print(f"No recordings found for '{name}'")
        return 0
    
    total = 0
    for f in sorted(session_dir.glob("*.wav")):
        dur = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(f)],
            capture_output=True, text=True, timeout=10
        )
        if dur.returncode == 0 and dur.stdout.strip():
            d = float(dur.stdout.strip())
            total += d
            print(f"  {f.name}: {d:.0f}s")

    print(f"\n  Total: {total:.0f}s ({total/60:.1f} min)")
    print(f"  Target: 1800s (30 min) — {'✓ DONE' if total >= 1800 else f'{1800-total:.0f}s more needed'}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Zaya Voice Recorder")
    parser.add_argument("--name", type=str, default="bcloud",
                        help="Voice name")
    parser.add_argument("--session", type=int, default=None,
                        help="Session number (1-6 recommended)")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help=f"Recording duration in seconds (default: {DEFAULT_DURATION})")
    parser.add_argument("--device", type=str, default=None,
                        help="ALSA device (e.g., 'hw:0,0'). Auto-detected if not set.")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices")
    parser.add_argument("--check", action="store_true",
                        help="Check total recorded duration")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.check:
        check_total(args.name)
        return

    if args.session is None:
        # Auto-detect next session number
        session_dir = SAMPLES_DIR / args.name
        if session_dir.exists():
            existing = list(session_dir.glob("session-*.wav"))
            if existing:
                max_n = max(int(f.stem.split("-")[1]) for f in existing)
                args.session = max_n + 1
            else:
                args.session = 1
        else:
            args.session = 1

    record_session(args.name, args.session, args.duration, args.device)

    # Show total
    print("── Total recordings ──")
    check_total(args.name)


if __name__ == "__main__":
    main()
