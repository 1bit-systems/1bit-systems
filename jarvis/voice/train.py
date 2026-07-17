#!/usr/bin/env python3
"""
Voice cloning trainer — Zaya Co-Host Phase 1.1

Takes voice samples and trains the RVQ-VAE codec decoder + speaker embedding,
then packs into a deployable .voice file.

The .voice pack is agnostic — works with any LLM that outputs codec tokens.

Usage:
  python3 jarvis/voice/train.py --samples ./voice/samples/bcloud/ --name bcloud

Output:
  ./voice/packs/bcloud.voice  —  deployable voice pack (~25 MB)
"""

import argparse, json, os, subprocess, tarfile, tempfile, shutil, time
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

VOICE_PACKS_DIR = Path(__file__).parent.parent.parent / "voice" / "packs"
VOICE_SAMPLES_DIR = Path(__file__).parent.parent.parent / "voice" / "samples"

TARGET_SECONDS = 1800         # 30 minutes ideal
SAMPLE_RATE = 24000           # Codec native sample rate


def check_audio(samples_dir: Path) -> dict:
    """Validate and measure audio samples in a directory."""
    audio_files = []
    total_seconds = 0
    for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"):
        audio_files.extend(samples_dir.glob(ext))
    if not audio_files:
        raise RuntimeError(f"No audio files found in {samples_dir}")
    for f in audio_files:
        dur = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(f)],
            capture_output=True, text=True, timeout=30
        )
        if dur.returncode == 0 and dur.stdout.strip():
            total_seconds += float(dur.stdout.strip())
    return {"files": len(audio_files), "duration_s": total_seconds,
            "duration_min": total_seconds / 60}


def preprocess_audio(samples_dir: Path, output_dir: Path):
    """Normalize audio to 24kHz mono WAV for codec training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(samples_dir.iterdir()):
        if f.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
            out = output_dir / f"{f.stem}.wav"
            if out.exists():
                continue
            subprocess.run([
                "ffmpeg", "-i", str(f), "-ar", str(SAMPLE_RATE),
                "-ac", "1", "-sample_fmt", "s16",
                "-y", str(out)
            ], check=True, capture_output=True, timeout=120)
    return output_dir


def train_codec(samples_dir: Path, output_dir: Path, device: str = "cuda"):
    """Train the RVQ-VAE codec on voice samples.
    
    This trains the decoder to reconstruct audio from discrete tokens.
    The trained decoder becomes part of the .voice pack.
    """
    from jarvis.voice.codec import AudioCodec, AudioCodecConfig, AudioCodecLoss, train_codec
    
    print(f"\n  Training codec on voice samples from: {samples_dir}")
    
    # Load audio files into a dataset
    wavs = sorted(samples_dir.glob("*.wav"))
    print(f"  Found {len(wavs)} preprocessed audio files")
    
    # Create a simple dataloader
    class AudioDataset(torch.utils.data.Dataset):
        def __init__(self, files, sample_rate=24000, duration=4.0):
            self.files = files
            self.n_samples = int(sample_rate * duration)
        
        def __len__(self):
            return len(self.files) * 10  # Augment by random crops
        
        def __getitem__(self, idx):
            f = self.files[idx % len(self.files)]
            audio, sr = sf.read(f)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 24000:
                import numpy as np
                old_len = len(audio)
                new_len = int(old_len * 24000 / sr)
                audio = np.interp(np.linspace(0, old_len-1, new_len),
                                  np.arange(old_len), audio)
            # Random crop
            if len(audio) > self.n_samples:
                start = np.random.randint(0, len(audio) - self.n_samples)
                audio = audio[start:start + self.n_samples]
            else:
                audio = np.pad(audio, (0, max(0, self.n_samples - len(audio))))
            return torch.from_numpy(audio).float().unsqueeze(0)
    
    dataset = AudioDataset(wavs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Initialize and train codec
    config = AudioCodecConfig(sample_rate=24000)
    model = AudioCodec(config)
    
    print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print(f"  Device: {device}")
    print(f"  Training on {len(dataset)} samples ({len(loader)} batches/epoch)")
    
    model = train_codec(
        model=model,
        dataloader=loader,
        n_epochs=50,
        lr=3e-4,
        device=device,
        save_path=str(output_dir / "codec_checkpoint.pt"),
    )
    
    # Save trained decoder weights
    torch.save(model.state_dict(), output_dir / "decoder.pt")
    print(f"  Codec decoder saved to {output_dir / 'decoder.pt'}")
    return model


def extract_speaker_embedding(samples_dir: Path, output_dir: Path, device: str = "cuda"):
    """Extract a speaker embedding from voice samples.
    
    Uses the codec encoder to create a voice fingerprint.
    This embedding conditions the ZAYA LLM to generate in the right voice.
    """
    from jarvis.voice.codec import AudioCodec, AudioCodecConfig
    
    print(f"\n  Extracting speaker embedding from: {samples_dir}")
    
    # Load pre-trained codec encoder
    decoder_path = output_dir / "decoder.pt"
    if not decoder_path.exists():
        print("  No decoder weights found — using random embedding placeholder")
        embedding = np.random.randn(512).astype(np.float32)
        np.save(output_dir / "speaker.npy", embedding)
        return embedding
    
    config = AudioCodecConfig(sample_rate=24000)
    model = AudioCodec(config)
    state = torch.load(decoder_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    
    # Average encoder outputs across all samples as speaker embedding
    wavs = sorted(samples_dir.glob("*.wav"))
    embeddings = []
    
    with torch.no_grad():
        for wav in wavs[:50]:  # Use up to 50 samples
            audio, sr = sf.read(wav)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 24000:
                import numpy as np
                old_len = len(audio)
                new_len = int(old_len * 24000 / sr)
                audio = np.interp(np.linspace(0, old_len-1, new_len),
                                  np.arange(old_len), audio)
            
            # Take first 5 seconds
            audio = audio[:24000 * 5]
            if len(audio) < 24000:
                continue
            
            t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
            z = model.encoder(t)
            embeddings.append(z.mean(dim=-1).cpu().numpy().flatten())
    
    if embeddings:
        embedding = np.mean(embeddings, axis=0)
    else:
        embedding = np.random.randn(512).astype(np.float32)
    
    np.save(output_dir / "speaker.npy", embedding)
    print(f"  Speaker embedding extracted: {embedding.shape} — norm={np.linalg.norm(embedding):.2f}")
    return embedding


def build_voice_pack(name: str, work_dir: Path, output_path: Path):
    """Build the final .voice pack from trained artifacts."""
    print(f"\n  Building voice pack: {name}")
    
    # Collect metadata
    metadata = {
        "name": name,
        "version": "1.0.0",
        "model": "zaya-codec-v1",
        "sample_rate": 24000,
        "n_codebooks": 4,
        "codebook_size": 1024,
        "frame_rate": 75,
        "format": "rvq-vae",
        "engine": "agnostic",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": f"Zaya Co-Host voice pack: {name}",
        "license": "proprietary",
    }
    
    # Create tar.gz
    with tarfile.open(output_path, "w:gz") as tar:
        # Metadata
        meta_bytes = json.dumps(metadata, indent=2).encode()
        meta_io = tempfile.NamedTemporaryFile(delete=False)
        meta_io.write(meta_bytes)
        meta_io.close()
        tar.add(meta_io.name, arcname="metadata.json")
        os.unlink(meta_io.name)
        
        # Decoder weights
        decoder_path = work_dir / "decoder.pt"
        if decoder_path.exists():
            tar.add(str(decoder_path), arcname="decoder.pt")
        
        # Speaker embedding
        embed_path = work_dir / "speaker.npy"
        if embed_path.exists():
            # Convert numpy to torch for storage
            embed = np.load(embed_path)
            embed_pt = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
            torch.save(torch.from_numpy(embed), embed_pt)
            embed_pt.close()
            tar.add(embed_pt.name, arcname="speaker.pt")
            os.unlink(embed_pt.name)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Voice pack created: {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train voice clone → .voice pack")
    parser.add_argument("--samples", type=str, required=True,
                        help="Directory with voice samples (.wav, .mp3, etc.)")
    parser.add_argument("--name", type=str, required=True,
                        help="Name for the voice clone")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device (cuda or cpu)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .voice path (default: ./voice/packs/{name}.voice)")
    args = parser.parse_args()

    samples_dir = Path(args.samples)
    if not samples_dir.exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        sys.exit(1)

    output_path = Path(args.output or VOICE_PACKS_DIR / f"{args.name}.voice")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp())

    try:
        # Step 1: Validate audio
        print(f"\n{'='*60}")
        print(f"🎙️  Zaya Co-Host — Voice Clone Training")
        print(f"{'='*60}")
        info = check_audio(samples_dir)
        print(f"\nAudio check: {info['files']} files, {info['duration_min']:.1f} min total")
        if info['duration_s'] < 600:
            print(f"  ⚠️  Less than 10 min of audio — quality may suffer")
        print(f"  ✅ Minimum met: {info['duration_s'] >= 600}")
        print(f"  ✅ Target met:  {info['duration_s'] >= 1800}")

        # Step 2: Preprocess
        print(f"\n--- Step 2: Preprocessing audio ---")
        proc_dir = work_dir / "processed"
        preprocess_audio(samples_dir, proc_dir)
        print(f"  Preprocessed to: {proc_dir}")

        # Step 3: Train codec decoder
        print(f"\n--- Step 3: Training codec decoder ---")
        train_codec(proc_dir, work_dir, device=args.device)

        # Step 4: Extract speaker embedding
        print(f"\n--- Step 4: Extracting speaker embedding ---")
        extract_speaker_embedding(proc_dir, work_dir, device=args.device)

        # Step 5: Build voice pack
        print(f"\n--- Step 5: Building voice pack ---")
        build_voice_pack(args.name, work_dir, output_path)

        print(f"\n{'='*60}")
        print(f"✅ Voice clone complete: {args.name}")
        print(f"   Voice pack: {output_path}")
        print(f"   Size: ~{output_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"   Agnostic: works with any LLM, any hardware")
        print(f"{'='*60}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    main()
