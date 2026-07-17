#!/usr/bin/env python3
"""
ZAYA Voice Dataset — Prepare voice cloning training data.

Records, transcribes, and tokenizes voice samples for ZAYA
fine-tuning. This is Step 2 of the voice cloning pipeline.

Usage:
    from zaya_audio.dataset import VoiceDataset, prepare_voice_dataset

    # After recording ~1hr of voice samples:
    dataset = prepare_voice_dataset(
        audio_dir="path/to/wavs/",
        transcript_file="path/to/transcripts.json",
    )
    
    # Each sample: {"text": "...", "audio_tokens": (N_q, T)}
"""

import os
import json
import glob
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .codec import AudioCodec, AudioCodecConfig

logger = logging.getLogger(__name__)


@dataclass
class VoiceSample:
    """A single voice sample for training."""
    text: str                          # Transcription
    audio_path: str                    # Path to WAV file
    duration: float = 0.0              # Duration in seconds
    speaker_id: str = "default"        # Speaker identifier


class VoiceDataset(Dataset):
    """Dataset of voice samples for training the voice adapter.
    
    Each sample is a dict with:
        - input_ids: tokenized text
        - audio_tokens: encoded audio tokens (N_q, T')
        - attention_mask: mask for padding
    """
    
    def __init__(
        self,
        samples: List[VoiceSample],
        codec: AudioCodec,
        tokenizer=None,
        max_text_length: int = 256,
        max_audio_frames: int = 750,   # ~10 seconds at 75 fps
        device: str = "cpu",
    ):
        self.samples = samples
        self.codec = codec
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.max_audio_frames = max_audio_frames
        self.device = device
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load and encode audio
        audio = self.codec.load_audio(sample.audio_path)
        audio = audio.to(self.device)
        
        # Pad/crop to minimum length
        min_samples = self.codec.config.hop_length * 10  # At least 10 frames
        if audio.shape[-1] < min_samples:
            audio = torch.nn.functional.pad(audio, (0, min_samples - audio.shape[-1]))
        
        # Encode to tokens
        with torch.no_grad():
            audio_tokens = self.codec.encode(audio)  # (1, N_q, T')
        
        audio_tokens = audio_tokens.squeeze(0)  # (N_q, T')
        
        # Truncate if too long
        if audio_tokens.shape[-1] > self.max_audio_frames:
            audio_tokens = audio_tokens[:, :self.max_audio_frames]
        
        # Tokenize text (simple char-level if no tokenizer)
        if self.tokenizer:
            text_ids = self.tokenizer.encode(sample.text)
            if len(text_ids) > self.max_text_length:
                text_ids = text_ids[:self.max_text_length]
            input_ids = torch.tensor(text_ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
        else:
            # Simple char-level tokenization as fallback
            chars = list(sample.text.lower())
            char_ids = [ord(c) % 256 for c in chars]
            if len(char_ids) > self.max_text_length:
                char_ids = char_ids[:self.max_text_length]
            input_ids = torch.tensor(char_ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_tokens": audio_tokens,
            "text": sample.text,
            "audio_path": sample.audio_path,
        }


def collate_voice_samples(batch: List[Dict]) -> Dict:
    """Collate function for voice dataset batching."""
    # Pad text
    max_text_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids_list = []
    attention_mask_list = []
    
    for item in batch:
        pad_len = max_text_len - item["input_ids"].shape[0]
        input_ids_list.append(
            torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=0)
        )
        attention_mask_list.append(
            torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0)
        )
    
    # Pad audio tokens
    max_audio_len = max(item["audio_tokens"].shape[1] for item in batch)
    audio_tokens_list = []
    
    for item in batch:
        pad_len = max_audio_len - item["audio_tokens"].shape[1]
        audio_tokens_list.append(
            torch.nn.functional.pad(item["audio_tokens"], (0, pad_len), value=0)
        )
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "audio_tokens": torch.stack(audio_tokens_list),
        "texts": [item["text"] for item in batch],
        "audio_paths": [item["audio_path"] for item in batch],
    }


def prepare_voice_dataset(
    audio_dir: str,
    transcript_file: Optional[str] = None,
    codec: Optional[AudioCodec] = None,
    device: str = "cuda",
) -> VoiceDataset:
    """Prepare a voice dataset from audio files.
    
    Args:
        audio_dir: Directory containing WAV files
        transcript_file: Optional JSON file mapping filenames to text
        codec: Audio codec (creates default if None)
        device: Device to use
        
    Returns:
        VoiceDataset ready for training
    """
    if codec is None:
        codec = AudioCodec(AudioCodecConfig()).to(device)
    
    # Load transcripts
    transcripts = {}
    if transcript_file and os.path.exists(transcript_file):
        with open(transcript_file) as f:
            transcripts = json.load(f)
    
    # Find audio files
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not audio_files:
        audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.mp3")))
    
    samples = []
    for af in audio_files:
        basename = os.path.splitext(os.path.basename(af))[0]
        text = transcripts.get(basename, transcripts.get(af, ""))
        
        # Get duration
        import soundfile as sf
        info = sf.info(af)
        
        samples.append(VoiceSample(
            text=text,
            audio_path=af,
            duration=info.duration,
        ))
    
    logger.info(f"Loaded {len(samples)} voice samples from {audio_dir}")
    
    return VoiceDataset(samples=samples, codec=codec, device=device)


# ─── Recording utility ──────────────────────────────────────────────────────

def record_voice_dataset(
    output_dir: str,
    prompts: List[str],
    sample_rate: int = 24000,
):
    """Record voice samples from prompts.
    
    This is a placeholder — you'll record using your preferred tool.
    
    Args:
        output_dir: Where to save WAV files
        prompts: List of prompts to read
        sample_rate: Recording sample rate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    transcripts = {}
    for i, prompt in enumerate(prompts):
        filename = f"voice_sample_{i:04d}.wav"
        filepath = os.path.join(output_dir, filename)
        
        print(f"\n[{i+1}/{len(prompts)}] Read this prompt:")
        print(f"  \"{prompt}\"")
        print(f"  Saving to: {filepath}")
        print(f"  Press Enter when done recording...")
        input()
        
        transcripts[filename] = prompt
    
    # Save transcripts
    with open(os.path.join(output_dir, "transcripts.json"), "w") as f:
        json.dump(transcripts, f, indent=2)
    
    print(f"\nSaved {len(prompts)} recordings to {output_dir}")
    print(f"Transcripts saved to {os.path.join(output_dir, 'transcripts.json')}")


# ─── Default prompt set ─────────────────────────────────────────────────────

DEFAULT_PROMPTS = [
    # Phonetically balanced sentences
    "The quick brown fox jumps over the lazy dog.",
    "She sells sea shells by the sea shore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers.",
    "I have a dream that one day this nation will rise up.",
    "To be or not to be, that is the question.",
    "The only thing we have to fear is fear itself.",
    "Ask not what your country can do for you.",
    "Four score and seven years ago our fathers brought forth on this continent.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    # Numbers and dates
    "My phone number is five five five, zero one two three.",
    "Today is July fifteenth, twenty twenty six.",
    "The temperature is seventy two degrees Fahrenheit.",
    # Technical terms
    "The API endpoint returns JSON formatted data.",
    "Please deploy the latest commit to the production server.",
    "Git merge conflicts need to be resolved before pushing.",
    "The neural network has forty layers of hybrid attention.",
    "Training on AMD ROCm gives us full control over the stack.",
    # Emotional range
    "I am absolutely thrilled with this result!",
    "Well, that's disappointing. Let me try again.",
    "Are you sure about that?",
    "Congratulations on your achievement!",
    # -thearchitect specific
    "Hello, I am the architect. How can I help you today?",
    "I have analyzed your codebase and found three issues.",
    "Your deployment is complete. Everything is running smoothly.",
    "Let me review that pull request for you.",
    "I've trained my voice on your speech patterns.",
    "This is running on AMD hardware with no cloud dependencies.",
    "Your monthly subscription gives you full access to my capabilities.",
]
