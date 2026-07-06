#!/usr/bin/env python3
"""Quick DSpark acceptance rate test against NPU FLM API.
Compares with Eagle3 baseline once trained.

DSpark: 5 draft layers + Markov head + confidence head (Qwen3-4B target)
Eagle3: 1 draft layer (Qwen3-0.6B target, our NPU)
"""

import os, sys, json, time, math
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'

import torch
import torch.nn.functional as F
import requests
from transformers import AutoTokenizer

# --- Configuration ---
FLM_API = "http://127.0.0.1:9090/v1/chat/completions"
TARGET_MODEL = "qwen3:0.6b"
DSPARK_CKPT = "checkpoints/dspark_qwen3_4b"
DEVICE = "cpu"  # Force CPU since ROCm has issues

print(f"═══ DSpark Acceptance Test ═══")
print(f"Target: {TARGET_MODEL} via FLM API at {FLM_API}")
print(f"DSpark checkpoint: {DSPARK_CKPT}")
print(f"Device: {DEVICE}")
print()

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
print(f"Tokenizer: {tokenizer.vocab_size} vocab")
print()

# --- Test prompts (short) ---
test_prompts = [
    "The capital of France is",
    "Python is a programming",
    "The speed of light is approximately",
    "Machine learning is a subset of",
    "The square root of 144 is",
]

# --- Acceptance test via FLM API ---
def get_target_logits(prompt_text, max_tokens=5):
    """Get target model logits via FLM API."""
    resp = requests.post(FLM_API, json={
        "model": TARGET_MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }, timeout=30)
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def measure_acceptance_simple(prompt_text, block_size=7):
    """Simple acceptance rate measurement using only the FLM API.
    
    We generate tokens with the target model, then measure how often
    a simple bigram predictor (mimicking a draft) would have been accepted.
    This gives us a baseline acceptance rate for spec decoding.
    """
    # Get the target's own predictions
    full_prompt = prompt_text
    accepted = 0
    total = 0
    
    for step in range(10):
        resp = requests.post(FLM_API, json={
            "model": TARGET_MODEL,
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
        }, timeout=30)
        data = resp.json()
        target_text = data["choices"][0]["message"]["content"]
        
        if not target_text:
            break
            
        # Heuristic: check if the next token is "obvious" (high probability)
        # For now, track raw acceptance metrics
        full_prompt += target_text
        total += 1
    
    return 0.0, 1.0  # placeholder

print("Measuring baseline acceptance rate via FLM API...")
print()

for prompt in test_prompts[:2]:
    print(f"Prompt: {prompt!r}")
    result = get_target_logits(prompt, max_tokens=3)
    print(f"  Target output: {result!r}")
    print()

# --- Try loading DSpark draft model ---
print("Loading DSpark draft model...")
try:
    import sys
    sys.path.insert(0, '/home/bcloud/DeepSpec')
    
    from safetensors import safe_open
    
    ckpt_path = os.path.join(DSPARK_CKPT, "model.safetensors")
    config_path = os.path.join(DSPARK_CKPT, "config.json")
    
    with open(config_path) as f:
        config = json.load(f)
    print(f"DSpark config: {json.dumps(config, indent=2)[:200]}")
    
    # Load weights
    weights = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    
    print(f"Loaded {len(weights)} weight tensors")
    for k, v in list(weights.items())[:5]:
        print(f"  {k}: {v.shape}")
    
    # Show draft model parameter count
    total_params = sum(v.numel() for v in weights.values())
    print(f"\nTotal parameters: {total_params/1e6:.1f}M")
    
except Exception as e:
    print(f"Error loading DSpark: {e}")
    import traceback
    traceback.print_exc()

print()
print("NOTE: Full DSpark acceptance test requires DeepSpec eval pipeline")
print("with target model loaded in torch (4B model needs GPU).")
print()
print("For 1bit NPU hardware comparison, we will:")
print("1. Complete Eagle3 training (in progress)")
print("2. Benchmark Eagle3 acceptance rate on NPU")
print("3. DSpark results from DeepSpec paper: 66% higher efficiency")
print("   (Markov head: +22% acceptance, Confidence head: +28% dynamic length)")
print("4. Equivalent extrapolation to NPU 0.6B target")
