#!/usr/bin/env python3
"""DSpark vs Eagle3: head-to-head acceptance rate comparison.

DSpark: 5 draft layers + Markov head + confidence head (Qwen3-4B target)
Eagle3: 1 draft layer (Qwen3-0.6B target, our NPU)

Measures:
- Acceptance rate (% of draft tokens accepted by target)
- Effective speedup (tokens per verify call)
"""

import os, sys, json, time, math
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from safetensors import safe_open

# --- Paths ---
DSPARK_CKPT = "/home/bcloud/spec-decode/checkpoints/dspark_qwen3_4b"
TARGET_ID = "Qwen/Qwen3-4B"
DEVICE = "cpu"
DTYPE = torch.bfloat16

# --- DeepSpec imports ---
sys.path.insert(0, '/home/bcloud/DeepSpec')
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.common import DSparkForwardOutput, extract_context_feature

print(f"═══ DSpark vs Eagle3 Acceptance Test ═══")
print(f"Target: {TARGET_ID}")
print(f"Draft:  DSpark (5 layers + Markov + confidence)")
print(f"Device: {DEVICE} ({torch.get_num_threads()} threads)")
print()

# --- Load target ---
print("Loading target model (Qwen3-4B)...")
t0 = time.time()
target = AutoModelForCausalLM.from_pretrained(
    TARGET_ID,
    torch_dtype=DTYPE,
    attn_implementation="sdpa",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to(DEVICE).eval()
print(f"  Target loaded: {sum(p.numel() for p in target.parameters())/1e9:.1f}B params in {time.time()-t0:.0f}s")
print()

# --- Load DSpark draft ---
print("Loading DSpark draft model...")
t0 = time.time()
draft = Qwen3DSparkModel.from_pretrained(
    DSPARK_CKPT,
    torch_dtype=DTYPE,
    attn_implementation="sdpa",
).to(DEVICE).eval()
print(f"  Draft loaded: {sum(p.numel() for p in draft.parameters())/1e6:.1f}M params in {time.time()-t0:.0f}s")
print(f"  Block size: {draft.block_size}")
print(f"  Draft layers: {len(draft.layers)}")
print(f"  Target layers: {draft.target_layer_ids}")
print(f"  Hidden size: {draft.config.hidden_size}")
print(f"  Has confidence head: {draft.confidence_head is not None}")
print()

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(TARGET_ID, trust_remote_code=True)
print(f"Tokenizer: {tokenizer.vocab_size} vocab")

# --- Test prompts ---
test_prompts = [
    "The capital of France is",
    "Python is a programming",
    "The speed of light is approximately",
    "Machine learning is a subset of",
    "The square root of 144 is",
    "In the beginning, God created",
    "The theory of relativity was developed by",
    "Water freezes at",
    "The largest planet in our solar system is",
    "DNA is composed of",
    "The Industrial Revolution began in",
    "HTTP status code 404 means",
    "Shakespeare wrote",
    "The Pythagorean theorem states",
    "Photosynthesis converts",
]

print(f"\nUsing {len(test_prompts)} test prompts")
print()

# --- Acceptance test ---
def measure_acceptance(prompts, max_new=32):
    """Measure DSpark speculative decoding acceptance rate."""
    total_draft = 0
    total_accepted = 0
    total_verify_calls = 0
    total_target_tokens = 0
    
    for prompt_idx, prompt_text in enumerate(prompts):
        print(f"  [{prompt_idx+1}/{len(prompts)}] {prompt_text[:40]}...", end=" ", flush=True)
        
        # Tokenize
        inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        input_len = inputs.input_ids.shape[1]
        
        # Run target to get initial hidden states
        with torch.no_grad():
            outputs = target(
                **inputs,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Extract context features for draft
        context_feature = extract_context_feature(
            outputs.hidden_states,
            draft.target_layer_ids,
        )
        
        # Initialize KV cache for draft
        past_key_values_draft = DynamicCache()
        
        # Speculative decoding loop
        generated = input_ids = inputs.input_ids.clone()
        tokens_generated = 0
        verify_calls = 0
        accepted = 0
        proposed = 0
        
        while tokens_generated < max_new:
            # Get target's last token logits
            last_logits = outputs.logits[0, -1, :]
            last_token = last_logits.argmax().item()
            
            # Draft proposal
            with torch.no_grad():
                draft_out = draft(
                    input_ids=generated[:, -1:],
                    past_key_values=past_key_values_draft,
                    context_feature=context_feature,
                    use_cache=True,
                    return_dict=True,
                )
            
            # Draft generates block_size tokens autoregressively
            draft_tokens = []
            draft_logits_list = []
            current_input = generated[:, -1:]
            pkv = past_key_values_draft
            
            for step_idx in range(draft.block_size):
                with torch.no_grad():
                    d_out = draft(
                        input_ids=current_input,
                        past_key_values=pkv,
                        context_feature=context_feature,
                        use_cache=True,
                        return_dict=True,
                    )
                step_logits = d_out.logits[0, -1, :]
                step_token = step_logits.argmax().item()
                draft_tokens.append(step_token)
                draft_logits_list.append(step_logits)
                current_input = torch.tensor([[step_token]], device=DEVICE)
            
            # Verify: run target on draft tokens
            verify_input = torch.cat([generated[:, -1:]] + 
                [torch.tensor([[t]], device=DEVICE) for t in draft_tokens], dim=1)
            
            with torch.no_grad():
                verify_out = target(
                    input_ids=verify_input[:, :-1],  # all but last
                    past_key_values=outputs.past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Check acceptance
            verify_calls += 1
            n_accepted = 0
            for i in range(len(draft_tokens)):
                target_logit = verify_out.logits[0, i, :]
                target_token = target_logit.argmax().item()
                
                if draft_tokens[i] == target_token:
                    # Accept
                    generated = torch.cat([generated, torch.tensor([[draft_tokens[i]]], device=DEVICE)], dim=1)
                    tokens_generated += 1
                    n_accepted += 1
                    accepted += 1
                    if tokens_generated >= max_new:
                        break
                else:
                    # Reject: use target's token
                    generated = torch.cat([generated, torch.tensor([[target_token]], device=DEVICE)], dim=1)
                    tokens_generated += 1
                    break
                proposed += 1
            
            # Bonus token (all accepted)
            if n_accepted == len(draft_tokens) and tokens_generated < max_new:
                bonus_token = verify_out.logits[0, -1, :].argmax().item()
                generated = torch.cat([generated, torch.tensor([[bonus_token]], device=DEVICE)], dim=1)
                tokens_generated += 1
            
            # Update context features for next round
            if tokens_generated < max_new:
                with torch.no_grad():
                    outputs = target(
                        input_ids=generated[:, -1:],
                        past_key_values=verify_out.past_key_values if 'verify_out' in dir() else outputs.past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                context_feature = extract_context_feature(
                    outputs.hidden_states,
                    draft.target_layer_ids,
                )
                past_key_values_draft = DynamicCache()
        
        total_draft += proposed
        total_accepted += accepted
        total_verify_calls += verify_calls
        total_target_tokens += tokens_generated
        
        accept_rate = (accepted / proposed * 100) if proposed > 0 else 0
        speedup = tokens_generated / verify_calls if verify_calls > 0 else 1
        print(f"accept={accept_rate:.0f}% speedup={speedup:.1f}x")
    
    return {
        "acceptance_rate": total_accepted / total_draft if total_draft > 0 else 0,
        "acceptance_pct": total_accepted / total_draft * 100 if total_draft > 0 else 0,
        "effective_speedup": total_target_tokens / total_verify_calls if total_verify_calls > 0 else 1,
        "total_draft_proposed": total_draft,
        "total_accepted": total_accepted,
        "total_verify_calls": total_verify_calls,
        "total_tokens": total_target_tokens,
    }

# Run with small subset first (CPU is slow)
print("Running DSpark acceptance test (15 prompts × 16 tokens each)...")
t_start = time.time()
results = measure_acceptance(test_prompts, max_new=16)
t_elapsed = time.time() - t_start

print()
print("═══ RESULTS ═══")
print(f"DSpark (Qwen3-4B target):")
print(f"  Acceptance rate:  {results['acceptance_pct']:.1f}%")
print(f"  Effective speedup: {results['effective_speedup']:.2f}x")
print(f"  Total tokens:     {results['total_tokens']}")
print(f"  Verify calls:     {results['total_verify_calls']}")
print(f"  Elapsed:          {t_elapsed:.0f}s")
print()

# Reference Eagle3 numbers from paper
print("═══ COMPARISON ═══")
print(f"                     Acceptance    Speedup")
print(f"  Eagle3 (paper):    ~78% (greedy) ~2.5x")
print(f"  DSpark (paper):    ~85-90%       ~3.5-4.5x")
print(f"  DSpark (measured): {results['acceptance_pct']:.1f}%        {results['effective_speedup']:.2f}x")
print()
print("Note: Paper numbers are for larger models (4B-14B) on GPU.")
print("Our NPU uses Qwen3-0.6B target with ~94 tok/s baseline.")
print()
print(f"NPU speedup estimate with DSpark-like acceptance:")
print(f"  Eagle3 target at 94 tok/s + {results['acceptance_pct']:.0f}% acceptance:")
print(f"    ~{94 * results['effective_speedup']:.0f} tok/s effective")
