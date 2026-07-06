#!/usr/bin/env python3
"""Quick DSpark acceptance rate test using DeepSpec evaluator internals."""

import os, sys, time, json
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, '/home/bcloud/DeepSpec')
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.common import extract_context_feature
from deepspec.eval.dspark.draft_ops import (
    forward_dspark_draft_block,
    build_dspark_proposal,
    DSparkDraftProposal,
)

DEVICE = "cpu"
DTYPE = torch.bfloat16
TARGET_ID = "Qwen/Qwen3-4B"
DRAFT_CKPT = "/home/bcloud/spec-decode/checkpoints/dspark_qwen3_4b"
BLOCK_SIZE = 7
MAX_NEW = 32
NUM_PROMPTS = 5  # Keep small for CPU speed

print(f"═══ DSpark Acceptance Test ═══")
print(f"Target: {TARGET_ID}")
print(f"Draft:  DSpark (5 layers, Markov head, confidence head)")
print(f"Device: {DEVICE}")
print()

# --- Load models ---
print("Loading target model...")
t0 = time.time()
target = AutoModelForCausalLM.from_pretrained(
    TARGET_ID,
    torch_dtype=DTYPE,
    attn_implementation="sdpa",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to(DEVICE).eval()
print(f"  {time.time()-t0:.0f}s, {sum(p.numel() for p in target.parameters())/1e9:.1f}B params")
print()

print("Loading DSpark draft model...")
t0 = time.time()
draft = Qwen3DSparkModel.from_pretrained(
    DRAFT_CKPT,
    torch_dtype=DTYPE,
    attn_implementation="sdpa",
).to(DEVICE).eval()
print(f"  {time.time()-t0:.0f}s, {sum(p.numel() for p in draft.parameters())/1e6:.1f}M params")
print(f"  Block size: {draft.block_size}, Draft layers: {len(draft.layers)}")
print(f"  Target layers: {draft.target_layer_ids}")
print()

tokenizer = AutoTokenizer.from_pretrained(TARGET_ID, trust_remote_code=True)
print(f"Vocab: {tokenizer.vocab_size}")
print()

# --- Test prompts ---
prompts = [
    "The capital of France is",
    "Python is a programming",
    "The speed of light is approximately",
    "Machine learning is a subset of",
    "The square root of 144 is",
]
prompts = prompts[:NUM_PROMPTS]

# --- Evaluation ---
total_accepted = 0
total_proposed = 0
total_verify = 0
total_tokens = 0

for pi, prompt_text in enumerate(prompts):
    print(f"[{pi+1}/{len(prompts)}] {prompt_text[:40]}...", flush=True)
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]
    generated = inputs.input_ids.clone()
    
    # Initial target forward to get hidden states
    with torch.no_grad():
        out = target(
            **inputs,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    
    context_feature = extract_context_feature(
        out.hidden_states,
        draft.target_layer_ids,
    )
    pkv_draft = DynamicCache()
    
    tokens_gen = 0
    verify_calls = 0
    accepted = 0
    proposed = 0
    
    while tokens_gen < MAX_NEW:
        # Draft proposal: generate block_size tokens
        draft_input_ids = generated[:, -1:]
        # Position IDs must cover the full sequence from the draft's perspective
        # (positions 0 to input_len + tokens_gen)
        total_pos = input_len + tokens_gen
        position_ids = torch.arange(total_pos, device=DEVICE).unsqueeze(0)
        
        # Run draft backbone to get proposal hidden states
        block_hidden = forward_dspark_draft_block(
            model=draft,
            draft_input_ids=draft_input_ids,
            position_ids=position_ids,
            past_key_values_draft=pkv_draft,
            target_hidden_states=context_feature,
            start=position_ids[0, -1].item(),
            block_size=BLOCK_SIZE,
        )
        
        proposal = build_dspark_proposal(
            model=draft,
            draft_input_ids=draft_input_ids,
            block_hidden=block_hidden,
            block_size=BLOCK_SIZE,
            temperature=0.0,  # greedy
            confidence_threshold=0.0,  # no early stop
        )
        
        if proposal.draft_token_count == 0:
            # No draft tokens proposed
            with torch.no_grad():
                single_out = target(
                    input_ids=generated[:, -1:],
                    past_key_values=out.past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            next_token = single_out.logits[0, -1, :].argmax().item()
            generated = torch.cat([generated, torch.tensor([[next_token]], device=DEVICE)], dim=1)
            tokens_gen += 1
            verify_calls += 1
            out = single_out
            context_feature = extract_context_feature(out.hidden_states, draft.target_layer_ids)
            pkv_draft = DynamicCache()
            continue
        
        # Verify with target
        with torch.no_grad():
            verify_out = target(
                input_ids=proposal.verify_input_ids,
                past_key_values=out.past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Rejection sampling (greedy)
        n_accepted = 0
        verify_calls += 1
        
        for i in range(proposal.draft_token_count):
            target_token = verify_out.logits[0, i, :].argmax().item()
            draft_token = proposal.verify_input_ids[0, 1 + i].item()
            
            if draft_token == target_token:
                n_accepted += 1
                tokens_gen += 1
                accepted += 1
                if tokens_gen >= MAX_NEW:
                    break
            else:
                generated = torch.cat([generated, torch.tensor([[target_token]], device=DEVICE)], dim=1)
                tokens_gen += 1
                break
            proposed += 1
        
        # Bonus token (all accepted)
        if n_accepted == proposal.draft_token_count and tokens_gen < MAX_NEW:
            bonus = verify_out.logits[0, proposal.draft_token_count, :].argmax().item()
            generated = torch.cat([generated, torch.tensor([[bonus]], device=DEVICE)], dim=1)
            tokens_gen += 1
        
        # Update context for next round
        if tokens_gen < MAX_NEW:
            with torch.no_grad():
                out = target(
                    input_ids=generated[:, -1:],
                    past_key_values=verify_out.past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            context_feature = extract_context_feature(out.hidden_states, draft.target_layer_ids)
            pkv_draft = DynamicCache()
    
    accept_rate = (accepted / proposed * 100) if proposed > 0 else 0
    speedup = tokens_gen / verify_calls if verify_calls > 0 else 1
    total_accepted += accepted
    total_proposed += proposed
    total_verify += verify_calls
    total_tokens += tokens_gen
    print(f"  Accept: {accept_rate:.0f}%  Speedup: {speedup:.1f}x  Tokens: {tokens_gen}", flush=True)

print()
print("═══ FINAL RESULTS ═══")
all_accept = total_accepted / total_proposed * 100 if total_proposed > 0 else 0
all_speedup = total_tokens / total_verify if total_verify > 0 else 1
print(f"DSpark (Qwen3-4B target):")
print(f"  Acceptance rate: {all_accept:.1f}%")
print(f"  Effective speedup: {all_speedup:.2f}x")
print(f"  Total tokens: {total_tokens}")
print(f"  Verify calls: {total_verify}")
print()
print("For reference (DeepSpec paper, Qwen3-4B):")
print(f"  Eagle3: ~78% acceptance, ~2.5x speedup")
print(f"  DSpark: ~88% acceptance, ~4.2x speedup (+68% over Eagle3)")
print()
print(f"Our Eagle3 (NPU, Qwen3-0.6B) at 94 tok/s baseline:")
print(f"  With Eagle3 acceptance: ~{94 * 2.5:.0f} tok/s")
print(f"  With DSpark acceptance: ~{94 * 4.2:.0f} tok/s (+68%)")
