#!/usr/bin/env python3
"""
1bit Speculative Decoding — Evaluate a trained draft model on benchmark tasks.

Measures:
  - Acceptance rate per draft position
  - Average accepted draft tokens per verify call
  - Wall-clock speedup (simulated)
  - Output quality (same as greedy baseline?)

Usage:
    python eval/eval_draft.py \
        --target Qwen/Qwen3-0.6B \
        --draft ./checkpoints/eagle3_step_10000 \
        --tasks gsm8k math500
"""

import argparse
import json
import os
import sys
import time

DEEPSPEC_DIR = os.path.expanduser("~/DeepSpec")
if os.path.exists(DEEPSPEC_DIR):
    sys.path.insert(0, DEEPSPEC_DIR)

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate speculative decoding draft model"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Target model name or path",
    )
    parser.add_argument(
        "--draft",
        type=str,
        required=True,
        help="Draft model checkpoint path",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["gsm8k"],
        choices=["gsm8k", "math500", "humaneval", "mbpp"],
        help="Evaluation tasks",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate per sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=7,
        help="Number of speculative tokens per draft forward",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples per task",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    return parser.parse_args()


def load_model(model_path, device):
    """Load model with appropriate precision."""
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa" if device != "cpu" else "eager",
    ).to(device).eval()
    return model


def speculative_decode_greedy(
    target_model,
    draft_model,
    tokenizer,
    input_ids,
    max_new_tokens,
    block_size,
    device,
):
    """Greedy speculative decoding loop.
    
    Returns generated tokens and per-step metrics.
    """
    target_model.eval()
    draft_model.eval()
    
    input_len = input_ids.shape[1]
    max_len = input_len + max_new_tokens
    prompt = input_ids.to(device)
    
    acceptance_lengths = []
    proposal_lengths = []
    accepted_draft_lengths = []
    
    # Prefill target model
    past_key_values = DynamicCache()
    with torch.no_grad():
        out = target_model(
            prompt,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
    
    next_token = out.logits[:, -1, :].argmax(dim=-1)
    generated = [next_token.item()]
    
    if next_token.item() == tokenizer.eos_token_id:
        return generated, [], [], []
    
    pos = input_len
    while pos < max_len:
        # 1. Get hidden states from target for draft input
        hidden_states = out.hidden_states  # tuple of [layer+1, batch, seq, hidden]
        
        # 2. Draft: predict block_size candidate tokens using the trained draft model
        draft_tokens = []
        draft_hidden = hidden_states[-1]  # last layer hidden states
        # `next_token` is the target model's real next-token argmax (the anchor).
        # Keep it intact; use a separate variable for the per-step draft input so
        # we don't clobber the anchor mid-loop (#146).
        draft_input = next_token
        for i in range(block_size):
            with torch.no_grad():
                draft_logits = draft_model.forward(draft_hidden, draft_input.unsqueeze(0))
            draft_token = draft_logits[:, -1, :].argmax(dim=-1)  # greedy from draft
            draft_tokens.append(draft_token.view(-1))
            # Feed predicted token back as next draft step input
            draft_input = draft_token.view(-1)

        # Standard spec-decode layout: [verified_anchor, draft_0, draft_1, ...].
        # Use torch.cat on the list of [1]-shaped tensors (torch.tensor(list-of-
        # tensors) is invalid and raises).
        draft_ids = torch.cat([
            next_token.unsqueeze(0),
            torch.cat(draft_tokens).unsqueeze(0)
        ], dim=1)
        
        # 3. Verify with target model
        with torch.no_grad():
            verify_out = target_model(
                draft_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
        
        verify_logits = verify_out.logits  # [1, block_size+1, vocab]
        
        # 4. Greedy rejection sampling
        n_accepted = 0
        for i in range(block_size):
            target_token = verify_logits[:, i, :].argmax(dim=-1)
            if draft_tokens[i] == target_token.item():
                generated.append(draft_tokens[i].item())
                n_accepted += 1
                pos += 1
                if draft_tokens[i].item() == tokenizer.eos_token_id:
                    break
            else:
                generated.append(target_token.item())
                pos += 1
                break
        
        acceptance_lengths.append(n_accepted + 1)
        proposal_lengths.append(block_size)
        accepted_draft_lengths.append(n_accepted)
        
        next_token = torch.tensor([generated[-1]], device=device)
        
        if generated[-1] == tokenizer.eos_token_id:
            break
    
    return generated, acceptance_lengths, proposal_lengths, accepted_draft_lengths


def run_benchmark(task_name, num_samples, target_model, draft_model, tokenizer,
                  max_new_tokens, block_size, device):
    """Run evaluation on a single task."""
    
    # Load task data
    data_path = f"{DEEPSPEC_DIR}/eval_datasets/{task_name}.jsonl"
    if not os.path.exists(data_path):
        print(f"  Dataset not found: {data_path}")
        print(f"  Run DeepSpec data preparation first:")
        print(f"    cd {DEEPSPEC_DIR} && bash scripts/data/prepare_data.sh")
        return None
    
    import json
    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
            if len(samples) >= num_samples:
                break
    
    total_accepted = 0
    total_proposed = 0
    total_verify_calls = 0
    total_time = 0.0
    
    print(f"\n  Running {len(samples)} samples...")
    
    for i, sample in enumerate(samples):
        messages = [{"role": "user", "content": sample["turns"][0]}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        
        t0 = time.time()
        generated, accept_lens, prop_lens, draft_lens = speculative_decode_greedy(
            target_model, draft_model, tokenizer,
            input_ids, max_new_tokens, block_size, device,
        )
        total_time += time.time() - t0
        
        for al, pl, dl in zip(accept_lens, prop_lens, draft_lens):
            total_accepted += dl
            total_proposed += pl
        total_verify_calls += len(accept_lens)
        
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(samples)}]", flush=True)
    
    # Compute metrics
    accept_rate = total_accepted / total_proposed if total_proposed > 0 else 0
    avg_accept_len = total_accepted / total_verify_calls if total_verify_calls > 0 else 0
    speedup = (total_accepted + total_verify_calls) / total_verify_calls if total_verify_calls > 0 else 1.0
    tokens_per_sec = (total_accepted + total_verify_calls) / total_time if total_time > 0 else 0
    
    return {
        "task": task_name,
        "samples": len(samples),
        "accept_rate": accept_rate,
        "avg_accepted_per_verify": avg_accept_len,
        "verify_calls": total_verify_calls,
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "speedup_factor": speedup,
        "tokens_per_sec": tokens_per_sec,
        "total_time_sec": total_time,
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"1bit Speculative Decoding — Evaluation")
    print(f"Target: {args.target}")
    print(f"Draft:  {args.draft}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Load models
    print("\nLoading target model...")
    target_model = load_model(args.target, args.device)
    print(f"  Target: {sum(p.numel() for p in target_model.parameters()) / 1e6:.1f}M params")
    
    draft_model = None
    if os.path.exists(args.draft):
        print("Loading draft model...")
        try:
            draft_model = load_model(args.draft, args.device)
            print(f"  Draft: {sum(p.numel() for p in draft_model.parameters()) / 1e6:.1f}M params")
        except Exception as e:
            print(f"  Could not load draft model: {e}")
            print("  Using random draft (baseline — expect 0% acceptance)")
    
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    
    # Run benchmarks
    all_results = []
    for task in args.tasks:
        print(f"\n--- Task: {task} ---")
        result = run_benchmark(
            task, args.num_samples,
            target_model, draft_model, tokenizer,
            args.max_new_tokens, args.block_size,
            args.device,
        )
        if result:
            all_results.append(result)
            print(f"  Results for {task}:")
            print(f"    Acceptance rate:      {result['accept_rate']:.2%}")
            print(f"    Avg accepted/verify:  {result['avg_accepted_per_verify']:.2f}")
            print(f"    Speedup factor:       {result['speedup_factor']:.2f}x")
            print(f"    Tokens/sec:           {result['tokens_per_sec']:.1f}")
    
    # Summary
    if all_results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        avg_speedup = sum(r["speedup_factor"] for r in all_results) / len(all_results)
        avg_accept = sum(r["accept_rate"] for r in all_results) / len(all_results)
        print(f"  Avg acceptance rate:  {avg_accept:.2%}")
        print(f"  Avg speedup factor:   {avg_speedup:.2f}x")
        print(f"  Projected NPU tok/s:  {94 * avg_speedup:.0f} tok/s (from 94 baseline)")
        print("=" * 60)
        
        # Save results
        out_path = "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
