#!/usr/bin/env python3
"""
Generate ground truth reference traces from HuggingFace FP32 model.

Dumps per-layer hidden states and logits for direct comparison against
NPU engine traces.

Usage:
    python3 tools/hf_ref_forward.py [--model /path/to/hf/model]
                                     [--prompt "text"]
                                     [--output /tmp/trace_hf/]
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np

HF_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)

def stats(name, t, fmt=".4f"):
    t = t.detach().to(torch.float64)
    s = t.sum().item()
    sq = (t*t).sum().item()
    rms = (sq / t.numel()) ** 0.5
    first4 = [f"{x:.4f}" for x in t.flatten()[:4].tolist()]
    print(f"  {name:30s} norm={rms:{fmt}} sum={s:{fmt}} first4={first4}")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=HF_MODEL)
    ap.add_argument("--prompt", default="Hi")
    ap.add_argument("--output", default="/tmp/trace_hf")
    ap.add_argument("--layers", default=None, help="Comma-separated (default: all)")
    args = ap.parse_args()
    
    global torch
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Parse layers
    if args.layers:
        layer_list = [int(x) for x in args.layers.split(",")]
    else:
        # Check model config for num_hidden_layers
        with open(os.path.join(args.model, "config.json")) as f:
            cfg = json.load(f)
        num_layers = cfg.get("num_hidden_layers", 28)
        layer_list = list(range(num_layers))
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model in FP32
    print(f"Loading model from {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32
    )
    model.eval()
    print(f"  Model dtype: {model.dtype}")
    
    # Tokenize
    input_ids = tok(args.prompt, return_tensors="pt").input_ids
    npt = input_ids.shape[1]
    print(f"Prompt: {repr(args.prompt)} ({npt} tokens)")
    print(f"Token IDs: {input_ids[0].tolist()}")
    
    # Save token IDs for NPU comparison
    tok_path = out_dir / "token_ids.txt"
    with open(tok_path, "w") as f:
        for tid in input_ids[0].tolist():
            f.write(f"{tid}\n")
    print(f"Saved token IDs to {tok_path}")
    
    # Forward pass with hidden states
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    
    # hidden_states[0] = embedding output
    # hidden_states[i] = after layer i-1 (residual already applied)
    # hidden_states[-1] = pre-final-norm
    hs = out.hidden_states
    
    # Save embedding
    emb_out = hs[0][0, -1].cpu().numpy().astype(np.float32)
    emb_out.tofile(str(out_dir / "embedding_output.f32"))
    stats("embedding_output", hs[0][0, -1])
    
    # Save per-layer hidden states (last token)
    for li in layer_list:
        if li < len(hs) - 1:
            h = hs[li + 1][0, -1]  # after layer li
            h_np = h.cpu().numpy().astype(np.float32)
            h_np.tofile(str(out_dir / f"layer_{li:02d}_hidden.f32"))
            stats(f"layer_{li}_hidden", h)
    
    # Save final norm
    # Qwen3: model.model.norm; other models: model.norm
    norm_layer = getattr(model.model, 'norm', None) or getattr(model, 'norm', None)
    if norm_layer is not None:
        final_h = norm_layer(hs[-1][0, -1])
    else:
        final_h = hs[-1][0, -1]
    final_np = final_h.detach().cpu().numpy().astype(np.float32)
    final_np.tofile(str(out_dir / "final_norm_output.f32"))
    stats("final_norm_output", final_h.detach())
    
    # Save logits
    logits = out.logits[0, -1]
    logits_np = logits.detach().cpu().numpy().astype(np.float32)
    logits_np.tofile(str(out_dir / "logits.f32"))
    
    # Top-5
    top5 = torch.topk(logits.detach(), 5)
    print(f"\n  Top-5 tokens: {top5.indices.tolist()}")
    print(f"  Top-5 decoded: {[tok.decode([t]) for t in top5.indices.tolist()]}")
    
    # Generate a few tokens greedily for comparison
    with torch.no_grad():
        gen = model.generate(input_ids, max_new_tokens=5, do_sample=False)
    gen_tokens = gen[0, npt:].tolist()
    gen_text = tok.decode(gen_tokens)
    print(f"\n  Greedy continuation tokens: {gen_tokens}")
    print(f"  Greedy continuation text: {repr(gen_text)}")
    
    print(f"\nSaved HF reference traces to {out_dir}")

if __name__ == "__main__":
    main()
