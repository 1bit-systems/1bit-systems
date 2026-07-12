#!/usr/bin/env python3
"""Reference forward pass for a HuggingFace causal LM (default: Qwen3-0.6B).

Dumps per-layer hidden-state stats (sum, sum-of-squares/RMS, first 4 values)
for the last prompt token at a fixed set of layer checkpoints, for direct
comparison against matching debug prints added to the Zig fused-engine
(engine/fusion/fused_execute.zig's prefill()/firstDecodeToken()) to find
exactly where the two implementations' numbers diverge.

Requires a real torch + transformers install. On this machine that's
/home/bcloud/venv-hf (the system default `python3`'s `torch`/`transformers`
are broken stub installs missing their compiled extensions) -- run with:

    /home/bcloud/venv-hf/bin/python3 tools/reference_forward.py \\
        [--model /path/to/hf/model/dir] [--prompt "text"] [--layers 0,1,2,13,26,27]

Used to find the RoPE table layout bug, the QK-norm-never-loaded bug, and
(most recently) to confirm the remaining post-#56 incoherence is Q4NX
quantization error accumulation rather than a further code bug: the
embedding output (stored near-full-precision as BF16) matches this
reference exactly, but every I8-quantized-weight layer after it diverges
further, reaching ~30x the reference RMS by the final layer.
"""
import argparse
import sys

DEFAULT_MODEL = (
    "/home/bcloud/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)


def stats(name, t):
    import torch
    t = t.detach().to(torch.float64)
    s = t.sum().item()
    sq = (t * t).sum().item()
    rms = (sq / t.numel()) ** 0.5
    first4 = t.flatten()[:4].tolist()
    print(f"{name}: sum={s:.6f} sq={sq:.6f} rms={rms:.6f} first4={first4}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Path to HF model dir (config.json + *.safetensors)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layers", default="0,1,2,13,26,27", help="Comma-separated layer indices to checkpoint")
    ap.add_argument("--max-new-tokens", type=int, default=10)
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        print("Need a real torch+transformers install. Try:", file=sys.stderr)
        print("  /home/bcloud/venv-hf/bin/python3 " + " ".join(sys.argv), file=sys.stderr)
        sys.exit(1)

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    input_ids = tok(args.prompt, return_tensors="pt").input_ids
    print("prompt:", repr(args.prompt))
    print("prompt_tokens:", input_ids[0].tolist())

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)

    # hidden_states[0] is the embedding output; hidden_states[i] for i=1..NC
    # is the output after layer i-1 (residual already added); hidden_states[-1]
    # is the pre-final-norm hidden state. We apply final norm ourselves to
    # match what the Zig code calls "final RMSNorm output" right before lmHead.
    hs = out.hidden_states
    stats("embedding_output (last token)", hs[0][0, -1])
    for li in layers:
        if li < len(hs) - 1:
            stats(f"after_layer_{li} (last token)", hs[li + 1][0, -1])

    final_hidden = model.model.norm(hs[-1][0, -1])
    stats("final_norm_output (last token)", final_hidden)

    logits = out.logits[0, -1]
    top5 = torch.topk(logits, 5)
    print("top5_logits:", top5.values.tolist())
    print("top5_token_ids:", top5.indices.tolist())
    for tid in top5.indices.tolist():
        print(f"  {tid}: {tok.decode([tid])!r}")

    with torch.no_grad():
        gen = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen_tokens = gen[0, input_ids.shape[1]:].tolist()
    print("greedy_continuation_tokens:", gen_tokens)
    print("greedy_continuation_text:", repr(tok.decode(gen_tokens)))


if __name__ == "__main__":
    main()
