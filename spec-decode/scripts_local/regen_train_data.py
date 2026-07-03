#!/usr/bin/env python3
"""Local, single-GPU on-policy data regeneration — no SGLang server needed.
Loads Qwen3-0.6B directly via transformers and regenerates assistant turns,
matching generate_train_data.py's output schema so prepare_target_cache.py
can consume it unmodified."""
import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--input-file-path", required=True)
    p.add_argument("--output-file-path", required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    print(f"Loading {args.model} on {'cuda' if torch.cuda.is_available() else 'cpu'}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with open(args.input_file_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        lines = lines[: args.limit]

    out_f = open(args.output_file_path, "w")
    for sample in tqdm(lines, desc="Regenerating"):
        conv = sample.get("conversations")
        if not conv or conv[0].get("role") == "assistant":
            continue
        regenerated = []
        for msg in conv:
            if msg["role"] in ("system", "user"):
                regenerated.append(msg)
                continue
            if msg["role"] == "assistant":
                prompt_text = tok.apply_chat_template(
                    regenerated, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tok(prompt_text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        pad_token_id=tok.eos_token_id,
                    )
                gen_text = tok.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                regenerated.append({"role": "assistant", "content": gen_text})
        sample["conversations"] = regenerated
        sample["status"] = "ok"
        out_f.write(json.dumps(sample) + "\n")
        out_f.flush()
    out_f.close()
    print(f"Wrote regenerated data to {args.output_file_path}")


if __name__ == "__main__":
    main()
