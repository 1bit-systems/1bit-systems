#!/usr/bin/env python3
"""Local, single-GPU on-policy data regeneration — no SGLang server needed.
Loads Qwen3-0.6B directly via transformers and regenerates assistant turns,
matching generate_train_data.py's output schema so prepare_target_cache.py
can consume it unmodified. Batches across samples (round-robin by turn index)
for throughput; most rows in perfectblend are single-turn so this batches
nearly the whole dataset in round 1, falling back to smaller batches for the
multi-turn tail. Batches are sorted by prompt length before chunking — with
left-padding to the longest item, a batch mixing short and long prompts
wastes (and can OOM on) padding for the short ones; sorting keeps each
chunk's lengths close together. generate() is also wrapped with an
OOM-retry-by-halving fallback so one unusually long chunk degrades to a
smaller batch instead of killing an unattended multi-hour run."""
import argparse
import gc
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--input-file-path", required=True)
    p.add_argument("--output-file-path", required=True)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    print(f"Loading {args.model} on {'cuda' if torch.cuda.is_available() else 'cpu'}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with open(args.input_file_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        lines = lines[: args.limit]

    active = []
    for sample in lines:
        conv = sample.get("conversations")
        if not conv or conv[0].get("role") == "assistant":
            continue
        active.append({"sample": sample, "conv": conv, "idx": 0, "regenerated": []})

    done = []
    total = len(active)
    pbar = tqdm(total=total, desc="Regenerating")
    while active:
        batch_prompts, batch_items, still_active = [], [], []
        for item in active:
            if item.get("skip"):
                pbar.update(1)
                continue
            conv, idx, regenerated = item["conv"], item["idx"], item["regenerated"]
            while idx < len(conv) and conv[idx]["role"] != "assistant":
                regenerated.append(conv[idx])
                idx += 1
            if idx >= len(conv):
                if not item.get("finalized"):
                    item["finalized"] = True
                    item["sample"]["conversations"] = regenerated
                    item["sample"]["status"] = "ok"
                    done.append(item["sample"])
                    pbar.update(1)
                continue
            item["idx"] = idx
            prompt_text = tok.apply_chat_template(
                regenerated, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            batch_prompts.append(prompt_text)
            batch_items.append(item)
            still_active.append(item)
        active = still_active
        if not batch_prompts:
            break

        # Sort by character length (cheap O(1) proxy, no tokenizer calls needed) so each
        # fixed-size chunk has similar-length prompts — minimizes left-padding waste and
        # avoids one long outlier blowing up an otherwise-short batch's memory footprint.
        order = sorted(range(len(batch_prompts)), key=lambda i: len(batch_prompts[i]))
        batch_prompts = [batch_prompts[i] for i in order]
        batch_items = [batch_items[i] for i in order]

        def flush_completed(items):
            # Any item whose turn just finished AND has no more turns left is fully done —
            # move it to `done` and write incrementally so a crash mid-round (a giant round
            # can take hours) doesn't lose everything computed so far.
            newly_done = False
            for item in items:
                if item["idx"] >= len(item["conv"]) and not item.get("finalized"):
                    item["finalized"] = True
                    item["sample"]["conversations"] = item["regenerated"]
                    item["sample"]["status"] = "ok"
                    done.append(item["sample"])
                    pbar.update(1)
                    newly_done = True
            if newly_done:
                with open(args.output_file_path, "w") as out_f:
                    for s in done:
                        out_f.write(json.dumps(s) + "\n")

        def generate_chunk(prompts, items, batch_size):
            if batch_size < 1:
                raise RuntimeError("batch_size fell below 1 during OOM backoff")
            for bstart in range(0, len(prompts), batch_size):
                sub_prompts = prompts[bstart: bstart + batch_size]
                sub_items = items[bstart: bstart + batch_size]
                inputs = tok(sub_prompts, return_tensors="pt", padding=True).to(model.device)
                try:
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            pad_token_id=tok.pad_token_id,
                        )
                except torch.OutOfMemoryError:
                    del inputs
                    gc.collect()
                    torch.cuda.empty_cache()
                    if len(sub_prompts) == 1:
                        print(f"  OOM on a single sample (chars={len(sub_prompts[0])}) — skipping it", flush=True)
                        sub_items[0]["skip"] = True
                        continue
                    print(f"  OOM at batch_size={len(sub_prompts)}, retrying at {len(sub_prompts)//2}", flush=True)
                    generate_chunk(sub_prompts, sub_items, len(sub_prompts) // 2)
                    continue
                in_len = inputs["input_ids"].shape[1]
                for i, item in enumerate(sub_items):
                    gen_text = tok.decode(out[i][in_len:], skip_special_tokens=True)
                    item["regenerated"].append({"role": "assistant", "content": gen_text})
                    item["idx"] += 1
                flush_completed(sub_items)
                print(f"  chunk done: {bstart + len(sub_prompts)}/{len(prompts)} in this round "
                      f"({len(done)}/{total} total)", flush=True)

        generate_chunk(batch_prompts, batch_items, args.batch_size)
    pbar.close()

    with open(args.output_file_path, "w") as out_f:
        for s in done:
            out_f.write(json.dumps(s) + "\n")
    print(f"Wrote {len(done)} regenerated samples to {args.output_file_path}")


if __name__ == "__main__":
    main()
