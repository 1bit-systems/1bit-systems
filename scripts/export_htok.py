#!/usr/bin/env python3
"""Convert HuggingFace tokenizer.json → .htok binary for RCPP tokenizer.

Output format (from include/rocm_cpp/tokenizer.h):
  magic     : char[4] = "HTOK"
  version   : u32 LE = 1
  vocab_size: u32 LE
  merges_n  : u32 LE
  bos_id    : i32 LE
  eos_id    : i32 LE
  --- vocab table, vocab_size entries, in id order ---
    token_len : u16 LE
    token_bytes: u8[token_len]
  --- merges table, merges_n entries ---
    a_id    : i32 LE
    b_id    : i32 LE
    new_id  : i32 LE

Usage:
    python3 scripts/export_htok.py --input tokenizer.json --output tokenizer.htok
"""

import json
import struct
import sys
import os
import argparse


def find_tokenizer_json():
    """Search common locations for a tokenizer.json."""
    paths = [
        os.path.expanduser("~/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json"),
        os.path.expanduser("~/.config/flm/models/Qwen3-1.7B-NPU2/tokenizer.json"),
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"),
    ]
    for p in paths:
        if "*" in p:
            import glob
            matches = glob.glob(p)
            if matches:
                return matches[0]
        elif os.path.exists(p):
            return p
    return None


def convert(input_path: str, output_path: str):
    print(f"Reading {input_path}...")
    with open(input_path) as f:
        tok = json.load(f)

    model = tok.get("model", {})
    vocab_raw = model.get("vocab", {})  # str -> id
    merges_raw = model.get("merges", [])  # ["a b", ...]

    if not vocab_raw or not merges_raw:
        print("ERROR: tokenizer.json must have BPE model with vocab and merges")
        sys.exit(1)

    # Build id->token map sorted by id
    id_to_token = sorted(vocab_raw.items(), key=lambda x: x[1])  # (token_str, id)
    
    # Build token->id map for merge lookup
    token_to_id = vocab_raw

    # Parse merges (format: "a b")
    parsed_merges = []
    for line in merges_raw:
        if isinstance(line, str):
            parts = line.split()
            if len(parts) >= 2:
                a, b = parts[0], parts[1]
                if a in token_to_id and b in token_to_id:
                    parsed_merges.append((token_to_id[a], token_to_id[b]))
                else:
                    print(f"WARNING: merge '{a} {b}' references unknown tokens")
        elif isinstance(line, list) and len(line) >= 2:
            a, b = str(line[0]), str(line[1])
            if a in token_to_id and b in token_to_id:
                parsed_merges.append((token_to_id[a], token_to_id[b]))

    vocab_size = len(id_to_token)
    merges_n = len(parsed_merges)
    bos_id = tok.get("added_tokens", [{}])[0].get("id", 1) if tok.get("added_tokens") else 1
    eos_id = tok.get("added_tokens", [{}])[1].get("id", 2) if len(tok.get("added_tokens", [])) > 1 else 2

    # For Qwen3 models: BOS=151643, EOS=151645, PAD=151643
    for at in tok.get("added_tokens", []):
        if at.get("content") == "<|endoftext|>":
            eos_id = at["id"]
        elif at.get("content") in ("<|im_start|>", "<|begin_of_text|>"):
            bos_id = at["id"]

    print(f"  Vocab: {vocab_size} tokens")
    print(f"  Merges: {merges_n}")
    print(f"  BOS: {bos_id}, EOS: {eos_id}")

    with open(output_path, "wb") as f:
        # Header
        f.write(b"HTOK")
        f.write(struct.pack("<I", 1))      # version
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", merges_n))
        f.write(struct.pack("<i", bos_id))
        f.write(struct.pack("<i", eos_id))

        # Vocab table (sorted by id)
        for token_str, tid in id_to_token:
            token_bytes = token_str.encode("utf-8")
            if len(token_bytes) > 65535:
                print(f"WARNING: token #{tid} too long ({len(token_bytes)} bytes), truncating")
                token_bytes = token_bytes[:65535]
            f.write(struct.pack("<H", len(token_bytes)))
            f.write(token_bytes)

        # Merges table
        for a_id, b_id in parsed_merges:
            # new_id is the rank (index) in the merge list
            new_id = len(parsed_merges)  # placeholder, not critical for decoder-only
            f.write(struct.pack("<i", a_id))
            f.write(struct.pack("<i", b_id))
            f.write(struct.pack("<i", new_id))

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert tokenizer.json to .htok format")
    parser.add_argument("--input", help="Path to tokenizer.json")
    parser.add_argument("--output", default="/tmp/tokenizer.htok", help="Output .htok path")
    args = parser.parse_args()

    input_path = args.input or find_tokenizer_json()
    if not input_path:
        print("ERROR: No tokenizer.json found. Specify --input.")
        sys.exit(1)

    convert(input_path, args.output)


if __name__ == "__main__":
    main()
