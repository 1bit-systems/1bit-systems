#!/usr/bin/env python3
"""Export HF tokenizer.json → compact .tok binary for fast Zig loading.

Output format (.tok):
  [4 bytes]  magic: "ZTOK"
  [4 bytes]  version: 1 (u32 LE)
  [4 bytes]  vocab_size (u32 LE)
  [4 bytes]  merges_count (u32 LE)
  [4 bytes]  unk_token_id (u32 LE)
  [4 bytes]  bos_token_id (u32 LE, or 0xFFFFFFFF if none)
  [4 bytes]  eos_token_id (u32 LE, or 0xFFFFFFFF if none)
  ── vocab entries (vocab_size × variable-length) ──
  For each entry:
    [4 bytes]  token_id (u32 LE)
    [2 bytes]  token_str_len (u16 LE)
    [N bytes]  token_str (UTF-8)
  ── merge entries (merges_count × 8 bytes) ──
  For each entry:
    [4 bytes]  left_rank (u32 LE)  — index into vocab sorted by token_id
    [4 bytes]  right_rank (u32 LE)

Usage: python3 export_tokenizer.py --input tokenizer.json --output tokenizer.tok
"""
import json, os, struct, sys, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = args.input or os.path.expanduser("~/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json")
    output_path = args.output or os.path.expanduser("~/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.tok")

    with open(input_path) as f:
        tok = json.load(f)

    model = tok.get("model", {})
    vocab = model.get("vocab", {})   # str → id
    merges_raw = model.get("merges", [])  # [["a","b"], ...]

    # Sort vocab by id
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    vocab_size = len(sorted_vocab)
    merges_count = len(merges_raw)

    # Build ID → rank map for vocab
    id_to_rank = {entry[1]: i for i, entry in enumerate(sorted_vocab)}  # token_id → sorted position

    # Build merge data: for each merge, store the sorted ranks of left/right tokens
    merge_data = []
    for m in merges_raw:
        left = list(m) if isinstance(m, list) else m.split()
        if len(left) != 2:
            continue
        left_str, right_str = left[0], left[1]
        l_id = vocab.get(left_str)
        r_id = vocab.get(right_str)
        if l_id is not None and r_id is not None:
            l_rank = id_to_rank.get(l_id, 0xFFFFFFFF)
            r_rank = id_to_rank.get(r_id, 0xFFFFFFFF)
            merge_data.append((l_rank, r_rank))

    # Look up special tokens
    added = {t["id"]: t["content"] for t in tok.get("added_tokens", [])}
    unk_id = model.get("unk_token_id", 3)
    bos_id = next((t["id"] for t in tok.get("added_tokens", []) if t["content"] == "<|im_start|>"), 0xFFFFFFFF)
    eos_id = next((t["id"] for t in tok.get("added_tokens", []) if t["content"] == "<|im_end|>"), 0xFFFFFFFF)

    # Write binary
    with open(output_path, "wb") as f:
        # Header
        f.write(b"ZTOK")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", len(merge_data)))
        f.write(struct.pack("<I", unk_id))
        f.write(struct.pack("<I", bos_id))
        f.write(struct.pack("<I", eos_id))

        # Vocab entries
        for token_str, token_id in sorted_vocab:
            enc = token_str.encode("utf-8")
            f.write(struct.pack("<I", token_id))
            f.write(struct.pack("<H", len(enc)))
            f.write(enc)

        # Merge entries
        for l_rank, r_rank in merge_data:
            f.write(struct.pack("<I", l_rank))
            f.write(struct.pack("<I", r_rank))

    print(f"Wrote {vocab_size} vocab entries, {len(merge_data)} merges → {output_path} "
          f"({os.path.getsize(output_path)} bytes)")

if __name__ == "__main__":
    main()
