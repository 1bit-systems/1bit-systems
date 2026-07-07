#!/usr/bin/env python3
"""
prepare_npu_inputs.py — Tokenize training data and prepare inputs for the NPU capture tool.

Reads JSONL training data, tokenizes with the Qwen3 tokenizer, and writes
a flat binary file that capture_npu_hidden can read without needing a tokenizer.

Also shows which training examples are valid (non-empty after tokenization).

Output binary format:
  [num_examples: i32]
  For each example:
    [input_len: i32]
    [input_ids: input_len × i32]

Usage:
  source train-venv/bin/activate
  python3 tools/prepare_npu_inputs.py \
    train_data_10k/perfectblend_train_regen.jsonl \
    npu_inputs.bin \
    --max 200 --max-tokens 256
"""
import os, sys, json, struct, argparse
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

def main():
    parser = argparse.ArgumentParser(description='Prepare NPU inputs')
    parser.add_argument('jsonl_path', help='Path to JSONL training data')
    parser.add_argument('output_path', help='Output binary path')
    parser.add_argument('--max', type=int, default=200, help='Max examples')
    parser.add_argument('--max-tokens', type=int, default=256, help='Max tokens per prompt')
    parser.add_argument('--min-tokens', type=int, default=8, help='Min tokens per prompt')
    args = parser.parse_args()

    print(f"═══ Prepare NPU Inputs ═══")
    print(f"Tokenizer: {MODEL_NAME}")
    print(f"JSONL:     {args.jsonl_path}")
    print(f"Output:    {args.output_path}")
    print(f"Max:       {args.max} examples")
    print(f"Tokens:    {args.max_tokens} max, {args.min_tokens} min")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  EOS token:  {tokenizer.eos_token_id}")
    
    # Read JSONL
    print(f"\nReading JSONL...")
    with open(args.jsonl_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"  {len(lines)} lines")
    
    # Tokenize each line
    examples = []
    for idx, line in enumerate(lines):
        if len(examples) >= args.max:
            break
        
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # Extract text from conversations
        text = ""
        if 'conversations' in item:
            for msg in item['conversations']:
                if msg.get('role') == 'user' and msg.get('content'):
                    text = msg['content']
                    break
            if not text:
                # Try assistant fallback
                for msg in item['conversations']:
                    if msg.get('role') == 'assistant' and msg.get('content'):
                        text = msg['content']
                        break
        elif 'messages' in item:
            for msg in item['messages']:
                if msg.get('role') == 'user' and msg.get('content'):
                    text = msg['content']
                    break
        
        if not text:
            continue
        
        # Tokenize
        tokens = tokenizer.encode(
            text,
            truncation=True,
            max_length=args.max_tokens,
            add_special_tokens=True
        )
        
        if len(tokens) < args.min_tokens:
            continue
        
        examples.append(tokens)
        
        if (idx + 1) % 500 == 0:
            print(f"  Tokenized {idx + 1}/{len(lines)}... ({len(examples)} valid so far)")
    
    print(f"\n  Valid examples: {len(examples)}")
    
    # Write binary
    with open(args.output_path, 'wb') as f:
        # Header: num examples
        f.write(struct.pack('<i', len(examples)))
        
        for tokens in examples:
            # input_len
            f.write(struct.pack('<i', len(tokens)))
            # input_ids
            f.write(struct.pack(f'<{len(tokens)}i', *tokens))
    
    file_size = os.path.getsize(args.output_path)
    print(f"\n═══ Written ═══")
    print(f"  {args.output_path} ({file_size / 1024 / 1024:.1f} MB)")
    print(f"  {len(examples)} examples")
    print(f"  Mean tokens: {sum(len(e) for e in examples) / len(examples):.0f}")
    print(f"\nNext:")
    print(f"  sudo ./build/capture_npu_hidden {args.output_path} npu_hidden_cache.bin")

if __name__ == '__main__':
    main()
