#!/usr/bin/env python3
"""One-shot tokenizer using HuggingFace tokenizers library.

Usage:
  python3 tokenize.py --encode "Hello, world!"     → 9707 11 1879 0
  python3 tokenize.py --decode 9707 11 1879 0      → Hello, world!
  python3 tokenize.py --model path/to/tokenizer.json --encode "text"

Uses the HuggingFace tokenizers library (Rust backend) for correct
BPE encoding/decoding matching the reference implementation.
"""
import sys, argparse
from pathlib import Path
from tokenizers import Tokenizer


def load_tokenizer(path=None):
    if path is None:
        path = str(Path.home() / ".config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json")
    return Tokenizer.from_file(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--encode", nargs="*", help="Text to tokenize")
    parser.add_argument("--decode", nargs="+", type=int, help="Token IDs to decode")
    args = parser.parse_args()

    tok = load_tokenizer(args.model)

    if args.encode is not None:
        text = " ".join(args.encode) if args.encode else sys.stdin.read().strip()
        ids = tok.encode(text).ids
        print(" ".join(str(i) for i in ids))
    elif args.decode is not None:
        text = tok.decode(args.decode)
        print(text)
    else:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            ids = tok.encode(line).ids
            print(" ".join(str(i) for i in ids))


if __name__ == "__main__":
    main()
