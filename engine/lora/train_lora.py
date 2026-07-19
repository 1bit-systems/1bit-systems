#!/usr/bin/env python3
"""
train_lora.py — Train a single LoRA adapter on Qwen3-0.6B for the NPU engine.

Usage:
  python3 train_lora.py \
    --dataset alpaca \
    --split 100 \
    --rank 8 \
    --lr 3e-4 \
    --seed 42 \
    --target all \
    --name alpaca-rank8-lr3e-4

Output: engine/lora/adapters/<name>/  (adapter_model + config)
"""

import os, sys, json, time, argparse, random, shutil
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, Dataset
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
ADAPTER_DIR = os.path.expanduser("~/1bit-systems/engine/lora/adapters")
MAX_SEQ_LEN = 512
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def format_instruction(example: dict, dataset: str) -> str:
    """Format a dataset example into a text string for training."""
    # Extract base name from hf: prefix and :config suffix
    base = dataset
    if base.startswith("hf:"):
        base = base[3:]  # strip hf: prefix
    if ":" in base:
        base = base.split(":")[0]  # strip config suffix
    base = base.split("/")[-1]  # get last part of namespace
    
    if base in ("alpaca",):
        inst = example.get("instruction", "")
        inp = example.get("input", "")
        out = example.get("output", "")
        if inp:
            return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        else:
            return f"### Instruction:\n{inst}\n\n### Response:\n{out}"
    elif base in ("code_alpaca", "codealpaca"):
        inst = example.get("instruction", "")
        inp = example.get("input", "")
        out = example.get("output", "")
        if inp:
            return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        else:
            return f"### Instruction:\n{inst}\n\n### Response:\n{out}"
    elif base in ("dolly", "databricks-dolly-15k"):
        inst = example.get("instruction", "")
        ctx = example.get("context", "")
        res = example.get("response", "")
        if ctx:
            return f"### Instruction:\n{inst}\n\n### Context:\n{ctx}\n\n### Response:\n{res}"
        else:
            return f"### Instruction:\n{inst}\n\n### Response:\n{res}"
    elif base == "tldr":
        # TLDR summarization
        post = example.get("content", example.get("post", ""))
        summary = example.get("summary", example.get("tldr", ""))
        return f"### Article:\n{post}\n\n### TLDR:\n{summary}"
    elif base in ("gsm8k", "math"):
        q = example.get("question", "")
        a = example.get("answer", "")
        return f"### Question:\n{q}\n\n### Answer:\n{a}"
    elif base == "oasst1":
        # OpenAssistant
        conv = example.get("conversation", [])
        if isinstance(conv, list) and len(conv) >= 2:
            q = conv[0].get("content", conv[0].get("text", "")) if isinstance(conv[0], dict) else str(conv[0])
            a = conv[1].get("content", conv[1].get("text", "")) if isinstance(conv[1], dict) else str(conv[1])
            return f"### User:\n{q}\n\n### Assistant:\n{a}"
        return None
    elif base == "cnn_dailymail":
        article = example.get("article", "")
        highlights = example.get("highlights", "")
        return f"### Article:\n{article}\n\n### Summary:\n{highlights}"
    else:
        # Generic: try common field names
        inp = example.get("instruction") or example.get("question") or example.get("input") or example.get("text") or example.get("content") or example.get("document") or str(example)
        out_raw = example.get("output") or example.get("answer") or example.get("response") or example.get("completion") or example.get("summary") or example.get("label") or example.get("answers") or ""
        # Handle nested answer dicts (e.g. SQuAD answers={'text': ['...'], 'answer_start': [0]})
        if isinstance(out_raw, dict) and "text" in out_raw:
            out = out_raw["text"][0] if isinstance(out_raw["text"], list) and out_raw["text"] else str(out_raw)
        elif isinstance(out_raw, list):
            out = out_raw[0] if out_raw else ""
        else:
            out = str(out_raw)
        if isinstance(inp, str) and isinstance(out, str) and len(inp) > 5:
            return f"### Input:\n{inp}\n\n### Output:\n{out}"
        return None

def load_and_prepare_dataset(dataset_name: str, split_pct: float = 100, seed: int = 42):
    """Load a dataset, filter to valid examples, and take a slice."""
    log.info(f"Loading dataset: {dataset_name} (split={split_pct}%, seed={seed})")
    
    try:
        if dataset_name == "alpaca":
            ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        elif dataset_name == "code_alpaca":
            ds = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train", trust_remote_code=True)
        elif dataset_name == "dolly":
            ds = load_dataset("databricks/databricks-dolly-15k", split="train", trust_remote_code=True)
        elif dataset_name == "tldr":
            ds = load_dataset("trl-lib/tldr", split="train", trust_remote_code=True)
        elif dataset_name == "gsm8k":
            ds = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
        elif dataset_name == "oasst1":
            ds = load_dataset("OpenAssistant/oasst1", split="train", trust_remote_code=True)
        elif dataset_name == "cnn_dailymail":
            ds = load_dataset("cnn_dailymail", "3.0.0", split="train", trust_remote_code=True)
        elif dataset_name == "medical_qa":
            ds = load_dataset("bigbio/med_qa", split="train", trust_remote_code=True)
        elif dataset_name == "sparql":
            ds = load_dataset("mradermacher/sparql-model-era-lora-128-qwen3-0.6b-GGUF", split="train", trust_remote_code=True)
        elif dataset_name.startswith("hf:"):
            hf_name = dataset_name[3:]
            # Support "dataset:config" syntax for datasets needing a config
            if ":" in hf_name:
                parts = hf_name.split(":", 1)
                ds = load_dataset(parts[0], parts[1], split="train", trust_remote_code=True)
            else:
                ds = load_dataset(hf_name, split="train", trust_remote_code=True)
        else:
            # Try direct name
            ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
    except Exception as e:
        log.error(f"Failed to load dataset '{dataset_name}': {e}")
        return None, 0
    
    # Format examples
    texts = []
    total = len(ds)
    for ex in ds:
        formatted = format_instruction(ex, dataset_name)
        if formatted and len(formatted) > 20:
            texts.append(formatted)
    
    log.info(f"  Formatted {len(texts)}/{total} valid examples")
    
    if len(texts) == 0:
        return None, 0
    
    # Slice
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(texts))
    n = max(1, int(len(texts) * split_pct / 100))
    texts = [texts[i] for i in indices[:n]]
    
    log.info(f"  Using {len(texts)} examples ({split_pct}% slice)")
    
    # Create HF dataset
    ds_out = Dataset.from_dict({"text": texts})
    return ds_out, len(texts)

def train_adapter(
    dataset_name: str,
    split_pct: float = 100,
    rank: int = 8,
    lr: float = 3e-4,
    seed: int = 42,
    target: str = "all",
    name: str = None,
    num_epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
):
    """Train one LoRA adapter and save it."""
    set_seed(seed)
    
    # Name
    if name is None:
        name = f"{dataset_name.replace('/', '_')}-r{rank}-lr{lr}-s{seed}-t{target}"
    
    output_dir = os.path.join(ADAPTER_DIR, name)
    if os.path.exists(output_dir):
        log.warning(f"Adapter '{name}' already exists at {output_dir}, skipping")
        return {"name": name, "status": "skipped", "path": output_dir}
    
    # Load dataset
    ds, n = load_and_prepare_dataset(dataset_name, split_pct, seed)
    if ds is None or n < 10:
        log.warning(f"Not enough data for {dataset_name}, skipping")
        return {"name": name, "status": "failed", "error": "insufficient_data"}
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    def tokenize(ex):
        tokens = tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    ds = ds.map(tokenize, remove_columns=["text"], num_proc=2)
    
    # Load model
    log.info(f"Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    
    # Select target modules
    if target == "attention":
        modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    elif target == "ffn":
        modules = ['gate_proj', 'up_proj', 'down_proj']
    else:
        modules = TARGET_MODULES
    
    # LoRA config
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=1,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=seed,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )
    
    # Train
    log.info(f"Training adapter '{name}' (rank={rank}, lr={lr}, dataset={dataset_name}, n={n})")
    t0 = time.time()
    trainer.train()
    t_elapsed = time.time() - t0
    
    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    meta = {
        "name": name,
        "base_model": "Qwen3-0.6B",
        "dataset": dataset_name,
        "dataset_size": n,
        "rank": rank,
        "lr": lr,
        "seed": seed,
        "target": target,
        "epochs": num_epochs,
        "train_time_s": round(t_elapsed, 1),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "status": "completed",
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    log.info(f"✓ Saved adapter '{name}' ({t_elapsed:.1f}s, {meta['trainable_params']:,} params)")
    log.info(f"  Path: {output_dir}")
    
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for Qwen3-0.6B")
    parser.add_argument("--dataset", type=str, default="alpaca", help="Dataset name")
    parser.add_argument("--split", type=float, default=100, help="percent of dataset to use")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target", choices=["all", "attention", "ffn"], default="all", help="Module target")
    parser.add_argument("--name", type=str, default=None, help="Adapter name (auto-generated if not set)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()
    
    meta = train_adapter(
        dataset_name=args.dataset,
        split_pct=args.split,
        rank=args.rank,
        lr=args.lr,
        seed=args.seed,
        target=args.target,
        name=args.name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )
    
    print(f"\n=== Done: {json.dumps(meta, indent=2)} ===")

if __name__ == "__main__":
    main()
