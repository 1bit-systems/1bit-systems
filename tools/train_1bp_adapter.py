from peft import get_peft_model_state_dict
#!/usr/bin/env python3
"""
train_1bp_adapter.py — Train LoRA adapters for 1BP format models on AMD MI300X

Trains LoRA adapters on AMD datacenter GPUs (MI300X), exports to a simple binary
format that the 1bit.systems C++ inference engine can load at runtime alongside
the 1BP base model.

Usage:
  # Train on MI300X (takes ~30 min for 7B model, ~$1)
  python3 tools/train_1bp_adapter.py \
    --model Qwen/Qwen3-4B \
    --dataset yahmna/alpaca-cleaned \
    --output adapters/qwen3-4b-alpaca \
    --rank 16 \
    --epochs 3

  # Train all 10 models (batch job, ~$4 total)
  python3 tools/train_1bp_adapter.py --batch-file models/catalog/1bp_models.json
"""
import os, sys, json, time, struct, argparse
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA adapters for 1BP models on AMD MI300X")
    p.add_argument("--model", default="Qwen/Qwen3-4B", help="HF model name or 1BP base path")
    p.add_argument("--dataset", default="yahma/alpaca-cleaned", help="Training dataset")
    p.add_argument("--output", default="adapters/default", help="Output adapter directory")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    p.add_argument("--target-modules", nargs="+", default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                   help="Target modules for LoRA")
    p.add_argument("--batch-file", help="JSON file with list of training jobs")
    p.add_argument("--dry-run", action="store_true", help="Print config and exit")
    return p.parse_args()

def get_arch_modules(arch):
    """Return target modules for a given architecture string."""
    arch_map = {
        "qwen2": ["q_proj","k_proj","v_proj","o_proj"],
        "qwen3": ["q_proj","k_proj","v_proj","o_proj"],
        "llama": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "gemma": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "zamba": ["q_proj","k_proj","v_proj","o_proj"],  # shared attention only
        "mamba": [],  # Mamba1 SSM — no LoRA target (no attention/FFN)
        "zamba2": [],  # Mamba2 — no attention/FFN (SSM only)
        "phi": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    }
    for key, modules in arch_map.items():
        if key in arch.lower():
            return modules
    return ["q_proj","k_proj","v_proj","o_proj"]

def get_adapter_size(model_params, rank, target_modules_count):
    """Estimate adapter file size."""
    # Each LoRA pair (A, B) per target module: 2 * params * rank floats
    params_per_module = model_params / target_modules_count if target_modules_count > 0 else 0
    # Rough: A has (in_features × rank) params, B has (rank × out_features) params
    # For transformer, in_features ≈ out_features ≈ sqrt(params_per_module)
    dim = int(np.sqrt(params_per_module / 3)) if target_modules_count > 0 else 0
    bytes_per_adapter = target_modules_count * 2 * dim * rank * 4  # float32
    return bytes_per_adapter

def train_adapter(args):
    """Main training loop."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         1BP Adapter Training Pipeline — AMD MI300X          ║
╚══════════════════════════════════════════════════════════════╝
  Model:            {args.model}
  Dataset:          {args.dataset}
  LoRA rank:        {args.rank}
  Target modules:   {', '.join(args.target_modules)}
  Epochs:           {args.epochs}
  Batch size:       {args.batch_size}
  Max length:       {args.max_length}
  Output:           {args.output}
""")

    if args.dry_run:
        print("  [dry-run mode — exiting]")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    # Detect GPUs
    n_gpu = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpu)] if n_gpu > 0 else []
    print(f"  GPUs: {n_gpu} — {', '.join(gpu_names) if gpu_names else 'none (using CPU)'}")
    print(f"  ROCm: {torch.version.hip if hasattr(torch, 'version') and hasattr(torch.version, 'hip') else 'unknown'}")
    print()

    # Architecture detection
    arch = "unknown"
    try:
        # Quick arch detection from HF config
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(args.model, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        arch = cfg.get("model_type", cfg.get("architectures", ["unknown"])[0])
        # Only auto-detect modules if not explicitly provided
        if not args.target_modules or args.target_modules == argparse.Namespace(**{}).target_modules:
            args.target_modules = get_arch_modules(arch)
        print(f"  Detected architecture: {arch}")
        print(f"  Target modules: {', '.join(args.target_modules)}")
    except:
        print(f"  Architecture: {arch} (using manual target modules)")

    if not args.target_modules:
        print(f"  ⚠  No trainable modules for architecture '{arch}' — LoRA not applicable.")
        print(f"     This model type (e.g. Mamba1/Mamba2) needs full fine-tune, not LoRA.")
        print(f"     Skipping.")
        return

    # ── Load model ──
    t0 = time.time()
    print(f"  Loading {args.model}...")
    compute_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto" if n_gpu > 0 else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    model_total = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {model_total/1e9:.2f}B params in {time.time()-t0:.1f}s")
    if n_gpu > 0:
        print(f"  VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ── Apply LoRA ──
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=args.target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # ── Load dataset ──
    print(f"\n  Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")

    def format_alpaca(ex):
        i = ex.get("instruction", "")
        inp = ex.get("input", "")
        o = ex.get("output", "")
        if inp:
            text = f"### Instruction:\n{i}\n\n### Input:\n{inp}\n\n### Response:\n{o}"
        else:
            text = f"### Instruction:\n{i}\n\n### Response:\n{o}"
        return {"text": text}

    print(f"  Formatting dataset...")
    ds = ds.map(format_alpaca, remove_columns=ds.column_names)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(ex):
        tok = tokenizer(ex["text"], truncation=True, max_length=args.max_length, padding="max_length", return_tensors="pt")
        tok["labels"] = tok["input_ids"].clone()
        tok["labels"][tok["labels"] == tokenizer.pad_token_id] = -100
        # Strip batch dimension (return_tensors adds one)
        return {k: v[0] for k, v in tok.items()}

    ds = ds.map(tokenize, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    ds = ds.train_test_split(test_size=0.01, seed=42)
    train_ds, eval_ds = ds["train"], ds["test"]
    print(f"  Train: {len(train_ds)} samples, Eval: {len(eval_ds)} samples")

    # ── Training args ──
    from transformers import TrainingArguments, Trainer, default_data_collator

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(1, len(train_ds) // args.batch_size // 10),
        save_strategy="steps",
        save_steps=max(1, len(train_ds) // args.batch_size // 5),
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        bf16=False,
        tf32=False,
        max_grad_norm=1.0,
        warmup_ratio=0.0,
        weight_decay=0.0,
        gradient_checkpointing=False,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if n_gpu > 1 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
    )

    # ── Train ──
    print(f"\n  Starting training ({args.epochs} epochs)...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.0f}s ({train_time/60:.1f} min)")

    # ── Save adapter ──
    print(f"\n  Saving adapter to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Also export to our native binary format
    export_to_native_format(model, output_dir, args.rank, args.target_modules)

    # Print cost estimate
    if n_gpu > 0:
        gpu_hours = train_time / 3600
        if "MI300" in gpu_names[0] or "mi300" in gpu_names[0].lower():
            cost = gpu_hours * 1.99
            print(f"\n  💰 MI300X cost: ${cost:.2f} ({gpu_hours:.2f} GPU-hours @ $1.99/hr)")
        else:
            print(f"\n  ⏱  {gpu_hours:.2f} GPU-hours")

    print(f"\n  ✅ Adapter saved to {output_dir}/")
    print(f"     Use with: unified_server -m <model> --adapter {output_dir}/adapter.bin")

def export_to_native_format(model, output_dir, rank, target_modules):
    """Export LoRA weights to our native binary format for C++ inference engine."""
    import json, time
    import torch
    state_dict = get_peft_model_state_dict(model)
    
    metadata = {
        "format": "1bp-lora-v1",
        "rank": rank,
        "modules": {},
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    adapter_data = bytearray()
    offset = 0

    for key, tensor in state_dict.items():
        # Parse key: base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight
        parts = key.split(".")
        # Extract layer number and module name
        try:
            layer_idx = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue
        
        # Determine module name and LoRA type (A or B)
        module_name = None
        lora_type = None
        for i, p in enumerate(parts):
            if p in target_modules:
                module_name = p
                lora_type = parts[i + 1] if i + 1 < len(parts) else "A"
                break
        
        if not module_name:
            continue

        weight_np = tensor.cpu().to(torch.float32).numpy()
        module_key = f"layers.{layer_idx}.{module_name}.{lora_type}"
        metadata["modules"][module_key] = {
            "offset": offset,
            "shape": list(weight_np.shape),
            "size": weight_np.nbytes,
        }
        adapter_data.extend(weight_np.tobytes())
        offset += weight_np.nbytes

    # Write binary adapter file
    with open(f"{output_dir}/adapter.bin", "wb") as f:
        f.write(adapter_data)

    # Write metadata
    # Use .1bp.json suffix to avoid overwriting HF PEFT's adapter_config.json
    with open(f"{output_dir}/adapter_config.1bp.json", "w") as f:
        json.dump(metadata, f, indent=2)

    total_mb = offset / (1024 * 1024)
    print(f"  Exported {len(metadata['modules'])} LoRA modules to native format")
    print(f"  Adapter size: {total_mb:.2f} MB")

def batch_train(batch_file, is_dry_run=False):
    """Train adapters for multiple models from a batch config."""
    with open(batch_file) as f:
        data = json.load(f)
    
    # jobs can be top-level array or {jobs: [...]}
    if isinstance(data, dict):
        jobs = [j for j in data.get("jobs", []) if isinstance(j, dict)]
    else:
        jobs = [j for j in data if isinstance(j, dict)]
    
    print(f"Batch training: {len(jobs)} jobs\n")
    for i, job in enumerate(jobs):
        print(f"\n{'='*60}")
        print(f"Job {i+1}/{len(jobs)}: {job.get('name', job['model'])}")
        print(f"{'='*60}")
        
        job_args = {
            "model": job["model"],
            "dataset": job.get("dataset", "yahma/alpaca-cleaned"),
            "output": job.get("output", f"adapters/{job['model'].split('/')[-1].lower()}"),
            "rank": job.get("rank", 16),
            "epochs": job.get("epochs", 3),
            "batch_size": job.get("batch_size", 4),
            "lr": job.get("lr", 2e-4),
            "max_length": job.get("max_length", 512),
            "target_modules": job.get("target_modules", None),
            "batch_file": None,
            "dry_run": is_dry_run,
        }
        import argparse as ap
        a = ap.Namespace(**job_args)
        if a.target_modules is None:
            a.target_modules = get_arch_modules(job.get("arch", ""))
        train_adapter(a)

if __name__ == "__main__":
    args = parse_args()
    if args.batch_file:
        batch_train(args.batch_file, is_dry_run=args.dry_run)
    else:
        train_adapter(args)
