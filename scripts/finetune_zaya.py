#!/usr/bin/env python3
"""
Fine-tune Zyphra ZAYA1-8B with QLoRA on AMD ROCm (Radeon 8060S)
=============================================================
Usage:
    ./finetune_zaya.py                              # train with Alpaca
    ./finetune_zaya.py --dataset your/dataset         # custom dataset
    ./finetune_zaya.py --push-to-hub your-username    # upload to HF
"""

import os, sys, time, argparse, json, math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import transformers

# Monkey-patch: transformers' _is_mlx tries to load Apple's libmlx.so which
# doesn't exist on AMD Linux. This is triggered by accelerate's recursive
# convert_to_fp32 on model outputs. Short-circuit the mlx check.
import transformers.utils.generic as tf_utils
tf_utils._is_mlx_available = False
def _is_mlx_noop(x):
    return False
tf_utils._is_mlx = _is_mlx_noop
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


@dataclass
class ScriptArgs:
    model_id: str = "Zyphra/ZAYA1-8B"
    dataset: str = "yahma/alpaca-cleaned"
    max_seq_length: int = 512
    output_dir: str = "/tmp/zaya-finetune"
    push_to_hub: Optional[str] = None

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"

    # Training
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    max_steps: int = 200
    logging_steps: int = 10
    save_steps: int = 100
    warmup_ratio: float = 0.03
    save_total_limit: int = 2
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_8bit"

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


def format_alpaca(example):
    if example.get("input"):
        text = (
            f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        text = (
            f"Below is an instruction that describes a task.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": text}


def tokenize_fn(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)


class ZayaTrainer(Trainer):
    """Custom Trainer that disables MoE auxiliary loss for ZAYA."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Force output_router_logits=False to skip MoE aux loss
        model.config.output_router_logits = False
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main():
    parser = HfArgumentParser(ScriptArgs)
    args = parser.parse_args_into_dataclasses()[0]

    print("=" * 60)
    print(f"ZAYA1-8B QLoRA Fine-Tune on ROCm")
    print(f"Model: {args.model_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    props = torch.cuda.get_device_properties(0)
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    print("=" * 60)

    # ─── Quantization ──────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )

    # ─── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading {args.model_id}...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Disable MoE router logits at config level
    model.config.output_router_logits = False

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )

    t1 = time.time()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded in {t1-t0:.1f}s — {total_params/1e9:.2f}B total")
    print(f"GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ─── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── LoRA ───────────────────────────────────────────────────────────────
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    print(f"LoRA modules: {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA after prepare_model_for_kbit_training
    model = get_peft_model(model, lora_config)
    model.config.output_router_logits = False  # reassert after PEFT wrapper

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable/1e6:.2f}M ({trainable/total_params*100:.2f}%)")

    # ─── Dataset ────────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")

    if args.dataset == "yahma/alpaca-cleaned":
        dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)

    dataset = dataset.map(lambda x: tokenize_fn(x, tokenizer), remove_columns=["text"])
    dataset = dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")

    # ─── Data collator ─────────────────────────────────────────────────────
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ─── Training args ──────────────────────────────────────────────────────
    hub_model_id = f"{args.push_to_hub}/ZAYA1-8B-QLoRA" if args.push_to_hub else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=1,
        hub_model_id=hub_model_id,
        push_to_hub=bool(args.push_to_hub),
        hub_private_repo=True,
        hub_strategy="end",
        eval_strategy="steps",
        eval_steps=100,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
    )

    # ─── Trainer ────────────────────────────────────────────────────────────
    trainer = ZayaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    # ─── Train ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"Batch: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} (eff. {args.per_device_train_batch_size * args.gradient_accumulation_steps})")
    print(f"Steps: {args.max_steps if args.max_steps > 0 else 'full epoch'}")
    print("=" * 60)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # ─── Save ───────────────────────────────────────────────────────────────
    print(f"\n✓ Training completed in {train_time/60:.1f} min")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # ─── Push to Hub ────────────────────────────────────────────────────────
    if args.push_to_hub:
        print(f"\nPushing to Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id, private=True)
        tokenizer.push_to_hub(hub_model_id, private=True)

        with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as f:
            json.dump({
                "model": args.model_id,
                "dataset": args.dataset,
                "train_time_min": round(train_time / 60, 1),
                "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "trainable_params_m": round(trainable / 1e6, 2),
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "hardware": torch.cuda.get_device_name(0),
                "rocm": True,
                "gpu_mem_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            }, f, indent=2)

        print(f"✓ Model at https://huggingface.co/{hub_model_id}")
    else:
        print(f"\n✓ Done! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
