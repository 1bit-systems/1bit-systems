#!/usr/bin/env python3
"""
Fine-tune Zamba2 with QLoRA (attention-only) on AMD ROCm
Then convert to GGUF for your 77 tok/s inference engine.

Usage:
    python train_zamba2.py                      # 200 steps on Alpaca
    python train_zamba2.py --max-steps 500      # more steps
    python train_zamba2.py --push-to-hub user   # upload adapter
"""

import torch, time, json, os, sys
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import argparse

# Monkey-patch mlx issue
import transformers.utils.generic as tf_utils
tf_utils._is_mlx_available = False
tf_utils._is_mlx = lambda x: False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Zyphra/Zamba2-1.2B")
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-dir", default="/tmp/zamba2-finetune")
    parser.add_argument("--push-to-hub", default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Zamba2 QLoRA Fine-Tune on ROCm")
    print(f"Model: {args.model}")
    print(f"Steps: {args.max_steps}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print("=" * 60)

    # ─── Load 4-bit ─────────────────────────────────────
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )

    print("\nLoading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    print(f"Loaded ({time.time()-t0:.1f}s) — {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # ─── Light LoRA (attention only!, 10x faster) ───────
    lora = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.2f}M ({trainable/total*100:.2f}%)")
    print(f"Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ─── Tokenizer ───────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ─── Dataset ─────────────────────────────────────────
    ds = load_dataset(args.dataset, split="train")
    def fmt(ex):
        if ex.get("input"):
            t = f"Below is an instruction.\n\n### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        else:
            t = f"Below is an instruction.\n\n### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
        return {"text": t}
    ds = ds.map(fmt, remove_columns=ds.column_names)
    ds = ds.map(lambda x: tok(x["text"], truncation=True, max_length=512, padding=False), remove_columns=["text"])
    ds = ds.train_test_split(test_size=0.01, seed=42)
    print(f"Train: {len(ds['train'])}, Eval: {len(ds['test'])}")

    # ─── Custom Trainer ──────────────────────────────────
    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    hub_id = f"{args.push_to_hub}/Zamba2-QLoRA" if args.push_to_hub else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=1,
        hub_model_id=hub_id,
        push_to_hub=bool(args.push_to_hub),
        hub_private_repo=True,
        hub_strategy="end",
        eval_strategy="steps",
        eval_steps=100,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
    )

    trainer = MyTrainer(
        model=model, args=training_args,
        train_dataset=ds["train"], eval_dataset=ds["test"],
        processing_class=tok,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )

    # ─── Train ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Training Zamba2 ({args.max_steps} steps, ~{args.max_steps*7//60} min)")
    print(f"Attention-only LoRA — Mamba2 SSM skipped (engine runs it at 77 tok/s)")
    print("=" * 60)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    trainer.save_model(args.output_dir)
    print(f"\n✓ Completed in {elapsed/60:.1f} min")
    print(f"Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"Saved to: {args.output_dir}")
    print(f"Size: {sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fn in os.walk(args.output_dir) for f in fn)/1e6:.1f} MB")

    # Save metrics
    with open(f"{args.output_dir}/training_metrics.json", "w") as f:
        json.dump({
            "model": args.model, "steps": args.max_steps,
            "time_min": round(elapsed/60, 1),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
            "trainable_m": round(trainable/1e6, 2),
            "hardware": torch.cuda.get_device_name(0),
            "rocm": True,
        }, f, indent=2)

    # Push to Hub
    if args.push_to_hub:
        model.push_to_hub(hub_id, private=True)
        tok.push_to_hub(hub_id, private=True)
        print(f"✓ Pushed to https://huggingface.co/{hub_id}")

    print("\nTo convert to GGUF for your engine:")
    print(f"  cd {args.output_dir}")
    print("  # Use llama.cpp convert-hf-to-gguf.py")
    print("  python ./convert-hf-to-gguf.py --outtype q4_0 \\")
    print(f"    --output zamba2-finetuned-q4_0.gguf .")
    print("  # Then run on your 1bit.systems engine:")
    print("  ./build/run_zamba2 zamba2-finetuned-q4_0.gguf \\")
    print('    "Your prompt here"')

if __name__ == "__main__":
    main()
