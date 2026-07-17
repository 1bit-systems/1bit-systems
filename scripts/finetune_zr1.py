#!/usr/bin/env python3
"""Fine-tune Zyphra ZR1-1.5B on AMD Strix Halo with ROCm TheRock"""
import transformers.utils.generic as tf_utils
tf_utils._is_mlx_available = False
tf_utils._is_mlx = lambda x: False

import torch, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL = "Zyphra/ZR1-1.5B"
OUT = "/tmp/zr1-1.5b-finetune"

print("Loading Zyphra ZR1-1.5B (Qwen2 arch — full ROCm support)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16)
total = sum(p.numel() for p in model.parameters())
print(f"Loaded {total/1e9:.2f}B in {time.time()-t0:.1f}s, VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token

lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable/1e6:.2f}M ({trainable/total*100:.2f}%)")

ds = load_dataset("yahma/alpaca-cleaned", split="train")
def fmt(ex):
    i, inp, o = ex.get("instruction",""), ex.get("input",""), ex.get("output","")
    t = f"### Instruction:\n{i}\n\n### Input:\n{inp}\n\n### Response:\n{o}" if inp else f"### Instruction:\n{i}\n\n### Response:\n{o}"
    return {"text": t}
ds = ds.map(fmt, remove_columns=ds.column_names)
ds = ds.map(lambda x: tok(x["text"], truncation=True, max_length=512), remove_columns=["text"])
ds = ds.train_test_split(test_size=0.01, seed=42)
tr, ev = ds["train"], ds["test"]
print(f"Train: {len(tr)}, Eval: {len(ev)}")

args = TrainingArguments(
    output_dir=OUT, per_device_train_batch_size=2, gradient_accumulation_steps=4,
    learning_rate=2e-4, max_steps=200, logging_steps=5, save_steps=100, save_total_limit=2,
    bf16=True, optim="adamw_torch", report_to="none", remove_unused_columns=False,
    eval_strategy="steps", eval_steps=100,
)
trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=ev, processing_class=tok,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False))

print("\n=== Training ZR1-1.5B on Strix Halo ===")
t0 = time.time()
trainer.train()
train_time = time.time() - t0
trainer.save_model(OUT)
tok.save_pretrained(OUT)

m = {"model": MODEL, "steps": 200, "train_time_min": round(train_time/60, 1), "peak_gpu_gb": round(torch.cuda.max_memory_allocated()/1e9, 2), "hardware": torch.cuda.get_device_name(0)}
with open(f"{OUT}/training_metrics.json", "w") as f: json.dump(m, f, indent=2)
print(f"\nDone! {train_time/60:.1f} min - Peak: {m['peak_gpu_gb']} GB")
print(json.dumps(m))
