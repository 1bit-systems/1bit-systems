#!/bin/bash
set -euo pipefail
# Finetune Zamba2 with QLoRA on AMD ROCm
# Then convert to GGUF for 1bit.systems engine (77 tok/s 🚀)
#
# Usage:
#   ./finetune_zamba2.sh                   # fine-tune 1.2B with defaults
#   ./finetune_zamba2.sh 2.7b              # fine-tune 2.7B
#   MODEL=1.2b STEPS=500 ./finetune_zamba2.sh

set -euo pipefail

MODEL="${1:-1.2b}"
STEPS="${STEPS:-200}"
DATASET="${DATASET:-yahma/alpaca-cleaned}"
OUTDIR="${OUTDIR:-/tmp/zamba2-${MODEL}-finetune}"
PUSH_HUB="${PUSH_HUB:-}"

case "$MODEL" in
    1.2b)
        HF_MODEL="Zyphra/Zamba2-1.2B"
        ;;
    2.7b)
        HF_MODEL="Zyphra/Zamba2-2.7B"
        ;;
    7b)
        HF_MODEL="Zyphra/Zamba2-7B"
        ;;
    *)
        echo "Unknown model: $MODEL (choose: 1.2b, 2.7b, 7b)"
        exit 1
        ;;
esac

echo "=========================================="
echo "Zamba2-${MODEL} QLoRA Fine-Tune on AMD ROCm"
echo "Model: ${HF_MODEL}"
echo "Dataset: ${DATASET}"
echo "Steps: ${STEPS}"
echo "Output: ${OUTDIR}"
echo "=========================================="

# Activate environment
PYTHON=/tmp/torchtune-env/bin/python

# Export variables for safe Python access (no shell interpolation)
export HF_MODEL OUTDIR STEPS DATASET PUSH_HUB

$PYTHON -c "
import torch, transformers, os, json, time, math
from dataclasses import dataclass
from typing import Optional

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, HfArgumentParser, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Monkey-patch mlx issue
import transformers.utils.generic as tf_utils
tf_utils._is_mlx_available = False
tf_utils._is_mlx = lambda x: False

MODEL_ID = os.environ['HF_MODEL']
OUTPUT_DIR = os.environ['OUTDIR']
MAX_STEPS = int(os.environ['STEPS'])
DATASET = os.environ['DATASET']
PUSH_HUB = os.environ.get('PUSH_HUB', '')

# Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model
print(f'\nLoading {MODEL_ID}...')
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.config.output_router_logits = False  # safety

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
t1 = time.time()
total = sum(p.numel() for p in model.parameters())
print(f'Loaded in {t1-t0:.1f}s — {total/1e9:.2f}B total')
print(f'GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA — target attention + Mamba2 projections
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj','k_proj','v_proj','o_proj','in_proj','out_proj','gate_up_proj','down_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable params: {trainable/1e6:.2f}M ({trainable/total*100:.2f}%)')

# Dataset
dataset = load_dataset(DATASET, split='train')

def fmt(example):
    if example.get('input'):
        text = f\"Below is an instruction that describes a task, paired with an input that provides further context.\\\\n\\\\n### Instruction:\\\\n{example['instruction']}\\\\n\\\\n### Input:\\\\n{example['input']}\\\\n\\\\n### Response:\\\\n{example['output']}\"
    else:
        text = f\"Below is an instruction that describes a task.\\\\n\\\\n### Instruction:\\\\n{example['instruction']}\\\\n\\\\n### Response:\\\\n{example['output']}\"
    return {'text': text}

dataset = dataset.map(fmt, remove_columns=dataset.column_names)
dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, max_length=512, padding=False), remove_columns=['text'])
dataset = dataset.train_test_split(test_size=0.01, seed=42)
train_dataset, eval_dataset = dataset['train'], dataset['test']
print(f'Train: {len(train_dataset)}, Eval: {len(eval_dataset)}')

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Custom Trainer
class ZambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

hub_id = f'{PUSH_HUB}/Zamba2-${MODEL}-QLoRA' if PUSH_HUB else None

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_steps=MAX_STEPS,
    logging_steps=5,
    save_steps=100,
    save_total_limit=2,
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    optim='adamw_8bit',
    report_to='none',
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
    dataloader_num_workers=1,
    hub_model_id=hub_id,
    push_to_hub=bool(PUSH_HUB),
    hub_private_repo=True,
    hub_strategy='end',
    eval_strategy='steps',
    eval_steps=100,
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True,
)

trainer = ZambaTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=collator,
)

# Train
print('\\\\n' + '='*60)
print('Starting Zamba2 training...')
print(f'Steps: {MAX_STEPS}')
print('='*60)

t0 = time.time()
trainer.train()
train_time = time.time() - t0

# Save
print(f'\\\\n✓ Training completed in {train_time/60:.1f} min')
print(f'Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
trainer.save_model(OUTPUT_DIR)

metrics = {
    'model': MODEL_ID,
    'steps': MAX_STEPS,
    'train_time_min': round(train_time/60, 1),
    'peak_gpu_gb': round(torch.cuda.max_memory_allocated()/1e9, 2),
    'trainable_params_m': round(trainable/1e6, 2),
    'hardware': torch.cuda.get_device_name(0),
    'rocm': True,
}
with open(f'{OUTPUT_DIR}/training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f'\\\\n✓ Saved to {OUTPUT_DIR}')

if PUSH_HUB:
    model.push_to_hub(hub_id, private=True)
    tokenizer.push_to_hub(hub_id, private=True)
    print(f'✓ Pushed to https://huggingface.co/{hub_id}')
"
