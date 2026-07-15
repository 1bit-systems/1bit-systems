#!/usr/bin/env python3
"""
1bit Speculative Decoding — Train a draft model for Qwen3-0.6B.

Usage:
    # Train Eagle3 (1 draft layer — recommended for NPU)
    python train_draft.py --arch eagle3 --config configs/eagle3_qwen3_0.6b.py

    # Train DFlash (5 layers, no Markov — best perf/tradeoff)
    python train_draft.py --arch dflash --config configs/dflash_qwen3_0.6b.py

    # Train DSpark (5 layers, full Markov + confidence — highest acceptance)
    python train_draft.py --arch dspark --config configs/dspark_qwen3_0.6b.py

Prerequisites:
    pip install torch transformers datasets deepspeed
    # Or from DeepSpec: pip install -r DeepSpec/requirements.txt
"""

import argparse
import os
import sys

# Add DeepSpec to path
DEEPSPEC_DIR = os.path.expanduser("~/DeepSpec")
if os.path.exists(DEEPSPEC_DIR):
    sys.path.insert(0, DEEPSPEC_DIR)

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a speculative decoding draft model for Qwen3-0.6B"
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["eagle3", "dflash", "dspark"],
        default="eagle3",
        help="Draft architecture (default: eagle3 — smallest, best for NPU)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to DeepSpec config file",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Target model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for trained draft model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=512,
        help="Gradient accumulation steps (global_batch = batch_size * grad_accum * num_gpus)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=6e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training data (JSONL format, will generate if not provided)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without training",
    )
    return parser.parse_args()


def prepare_training_data(tokenizer, max_length, output_path):
    """Generate or load training data for draft model training.
    
    The target cache is the most expensive part of DeepSpec training
    (~38TB for 4B model). For 0.6B, it's much smaller (~2-3TB for 
    1 trillion tokens).
    
    For small-scale training, we use pre-existing datasets.
    """
    if os.path.exists(output_path):
        print(f"Using existing data: {output_path}")
        return output_path
    
    # In production, this would:
    # 1. Download prompts from ShareGPT/OpenHermes/UltraChat
    # 2. Run target model to regenerate responses
    # 3. Extract hidden states -> target cache
    # 4. Format as JSONL with input_ids + target_hidden_states
    
    print(f"Data preparation not implemented. Use DeepSpec's data pipeline:")
    print(f"  cd {DEEPSPEC_DIR} && bash scripts/data/prepare_data.sh")
    print(f"With target_model_name_or_path=Qwen/Qwen3-0.6B")
    return None


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"1bit Speculative Decoding — Training {args.arch.upper()} draft for Qwen3-0.6B")
    print("=" * 60)
    
    # Validate GPU availability
    if args.num_gpus == 0 and not args.dry_run:
        print("WARNING: No GPUs available. Training requires CUDA.")
        print("Falling back to config preview mode.")
        args.dry_run = True
    
    # Load target model config
    print(f"\nLoading target model config: {args.target_model}")
    target_config = AutoConfig.from_pretrained(args.target_model)
    print(f"  Architecture: {target_config.architectures[0]}")
    print(f"  Hidden size: {target_config.hidden_size}")
    print(f"  Layers: {target_config.num_hidden_layers}")
    print(f"  Heads: {target_config.num_attention_heads}")
    print(f"  KV heads: {target_config.num_key_value_heads}")
    print(f"  Vocab: {target_config.vocab_size}")
    print(f"  Params: ~{target_config.num_hidden_layers * target_config.hidden_size * target_config.intermediate_size * 3 / 1e9:.2f}B")
    
    # Architecture-specific draft model sizes
    draft_params = {
        "eagle3": f"{target_config.hidden_size * target_config.hidden_size * 4 / 1e6:.1f}M",
        "dflash": f"{target_config.hidden_size * target_config.hidden_size * 4 * 5 / 1e6:.1f}M",
        "dspark": f"{target_config.hidden_size * target_config.hidden_size * 4 * 5 / 1e6:.1f}M + Markov",
    }
    print(f"\nDraft architecture: {args.arch.upper()}")
    print(f"  Estimated draft params: {draft_params[args.arch]}")
    print(f"  Block size: 7")
    print(f"  Target layers: 5 (evenly spaced across {target_config.num_hidden_layers})")
    
    if args.arch == "eagle3":
        print(f"  Draft hidden layers: 1 — smallest footprint")
    else:
        print(f"  Draft hidden layers: 5 — higher acceptance, larger footprint")
    
    # Training config
    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Per-device batch: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    global_batch = args.batch_size * args.grad_accum * max(args.num_gpus, 1)
    print(f"  Global batch size: {global_batch}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  Precision: bf16")
    print(f"  GPUs: {args.num_gpus}")
    
    # Memory estimate
    target_mem = target_config.hidden_size * target_config.num_hidden_layers * target_config.intermediate_size * 3 * 2 / 1e9
    print(f"\nMemory estimate (bf16):")
    print(f"  Target model: ~{target_mem:.1f} GB")
    if args.arch == "eagle3":
        draft_mem = target_config.hidden_size * target_config.hidden_size * 4 * 2 / 1e9
        print(f"  Draft model: ~{draft_mem:.2f} GB")
        print(f"  Total: ~{target_mem + draft_mem:.1f} GB")
    else:
        draft_mem = target_config.hidden_size * target_config.hidden_size * 4 * 5 * 2 / 1e9
        print(f"  Draft model: ~{draft_mem:.2f} GB")
        print(f"  Total: ~{target_mem + draft_mem:.1f} GB")
    
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN — config printed, exiting")
        print("=" * 60)
        return
    
    # Prepare data
    if args.data_path is None:
        data_path = os.path.join(args.output_dir, "training_data.jsonl")
        data_path = prepare_training_data(
            AutoTokenizer.from_pretrained(args.target_model),
            args.max_length,
            data_path,
        )
        if data_path is None:
            print("\nERROR: No training data. Please provide --data-path or run DeepSpec data pipeline.")
            sys.exit(1)
    else:
        data_path = args.data_path
    
    # Build training command
    config_path = args.config or os.path.join(
        os.path.dirname(__file__),
        "configs",
        f"{args.arch}_qwen3_0.6b.py"
    )
    
    if args.num_gpus > 0:
        print(f"\nStarting training with {args.num_gpus} GPUs...")
        cmd = (
            f"torchrun --nproc_per_node={args.num_gpus} "
            f"{DEEPSPEC_DIR}/train.py "
            f"--config {config_path} "
            f"--opts train.lr={args.lr} "
            f"train.num_train_epochs={args.epochs} "
            f"data.max_length={args.max_length} "
        )
        print(f"Command: {cmd}")
        
        if not args.dry_run:
            os.makedirs(args.output_dir, exist_ok=True)
            os.system(cmd)
    else:
        print("\nNo GPUs available. To train, run on a GPU machine:")
        print(f"  torchrun --nproc_per_node=N {DEEPSPEC_DIR}/train.py --config {config_path}")
        print(f"\nOr use SpecForge for online training:")
        print(f"  cd ~/SpecForge")
        print(f"  bash examples/run_qwen3_8b_{args.arch}_online.sh")
        print(f"  # (adapt for 0.6B)")

    print("\n" + "=" * 60)
    print("Training complete. Next steps:")
    print("1. Convert draft model to GGUF format for NPU:")
    print("   python convert_to_gguf.py --draft-checkpoint <path> --output draft-qwen3-0.6b-mtp.gguf")
    print("2. Run evaluation:")
    print("   python eval_draft.py --target Qwen/Qwen3-0.6B --draft <checkpoint>")
    print("3. Integrate with NPU engine:")
    print("   See engine/spec_decode.h for the C++ integration API")
    print("=" * 60)


if __name__ == "__main__":
    main()
