#!/usr/bin/env python3
"""
Generate synthetic Zaya1-8B weights for benchmarking.

The Zaya engine requires specific weight files in /tmp/zaya_weights/ with
hardcoded dimensions (H=2048, L=40, NH=8, NKV=2, VOCAB=262272).
This script generates random FP32 weights for all required tensors.

MoE expert weights (mlp_experts_gate_up_proj.bin, mlp_experts_down_proj.bin)
are OPTIONAL — the engine treats null pointers as no-op. Skipping them
saves ~30 GB disk space but produces different inference behavior.

Usage:
    python3 scripts/generate_synthetic_weights.py [--dir /tmp/zaya_weights] [--seed 42]
    python3 scripts/generate_synthetic_weights.py --full-moe  # include 30GB MoE weights
"""

import os, sys, struct, time, argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic Zaya1-8B weights")
    parser.add_argument("--dir", default="/tmp/zaya_weights",
                        help="Output directory (default: /tmp/zaya_weights)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--full-moe", action="store_true",
                        help="Generate MoE expert weights (~30 GB)")
    return parser.parse_args()


def wbin(directory, name, arr):
    """Write numpy array as raw float32 binary."""
    path = os.path.join(directory, name)
    with open(path, 'wb') as f:
        f.write(arr.astype(np.float32).tobytes())
    return len(arr)


# Zaya1-8B architecture constants
H = 2048
N_LAYERS = 40
NQ, NKV, HD = 8, 2, 128
QD = NQ * HD       # 1024
KD = NKV * HD      # 256
QKV = QD + KD      # 1280
VOCAB = 262272
N_EXP = 16
N_EXP_T = 17
N_FF = 2048
RTR_H = 256        # router hidden


def generate(args):
    directory = args.dir
    os.makedirs(directory, exist_ok=True)
    np.random.seed(args.seed)

    def rg(n, s=0.05): return np.random.randn(n) * s
    def rs(n): return np.random.randn(n) * 0.02 + 1.0  # scale-like, near 1
    def z(n): return np.zeros(n)

    total_floats = 0
    t0 = time.time()

    print("Global weights...")
    total_floats += wbin(directory, "model_embed_tokens_weight.bin", rg(VOCAB * H, 0.02))
    total_floats += wbin(directory, "model_norm_weight.bin", rs(H))
    total_floats += wbin(directory, "model_input_hidden_states_scale.bin", rs(H))
    total_floats += wbin(directory, "model_input_hidden_states_bias.bin", rg(H, 0.01))

    for il in range(N_LAYERS):
        p = f"model_layers_{il}_"
        # Attention projections (FP16 on device)
        total_floats += wbin(directory, p + "input_layernorm_weight.bin", rs(H))
        total_floats += wbin(directory, p + "self_attn_qkv_proj_q_proj_weight.bin", rg(QD * H, 0.02))
        total_floats += wbin(directory, p + "self_attn_qkv_proj_k_proj_weight.bin", rg(KD * H, 0.02))
        total_floats += wbin(directory, p + "self_attn_qkv_proj_v_proj_current_weight.bin", rg((KD // 2) * H, 0.02))
        total_floats += wbin(directory, p + "self_attn_qkv_proj_v_proj_delayed_weight.bin", rg((KD // 2) * H, 0.02))
        total_floats += wbin(directory, p + "self_attn_o_proj_weight.bin", rg(H * QD, 0.02))

        # Conv QK (FP32)
        total_floats += wbin(directory, p + "self_attn_qkv_proj_conv_qk_depthwise_weight.bin", rg(QKV * 2, 0.01))
        total_floats += wbin(directory, p + "self_attn_qkv_proj_conv_qk_depthwise_bias.bin", z(QKV))
        total_floats += wbin(directory, p + "self_attn_qkv_proj_conv_qk_grouped_weight.bin", rg(QKV * 256, 0.01))
        total_floats += wbin(directory, p + "self_attn_qkv_proj_conv_qk_grouped_bias.bin", z(QKV))
        total_floats += wbin(directory, p + "self_attn_qk_norm_temp.bin", np.array([1.0, 1.0]))

        # Residual scales (FP32)
        total_floats += wbin(directory, p + "post_attention_residual_scale_hidden_states_scale.bin", rs(H))
        total_floats += wbin(directory, p + "post_attention_residual_scale_hidden_states_bias.bin", rg(H, 0.01))
        total_floats += wbin(directory, p + "post_attention_residual_scale_residual_scale.bin", rs(H))
        total_floats += wbin(directory, p + "post_attention_residual_scale_residual_bias.bin", rg(H, 0.01))
        total_floats += wbin(directory, p + "post_attention_layernorm_weight.bin", rs(H))

        # MoE router weights (FP32)
        total_floats += wbin(directory, p + "mlp_gate_down_proj_weight.bin", rg(RTR_H * H, 0.01))
        total_floats += wbin(directory, p + "mlp_gate_down_proj_bias.bin", z(RTR_H))
        total_floats += wbin(directory, p + "mlp_gate_router_mlp_norm_weight.bin", rs(RTR_H))
        total_floats += wbin(directory, p + "mlp_gate_router_mlp_fc1_weight.bin", rg(RTR_H * RTR_H, 0.01))
        total_floats += wbin(directory, p + "mlp_gate_router_mlp_fc1_bias.bin", z(RTR_H))
        total_floats += wbin(directory, p + "mlp_gate_router_mlp_fc2_weight.bin", rg(RTR_H * RTR_H, 0.01))
        total_floats += wbin(directory, p + "mlp_gate_router_mlp_fc2_bias.bin", z(RTR_H))
        total_floats += wbin(directory, p + "mlp_gate_router_mlp_out_proj_weight.bin", rg(N_EXP_T * RTR_H, 0.01))
        total_floats += wbin(directory, p + "mlp_gate_balancing_biases.bin", z(N_EXP_T))

        # Post-MLP residual scales
        total_floats += wbin(directory, p + "post_mlp_residual_scale_hidden_states_scale.bin", rs(H))
        total_floats += wbin(directory, p + "post_mlp_residual_scale_hidden_states_bias.bin", rg(H, 0.01))
        total_floats += wbin(directory, p + "post_mlp_residual_scale_residual_scale.bin", rs(H))
        total_floats += wbin(directory, p + "post_mlp_residual_scale_residual_bias.bin", rg(H, 0.01))

        # EDA scale
        total_floats += wbin(directory, p + "mlp_gate_router_states_scale.bin", np.array([0.1]))

        # MoE expert weights (optional — ~30 GB for all 40 layers)
        if args.full_moe:
            sz_gu = N_EXP * 2 * N_FF * H
            sz_dn = N_EXP * H * N_FF
            total_floats += wbin(directory, p + "mlp_experts_gate_up_proj.bin", rg(sz_gu, 0.01))
            total_floats += wbin(directory, p + "mlp_experts_down_proj.bin", rg(sz_dn, 0.01))

        if (il + 1) % 10 == 0:
            mb = total_floats * 4 / (1024 * 1024)
            print(f"  Layer {il+1}/{N_LAYERS} — {mb:.0f} MB")

    elapsed = time.time() - t0
    total_gb = total_floats * 4 / (1024 ** 3)
    print(f"\nDone! {total_floats / 1e6:.0f}M floats = {total_gb:.1f} GB in {elapsed:.0f}s")
    print(f"Weight files in {directory}/")
    return True


if __name__ == "__main__":
    args = parse_args()
    generate(args)
