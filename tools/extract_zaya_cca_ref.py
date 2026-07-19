#!/usr/bin/env python3
# Regenerates the CCA attention gold reference used by tests/test_cca_attn.cpp:
#   /tmp/zaya_weights/*.bin      (layer-0 CCA attention weights, float32)
#   /tmp/cca_ref_current.npz     (hs, nw, scaled — real model activations)
#
# Loads only layer 0 of the real ZAYA1-8B checkpoint (safetensors shard 1) via
# transformers, so it doesn't need the other 39 layers' weights in memory.
#
# Run: venv-hf/bin/python tools/extract_zaya_cca_ref.py
import os
import torch
import numpy as np
from transformers import AutoConfig, AutoModel

MODEL_DIR = os.environ.get("ZAYA_MODEL_DIR", os.path.expanduser("~/models/ZAYA1-8B"))
WEIGHTS_DIR = os.environ.get("ZAYA_WEIGHTS_DIR", "/tmp/zaya_weights")
NPZ_OUT = os.environ.get("ZAYA_CCA_REF_NPZ", "/tmp/cca_ref_current.npz")
TOKEN_ID = 100  # arbitrary valid token id; only used to get a real embedding vector

os.makedirs(WEIGHTS_DIR, exist_ok=True)

config = AutoConfig.from_pretrained(MODEL_DIR)
config.num_hidden_layers = 1
config.layer_types = config.layer_types[:1]

model = AutoModel.from_pretrained(MODEL_DIR, config=config, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()

captured = {}


def pre_hook(module, args, kwargs):
    captured["hs"] = args[0].detach().clone()


def attn_hook(module, args, kwargs, output):
    captured["attn_out"] = output[0].detach().clone()


model.layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
model.layers[0].self_attn.register_forward_hook(attn_hook, with_kwargs=True)

input_ids = torch.tensor([[TOKEN_ID]], dtype=torch.long)
with torch.no_grad():
    model(input_ids=input_ids, use_cache=False)

hs = captured["hs"][0, 0].numpy().astype(np.float32)            # [H] input residual to layer 0
nw = model.layers[0].input_layernorm.weight.detach().numpy().astype(np.float32)  # [H]
scaled = captured["attn_out"][0, 0].numpy().astype(np.float32)  # [H] self_attn (o_proj) output

np.savez(NPZ_OUT, hs=hs, nw=nw, scaled=scaled)
print("Saved", NPZ_OUT, "hs", hs.shape, "nw", nw.shape, "scaled", scaled.shape)

# Dump layer-0 CCA weights, named by dotted state_dict key with dots -> underscores
# (matches the naming convention tests/test_cca_attn.cpp and kernels/zaya_cca_attn.hip expect)
sd = model.state_dict()
prefix = "layers.0.self_attn."
wanted = {
    prefix + "qkv_proj.q_proj.weight": "model_layers_0_self_attn_qkv_proj_q_proj_weight.bin",
    prefix + "qkv_proj.k_proj.weight": "model_layers_0_self_attn_qkv_proj_k_proj_weight.bin",
    prefix + "qkv_proj.v_proj_current.weight": "model_layers_0_self_attn_qkv_proj_v_proj_current_weight.bin",
    prefix + "qkv_proj.v_proj_delayed.weight": "model_layers_0_self_attn_qkv_proj_v_proj_delayed_weight.bin",
    prefix + "o_proj.weight": "model_layers_0_self_attn_o_proj_weight.bin",
    prefix + "qkv_proj.conv_qk_depthwise.weight": "model_layers_0_self_attn_qkv_proj_conv_qk_depthwise_weight.bin",
    prefix + "qkv_proj.conv_qk_depthwise.bias": "model_layers_0_self_attn_qkv_proj_conv_qk_depthwise_bias.bin",
    prefix + "qkv_proj.conv_qk_grouped.weight": "model_layers_0_self_attn_qkv_proj_conv_qk_grouped_weight.bin",
    prefix + "qkv_proj.conv_qk_grouped.bias": "model_layers_0_self_attn_qkv_proj_conv_qk_grouped_bias.bin",
    prefix + "qk_norm.temp": "model_layers_0_self_attn_qk_norm_temp.bin",
}

for key, fname in wanted.items():
    t = sd[key].detach().numpy().astype(np.float32)
    t.tofile(os.path.join(WEIGHTS_DIR, fname))
    print(f"{key}: {t.shape} -> {fname}")

print("Done.")
