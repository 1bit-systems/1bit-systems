#!/usr/bin/env python3
"""Convert Zaya1-8B float32 .bin weights → Q4NX for NPU inference.

Reads all float32 .bin files from /home/bcloud/zaya_weights/ and packs them
into the Q4NX format: I4 group quantization (0.625 BPE) for 2D weight matrices,
BF16 for 1D tensors. Fully vectorized numpy — no Python loops per element.

Usage:
    python3 convert_float32_bins_to_q4nx.py [output.q4nx]
"""
import sys, os, json, struct, math, time
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
WEIGHTS_DIR = "/home/bcloud/zaya_weights"
OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "/home/bcloud/models/zaya1-8b.q4nx"

# ── Zaya Architecture ──────────────────────────────────────────────────
HIDDEN = 2048
N_LAYERS = 40
N_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 128
VOCAB = 262272
N_EXP = 16
N_FF = 2048
RTR_H = 256

Q_DIM = N_HEADS * HEAD_DIM       # 1024
K_DIM = N_KV_HEADS * HEAD_DIM    # 256
V_CUR_DIM = 128
V_DEL_DIM = 128
QKV_DIM = Q_DIM + K_DIM          # 1280
D_CONV = 2

# ── Q4NX constants ────────────────────────────────────────────────────
TILE_BYTES = 5120
LANE_BYTES = 2048
COLS_PER_TILE = 256
ROWS_PER_TILE = 32
GROUP_SIZE = 32


def _bf16_pack(f32_arr: np.ndarray) -> bytes:
    """Vectorized float32 → BF16 bytes."""
    ui32 = f32_arr.view(np.uint32)
    # Zero out NaN/Inf
    ui32 = np.where((ui32 & 0x7F800000) == 0x7F800000, 0, ui32)
    bf16 = (ui32 >> 16).astype(np.uint16)
    return bf16.tobytes()


def pack_q4nx_tile_vectorized(w_tile: np.ndarray) -> bytes:
    """Pack a 32×256 tile using fully vectorized numpy.

    Returns 5120 bytes: scales(512) + zps(512) + int4_data(4096).
    Uses no Python per-element loops — all numpy reshape/indexing.
    """
    assert w_tile.shape == (ROWS_PER_TILE, COLS_PER_TILE)

    # ── Quantize: per-row, per-group-of-32 ────────────────────────────
    # Reshape to (32, 8, 32) = (rows, groups, group_size)
    w_grp = w_tile.reshape(ROWS_PER_TILE, -1, GROUP_SIZE)  # (32, 8, 32)
    abs_max = np.max(np.abs(w_grp), axis=2, keepdims=True)  # (32, 8, 1)
    abs_max = np.maximum(abs_max, 1e-10)

    scales = abs_max / 7.0                          # (32, 8, 1)
    # Symmetric quantization: zp = 0 for signed
    zps = np.zeros_like(scales)

    q = np.clip(np.round(w_grp / scales), -7, 7)    # (32, 8, 32)
    uq = (q.astype(np.uint8)) & 0x0F                # unsigned nibble

    # ── Scales & zero-points as BF16 (row-major: row, then group) ────
    scales_flat = scales.reshape(ROWS_PER_TILE * 8)  # 256
    zps_flat = zps.reshape(ROWS_PER_TILE * 8)        # 256
    scale_bytes = _bf16_pack(scales_flat)
    zp_bytes = _bf16_pack(zps_flat)

    # ── INT4 data packing ────────────────────────────────────────────
    # Layout: 2 lanes × 2048 bytes
    #   lane[0]: rows 0..15, lane[1]: rows 16..31
    #   Within lane: column-major, 2 rows packed per byte (even=lo, odd=hi)
    #
    # Strategy: reshape to (2 lanes, 16 rows, 256 cols) → for each lane,
    # interleave even/odd row nibbles into packed bytes.

    uq_2d = uq.reshape(2, 16, COLS_PER_TILE)  # (2, 16, 256)
    # uq_2d[0] = rows 0..15, uq_2d[1] = rows 16..31

    # For each lane: (16, 256) → pack 2 rows per byte
    # Split into even rows (0,2,4,...) and odd rows (1,3,5,...)
    evens = uq_2d[:, 0::2, :]  # (2, 8, 256)
    odds  = uq_2d[:, 1::2, :]  # (2, 8, 256)

    # Pack: byte = even_nibble | (odd_nibble << 4)
    packed = evens | (odds.astype(np.uint16) << 4).astype(np.uint8)  # (2, 8, 256)

    # Transpose to (2, 256, 8) = (lane, col, byte_idx)
    packed_data = packed.transpose(0, 2, 1)  # (2, 256, 8)

    return scale_bytes + zp_bytes + packed_data.tobytes()


def pack_2d_weight(w: np.ndarray) -> bytes:
    """Pack a 2D float32 weight matrix into Q4NX tile format.

    w: [out_features, in_features] float32
    Tiles are row-major: tile_row * n_tile_cols + tile_col
    """
    out_f, in_f = w.shape

    n_tile_cols = (in_f + COLS_PER_TILE - 1) // COLS_PER_TILE
    n_tile_rows = (out_f + ROWS_PER_TILE - 1) // ROWS_PER_TILE

    # Pad to tile boundary
    pad_out = n_tile_rows * ROWS_PER_TILE - out_f
    pad_in = n_tile_cols * COLS_PER_TILE - in_f
    if pad_out > 0 or pad_in > 0:
        w = np.pad(w, ((0, pad_out), (0, pad_in)), mode='constant')

    # Tile and pack in one go: reshape to (n_tile_rows, ROWS_PER_TILE, n_tile_cols, COLS_PER_TILE)
    # then transpose to (n_tile_rows, n_tile_cols, ROWS_PER_TILE, COLS_PER_TILE)
    tiled = w.reshape(n_tile_rows, ROWS_PER_TILE, n_tile_cols, COLS_PER_TILE)
    tiled = tiled.transpose(0, 2, 1, 3)  # (tile_row, tile_col, 32, 256)

    result = bytearray()
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            tile = tiled[tr, tc]  # (32, 256)
            result.extend(pack_q4nx_tile_vectorized(tile))

    return bytes(result)


# ── Weight definitions ────────────────────────────────────────────────
def build_weight_defs():
    """Build list of (q4nx_name, bin_filename, out_dim, in_dim, do_quantize)."""
    defs = []

    def add(name, bin_name, out_dim, in_dim=0, quantize=False):
        defs.append((name, bin_name, out_dim, in_dim, quantize))

    # Model-level
    add("model.embed_tokens.weight",  "model_embed_tokens_weight.bin", VOCAB, HIDDEN, True)
    add("model.norm.weight",          "model_norm_weight.bin", HIDDEN)
    add("model.input_hidden_states_scale", "model_input_hidden_states_scale.bin", HIDDEN)
    add("model.input_hidden_states_bias",  "model_input_hidden_states_bias.bin", HIDDEN)

    for layer in range(N_LAYERS):
        p = f"model.layers.{layer}"

        # Layer norms
        add(f"{p}.input_layernorm.weight",
            f"model_layers_{layer}_input_layernorm_weight.bin", HIDDEN)
        add(f"{p}.post_attention_layernorm.weight",
            f"model_layers_{layer}_post_attention_layernorm_weight.bin", HIDDEN)

        # Attention projections (2D → Q4NX)
        add(f"{p}.self_attn.q_proj.weight",
            f"model_layers_{layer}_self_attn_qkv_proj_q_proj_weight.bin", Q_DIM, HIDDEN, True)
        add(f"{p}.self_attn.k_proj.weight",
            f"model_layers_{layer}_self_attn_qkv_proj_k_proj_weight.bin", K_DIM, HIDDEN, True)
        add(f"{p}.self_attn.v_proj_current.weight",
            f"model_layers_{layer}_self_attn_qkv_proj_v_proj_current_weight.bin", V_CUR_DIM, HIDDEN, True)
        add(f"{p}.self_attn.v_proj_delayed.weight",
            f"model_layers_{layer}_self_attn_qkv_proj_v_proj_delayed_weight.bin", V_DEL_DIM, HIDDEN, True)
        add(f"{p}.self_attn.o_proj.weight",
            f"model_layers_{layer}_self_attn_o_proj_weight.bin", HIDDEN, Q_DIM, True)

        # QK conv weights (small 2D → BF16 since dims < tile size)
        add(f"{p}.self_attn.conv_qk_depthwise.weight",
            f"model_layers_{layer}_self_attn_qkv_proj_conv_qk_depthwise_weight.bin", QKV_DIM, D_CONV)
        add(f"{p}.self_attn.conv_qk_depthwise.bias",
            f"model_layers_{layer}_self_attn_qkv_proj_conv_qk_depthwise_bias.bin", QKV_DIM)
        add(f"{p}.self_attn.conv_qk_grouped.weight",
            f"model_layers_{layer}_self_attn_qkv_proj_conv_qk_grouped_weight.bin",
            QKV_DIM, QKV_DIM // (N_HEADS + N_KV_HEADS) * D_CONV)
        add(f"{p}.self_attn.conv_qk_grouped.bias",
            f"model_layers_{layer}_self_attn_qkv_proj_conv_qk_grouped_bias.bin", QKV_DIM)

        # QK norm temperature
        add(f"{p}.self_attn.qk_norm.temp",
            f"model_layers_{layer}_self_attn_qk_norm_temp.bin", 2)

        # MoE experts (2D → Q4NX)
        add(f"{p}.mlp.experts.gate_up_proj.weight",
            f"model_layers_{layer}_mlp_experts_gate_up_proj.bin", N_EXP * N_FF * 2, HIDDEN, True)
        add(f"{p}.mlp.experts.down_proj.weight",
            f"model_layers_{layer}_mlp_experts_down_proj.bin", N_EXP * HIDDEN, N_FF, True)

        # Router MLP
        add(f"{p}.mlp.gate.router_mlp.fc1.weight",
            f"model_layers_{layer}_mlp_gate_router_mlp_fc1_weight.bin", RTR_H, HIDDEN)
        add(f"{p}.mlp.gate.router_mlp.fc1.bias",
            f"model_layers_{layer}_mlp_gate_router_mlp_fc1_bias.bin", RTR_H)
        add(f"{p}.mlp.gate.router_mlp.fc2.weight",
            f"model_layers_{layer}_mlp_gate_router_mlp_fc2_weight.bin", RTR_H, RTR_H)
        add(f"{p}.mlp.gate.router_mlp.fc2.bias",
            f"model_layers_{layer}_mlp_gate_router_mlp_fc2_bias.bin", RTR_H)
        add(f"{p}.mlp.gate.router_mlp.norm.weight",
            f"model_layers_{layer}_mlp_gate_router_mlp_norm_weight.bin", RTR_H)
        add(f"{p}.mlp.gate.router_mlp.out_proj.weight",
            f"model_layers_{layer}_mlp_gate_router_mlp_out_proj_weight.bin", N_EXP + 1, RTR_H)

        # Gate down projection
        add(f"{p}.mlp.gate.down_proj.weight",
            f"model_layers_{layer}_mlp_gate_down_proj_weight.bin", HIDDEN, RTR_H)
        add(f"{p}.mlp.gate.down_proj.bias",
            f"model_layers_{layer}_mlp_gate_down_proj_bias.bin", RTR_H)

        # Gate balancing
        add(f"{p}.mlp.gate.balancing_biases",
            f"model_layers_{layer}_mlp_gate_balancing_biases.bin", N_EXP + 1)
        add(f"{p}.mlp.gate.router_states_scale",
            f"model_layers_{layer}_mlp_gate_router_states_scale.bin", 1)

        # Residual scales
        for scope in ["post_attention", "post_mlp"]:
            for kind in ["residual_scale", "residual_bias",
                         "hidden_states_scale", "hidden_states_bias"]:
                add(f"{p}.{scope}_residual_scale.{kind}",
                    f"model_layers_{layer}_{scope}_residual_scale_{kind}.bin", HIDDEN)

    return defs


def main():
    t0 = time.time()
    print(f"=== Zaya1-8B → Q4NX Converter ===\n")
    print(f"Input:  {WEIGHTS_DIR}/")
    print(f"Output: {OUTPUT}\n")

    defs = build_weight_defs()
    print(f"Total tensors: {len(defs)}")

    # Phase 1: Pack all tensors
    print(f"\nPacking tensors...")
    packed_data = {}
    offsets = {}
    current_offset = 0
    last_report = 0

    for idx, (q4nx_name, bin_name, out_dim, in_dim, quantize) in enumerate(defs):
        full_path = os.path.join(WEIGHTS_DIR, bin_name)
        if not os.path.isfile(full_path):
            print(f"  WARNING: missing {bin_name}, skipping")
            continue

        arr = np.fromfile(full_path, dtype=np.float32)

        if quantize and in_dim > 0:
            w = arr.reshape(out_dim, in_dim).astype(np.float32)
            packed = pack_2d_weight(w)
        else:
            packed = _bf16_pack(arr)

        packed_data[q4nx_name] = packed
        offsets[q4nx_name] = (current_offset, current_offset + len(packed))
        current_offset += len(packed)

        # Progress every 30s
        elapsed = time.time() - t0
        if elapsed - last_report > 30 or idx == len(defs) - 1:
            pct = (idx + 1) * 100 // len(defs)
            rate = current_offset / elapsed / 1e6 if elapsed > 0 else 0
            print(f"  [{idx+1:4d}/{len(defs)}] {pct:3d}%  {current_offset/1e9:.2f} GB packed  "
                  f"({rate:.1f} MB/s)  elapsed={elapsed:.0f}s")
            last_report = elapsed

    # Add lm_head as alias (weight-tying)
    if "model.embed_tokens.weight" in offsets:
        packed_data["lm_head.weight"] = packed_data["model.embed_tokens.weight"]
        offsets["lm_head.weight"] = offsets["model.embed_tokens.weight"]

    total_data = current_offset
    t_pack = time.time()
    print(f"\nPacked {total_data:,} bytes ({total_data/1e9:.2f} GB) in {t_pack-t0:.0f}s")

    # Phase 2: Build JSON header
    print(f"Building JSON header...")
    metadata = {
        "hidden_size": HIDDEN,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "num_experts": N_EXP,
        "num_experts_per_tok": 2,
        "intermediate_size": N_FF,
        "max_position_embeddings": 256,
        "model_type": "zaya",
        "weight_quantization": "q4nx_int4",
    }

    for q4nx_name in defs:
        name, bin_name, out_dim, in_dim, quantize = q4nx_name
        if name not in packed_data:
            continue
        if quantize and in_dim > 0:
            n_tc = (in_dim + COLS_PER_TILE - 1) // COLS_PER_TILE
            n_tr = (out_dim + ROWS_PER_TILE - 1) // ROWS_PER_TILE
            metadata[name] = {
                "shape": [n_tr * n_tc, TILE_BYTES],
                "dtype": "I8",
                "data_offsets": list(offsets[name]),
            }
        else:
            shape = [out_dim] if in_dim == 0 else [out_dim, in_dim]
            metadata[name] = {
                "shape": shape,
                "dtype": "BF16",
                "data_offsets": list(offsets[name]),
            }

    # lm_head
    if "lm_head.weight" in offsets:
        metadata["lm_head.weight"] = metadata["model.embed_tokens.weight"].copy()

    meta_json = json.dumps(metadata, indent=2).encode("utf-8")
    header_size = len(meta_json)
    print(f"  Header: {header_size:,} bytes")

    # Phase 3: Write
    print(f"\nWriting {OUTPUT}...")
    with open(OUTPUT, "wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(meta_json)
        for q4nx_name in defs:
            name = q4nx_name[0]
            if name in packed_data:
                f.write(packed_data[name])

    file_size = os.path.getsize(OUTPUT)
    t_done = time.time()
    print(f"  File size: {file_size:,} bytes ({file_size/1e9:.2f} GB)")
    print(f"  Total time: {t_done - t0:.0f}s")

    # Stats
    i8_count = sum(1 for d in defs if d[4] and d[0] in packed_data)
    bf16_count = sum(1 for d in defs if not d[4] and d[0] in packed_data)
    print(f"\n=== Summary ===")
    print(f"  Tensors packed: {i8_count + bf16_count} ({i8_count} I8 + {bf16_count} BF16)")
    print(f"  Header overhead: {(header_size + 8)/file_size*100:.2f}%")
    print(f"  Compression ratio: raw={total_data/0.625/1e9:.1f} GB → {file_size/1e9:.2f} GB "
          f"(0.625 BPE)")
    print(f"\n✅ Done: {OUTPUT}")
    print(f"\nTo run on NPU:")
    print(f"  cd engine/npu && ./build/npu_engine_zaya {OUTPUT} 20")


if __name__ == "__main__":
    main()
