#!/usr/bin/env python3
"""Convert Zaya1-8B GGUF → Q4NX for NPU inference.
Uses the existing Q4NX converter infrastructure.

Usage: python3 convert_zaya_to_q4nx.py <input.gguf> <output.q4nx>
"""
import sys, os, json, struct, math
import numpy as np

# ── Zaya Architecture ──
HIDDEN = 2048
N_LAYERS = 40
N_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 128
VOCAB = 262272
N_EXP = 16
N_FF = 2048

# ── Q4NX Packing ──
def pack_q4nx(w: np.ndarray) -> bytes:
    """Pack fp32 weight matrix to Q4NX format using vectorized numpy.
    w: [rows, cols] float32
    Returns: bytes in Q4NX I4 group format
    """
    rows, cols = w.shape
    
    # Pad to 8-column alignment (groups of 8)
    cpad8 = (8 - cols % 8) % 8
    if cpad8:
        w = np.pad(w, ((0, 0), (0, cpad8)), mode='constant')
        cols = w.shape[1]
    
    # Reshape to groups of 8: [rows, cols/8, 8]
    w_grp = w.reshape(rows, -1, 8)  # [rows, n_groups_per_row, 8]
    n_groups = w_grp.shape[0] * w_grp.shape[1]
    
    # Vectorized I4 quantization
    abs_max = np.max(np.abs(w_grp), axis=2, keepdims=True)  # [rows, groups, 1]
    abs_max = np.maximum(abs_max, 1e-10)
    
    scales = abs_max / 7.0  # [rows, groups, 1]
    zps = np.zeros_like(scales)
    quantized = np.clip(np.round(w_grp / scales + zps), -7, 7).astype(np.int8)  # [rows, groups, 8]
    
    # Clamp to unsigned 0-15 for nibble packing
    q_clamped = np.clip(quantized, 0, 15).astype(np.uint8)  # [rows, groups, 8]
    
    # Pack 2 I4 values per byte
    q_even = q_clamped[:, :, 0::2]  # [rows, groups, 4]
    q_odd  = (q_clamped[:, :, 1::2].astype(np.uint16) << 4).astype(np.uint8)  # [rows, groups, 4]
    packed = q_even | q_odd  # [rows, groups, 4]
    
    # Flatten to bytes: [n_groups, 4]
    packed_flat = packed.reshape(-1, 4)  # [n_groups, 4]
    
    # Build output: [scale_bf16(2B)][zp_bf16(2B)][4 bytes I4] per group
    out = bytearray()
    scales_np = scales.reshape(-1)  # [n_groups]
    zps_np = zps.reshape(-1)
    packed_flat_np = packed_flat
    
    for g in range(n_groups):
        out.extend(struct.pack('e', np.float16(scales_np[g])))
        out.extend(struct.pack('e', np.float16(zps_np[g])))
        out.extend(packed_flat_np[g].tobytes())
    
    return bytes(out)

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.gguf> <output.q4nx>")
        sys.exit(1)
    
    gguf_path = sys.argv[1]
    q4nx_path = sys.argv[2]
    
    print(f"Loading GGUF: {gguf_path}")
    
    # Use gguf library if available, otherwise manual parse
    try:
        from gguf import GGUFReader
        reader = GGUFReader(gguf_path)
        tensors = {t.name: t for t in reader.tensors}
        print(f"Loaded {len(tensors)} tensors from GGUF")
    except ImportError:
        print("gguf library not available")
        sys.exit(1)
    
    # Build metadata
    meta = {
        "hidden_size": HIDDEN,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "num_kv_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "num_experts": N_EXP,
        "num_experts_per_tok": 2,
        "intermediate_size": N_FF,
        "max_position_embeddings": 256,
        "model_type": "zaya",
        "weight_quantization": "q4nx_int4",
        "tensor_names": [],
    }
    
    # Convert tensors
    q4nx_data = {}
    tensor_offsets = {}
    current_offset = 0
    
    for name, tensor in tensors.items():
        data = tensor.data
        if hasattr(data, 'numpy'):
            data = data.numpy()
        
        # Determine shape
        if len(data.shape) == 1:
            # 1D tensors (biases, norms): store as BF16 directly
            if hasattr(data, 'numpy'):
                data_np = data.numpy()
            else:
                data_np = np.array(data)
            w = data_np.astype(np.float32)
            # Pack as BF16 (no quantization)
            packed = b''.join(struct.pack('e', np.float16(v)) for v in w)
        elif len(data.shape) == 2:
            w = data.astype(np.float32) if not hasattr(data, 'numpy') else data.numpy().astype(np.float32)
            packed = pack_q4nx(w)
        elif len(data.shape) == 3:
            # 3D tensors: MoE expert weights [N_EXP, M, K]
            # Reshape to 2D: [N_EXP * M, K]
            w = data.astype(np.float32) if not hasattr(data, 'numpy') else data.numpy().astype(np.float32)
            N, M, K = w.shape
            w_2d = w.reshape(N * M, K)  # [N_EXP * M, K]
            packed = pack_q4nx(w_2d)
            print(f"  {name}: 3D {w.shape} -> 2D {w_2d.shape} -> {len(packed)} bytes")
        else:
            continue
        
        q4nx_data[name] = packed
        tensor_offsets[name] = (current_offset, current_offset + len(packed))
        current_offset += len(packed)
        meta["tensor_names"].append(name)
        
        if len(q4nx_data) <= 3 or len(q4nx_data) % 100 == 0:
            print(f"  [{len(q4nx_data):4d}/1203] {name}: {w.shape} → {len(packed)} bytes")
    
    # Write Q4NX file
    print(f"\nWriting Q4NX: {q4nx_path}")
    print(f"Total data: {current_offset:,} bytes ({current_offset/1e6:.1f} MB)")
    
    with open(q4nx_path, 'wb') as f:
        # Header: 8 bytes header_size
        meta_json = json.dumps(meta, indent=2).encode('utf-8')
        header_size = len(meta_json)
        f.write(struct.pack('<Q', header_size))
        f.write(meta_json)
        
        # Actual tensor data
        data_start = f.tell()
        for name in meta["tensor_names"]:
            f.write(q4nx_data[name])
        
        # Update metadata with data offsets
        meta["data_offsets"] = tensor_offsets
        # Rewrite header with offsets
        f.seek(0)
        meta_json = json.dumps(meta, indent=2).encode('utf-8')
        f.write(struct.pack('<Q', len(meta_json)))
        f.write(meta_json)
    
    print(f"Done: {os.path.getsize(q4nx_path):,} bytes")
    print(f"\nTo run on NPU:")
    print(f"  cd engine/npu && ./build/npu_engine_zaya {q4nx_path} 20")

if __name__ == "__main__":
    main()
