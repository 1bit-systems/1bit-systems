#!/usr/bin/env python3
"""Convert a HF safetensors BF16 checkpoint's GEMM weights into a flat
float32 blob for direct INT8 quantization, bypassing the Q4NX INT4
dequant->requant double-quantization path (issue #1074).

The Q4NX file stores GEMM weights as INT4 with per-32-element-group
BF16 scale/zero-point (see dequant_q4nx.cpp) -- packB() then re-quantizes
the dequantized float back down to INT8 with a single per-tensor/per-channel
scale for the NPU kernel. That's two lossy quantization passes stacked.
Quantizing straight from the original BF16 checkpoint removes the INT4
intermediate -- no NPU kernel/xclbin changes needed, just a different
float source feeding the same packB() path.

Layout: for layer 0..NC-1, tensors in fixed order
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
each written as raw float32, row-major [out_features, in_features], back to
back. No header -- offsets are computed analytically in C++ (see
npu_engine_universal.cpp's bf16d_tensor()/bf16d_sizes, gated on the
NPU_BF16_DIRECT_WEIGHTS env var) from the same fixed per-model shapes
already known via ModelConfig.

Usage:
  python3 convert_bf16_direct.py <model.safetensors> <output.f32bin> [num_layers]
"""
import json, struct, sys
import numpy as np

TENSOR_NAMES = [
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def bf16_bytes_to_f32(raw_bytes, shape):
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model.safetensors> <output.f32bin> [num_layers]", file=sys.stderr)
        return 1
    src, dst = sys.argv[1], sys.argv[2]

    with open(src, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len

        if len(sys.argv) >= 4:
            num_layers = int(sys.argv[3])
        else:
            layer_ids = {int(k.split(".")[2]) for k in hdr if k.startswith("model.layers.")}
            num_layers = max(layer_ids) + 1 if layer_ids else 0

        total_elems = 0
        with open(dst, "wb") as out:
            for l in range(num_layers):
                for tname in TENSOR_NAMES:
                    key = f"model.layers.{l}.{tname}"
                    info = hdr[key]
                    assert info["dtype"] == "BF16", f"{key} is {info['dtype']}, expected BF16"
                    off0, off1 = info["data_offsets"]
                    shape = info["shape"]
                    f.seek(data_start + off0)
                    raw = f.read(off1 - off0)
                    arr = bf16_bytes_to_f32(raw, shape)
                    out.write(arr.astype(np.float32).tobytes())
                    total_elems += arr.size
                if l % 7 == 0:
                    print(f"  layer {l}/{num_layers}", file=sys.stderr)

    print(f"Wrote {dst}: {total_elems} float32 elements ({total_elems * 4} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
