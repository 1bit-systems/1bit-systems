#!/usr/bin/env python3
"""Verify I4 dequantization matches captured FLM weight BO"""
import json, struct, sys

model_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
cap_file = "/home/bcloud/npu-sandbox/xrt-direct/captured_bo_dump/h7_1MB_weight_prepack_layer0.bin"

# Load captured BO (pre-packed BF16 weights from FLM)
with open(cap_file, "rb") as f:
    cap_data = f.read()
cap_bf16 = struct.unpack(f"<{len(cap_data)//2}H", cap_data)
cap_float = [struct.unpack("<f", struct.pack("<I", v << 16))[0] for v in cap_bf16]

# Load model metadata
with open(model_path, "rb") as f:
    header_size = struct.unpack("<Q", f.read(8))[0]
    json_bytes = f.read(header_size)
    metadata = json.loads(json_bytes.decode("utf-8"))

def dequant_i4_group(group_bytes):
    """Dequantize one 20-byte I4 group to 32 BF16 floats"""
    scale_u16 = struct.unpack("<H", group_bytes[0:2])[0]
    zp_u16 = struct.unpack("<H", group_bytes[2:4])[0]
    scale_f = struct.unpack("<f", struct.pack("<I", scale_u16 << 16))[0]
    zp_f = struct.unpack("<f", struct.pack("<I", zp_u16 << 16))[0]
    result = []
    for i in range(16):
        byte_val = group_bytes[4 + i]
        lo_nibble = byte_val & 0x0F
        hi_nibble = (byte_val >> 4) & 0x0F
        lo_signed = lo_nibble if lo_nibble < 8 else lo_nibble - 16
        hi_signed = hi_nibble if hi_nibble < 8 else hi_nibble - 16
        result.append(lo_signed * scale_f + zp_f)
        result.append(hi_signed * scale_f + zp_f)
    return result

def dequant_i4_to_bf16(model_data, i8_rows, i8_cols):
    """Dequantize full I4 tensor to BF16 float array"""
    groups_per_row = i8_cols // 20
    bf16_per_row = groups_per_row * 32
    total_bf16 = i8_rows * bf16_per_row
    result = [0.0] * total_bf16
    for row in range(i8_rows):
        for g in range(groups_per_row):
            group_start = row * i8_cols + g * 20
            group_bytes = model_data[group_start:group_start+20]
            dq = dequant_i4_group(group_bytes)
            for i in range(32):
                result[row * bf16_per_row + g * 32 + i] = dq[i]
    return result

# Test with lm_head
v = metadata["lm_head.weight"]
with open(model_path, "rb") as f:
    f.seek(v['data_offsets'][0])
    model_data = f.read(v['data_offsets'][1] - v['data_offsets'][0])

all_bf16 = dequant_i4_to_bf16(model_data, 18992, 5120)
bf16_matrix = [all_bf16[i*1024:(i+1)*1024] for i in range(151936)]

print("=== lm_head dequantized ===")
print(f"First 10: {[f'{v:.4f}' for v in bf16_matrix[0][:10]]}")
print(f"Shape: [{len(bf16_matrix)}, {len(bf16_matrix[0])}]")

# Test with q_proj layer 0
v = metadata["model.layers.0.self_attn.q_proj.weight"]
with open(model_path, "rb") as f:
    f.seek(v['data_offsets'][0])
    qp_data = f.read(v['data_offsets'][1] - v['data_offsets'][0])

all_qp = dequant_i4_to_bf16(qp_data, 256, 5120)
qp_matrix = [all_qp[i*1024:(i+1)*1024] for i in range(2048)]

print("\n=== q_proj layer 0 dequantized ===")
print(f"First 10: {[f'{v:.4f}' for v in qp_matrix[0][:10]]}")
print(f"Range: [{min(all_qp):.4f}, {max(all_qp):.4f}]")
print(f"Shape: [{len(qp_matrix)}, {len(qp_matrix[0])}]")

# Check which BO blocks match
print("\n=== Matching against captured BO ===")
for bo_row in range(2):
    row_start = bo_row * 1024
    row_data = cap_float[row_start:row_start+1024]
    # Check q_proj
    for r in range(2048):
        matches = sum(1 for a, b in zip(row_data, qp_matrix[r]) if abs(a-b) < 1e-3)
        if matches >= 100:
            print(f"  q_proj row {r}: {matches}/1024 match with captured BO row {bo_row}")

print("\n✅ Verification complete")
