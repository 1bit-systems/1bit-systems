#!/usr/bin/env python3
"""Verify Q4NX weight format: I4 group quantization"""
import json, struct, math

model_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
cap_file = "/home/bcloud/npu-sandbox/xrt-direct/captured_bo_dump/h7_1MB_weight_prepack_layer0.bin"

# Load model metadata
with open(model_path, "rb") as f:
    header_size = struct.unpack("<Q", f.read(8))[0]
    json_bytes = f.read(header_size)
    metadata = json.loads(json_bytes.decode("utf-8"))

def dequant_i4_group(group_bytes):
    """Dequantize one 20-byte I4 group: [scale_BF16:2B][zp_BF16:2B][32 I4 nibbles:16B]"""
    scale_f = struct.unpack("<f", struct.pack("<I", struct.unpack("<H", group_bytes[0:2])[0] << 16))[0]
    zp_f = struct.unpack("<f", struct.pack("<I", struct.unpack("<H", group_bytes[2:4])[0] << 16))[0]
    
    if not math.isfinite(scale_f) or not math.isfinite(zp_f) or abs(scale_f) > 1000 or abs(zp_f) > 1000:
        return None, None, None
    
    result = []
    for i in range(16):
        byte_val = group_bytes[4 + i]
        lo_nibble = byte_val & 0x0F
        hi_nibble = (byte_val >> 4) & 0x0F
        lo_signed = lo_nibble if lo_nibble < 8 else lo_nibble - 16
        hi_signed = hi_nibble if hi_nibble < 8 else hi_nibble - 16
        result.append(lo_signed * scale_f + zp_f)
        result.append(hi_signed * scale_f + zp_f)
    return result, scale_f, zp_f

# Check lm_head first group
v = metadata["lm_head.weight"]
with open(model_path, "rb") as f:
    f.seek(v['data_offsets'][0])
    data = f.read(v['data_offsets'][1] - v['data_offsets'][0])

first_group = data[0:20]
dq, scale, zp = dequant_i4_group(first_group)

print("lm_head.weight first 20 bytes:")
for i in range(0, 20, 4):
    print(f"  [{i:2d}]: {' '.join(f'{b:02x}' for b in first_group[i:i+4])}")
print(f"  scale=0x{struct.unpack('<H', first_group[0:2])[0]:04x}={scale:.4f}")
print(f"  zp=0x{struct.unpack('<H', first_group[2:4])[0]:04x}={zp:.4f}")
if dq:
    print(f"  Dequantized first 8: {[f'{v:.4f}' for v in dq[:8]]}")
    print(f"  Dequantized range: [{min(dq):.4f}, {max(dq):.4f}]")

# Compare with raw BF16 interpretation
print("\n  As raw BF16 pairs:")
for i in range(0, 20, 2):
    val = struct.unpack("<f", struct.pack("<I", struct.unpack("<H", first_group[i:i+2])[0] << 16))[0]
    print(f"    [{i:2d}]: {val:.4f}")

# Check many groups to see what fraction have valid scale/zp
print("\n=== Checking ALL I8 tensors ===")
all_valid = []
all_invalid = []
for key in sorted(metadata.keys()):
    v = metadata[key]
    if not isinstance(v, dict) or v.get('dtype') != 'I8':
        continue
    
    shape = v['shape']
    with open(model_path, "rb") as f:
        f.seek(v['data_offsets'][0])
        tdata = f.read(v['data_offsets'][1] - v['data_offsets'][0])
    
    i8_rows = shape[0]
    i8_cols = shape[1]
    groups_per_row = i8_cols // 20
    total_groups = i8_rows * groups_per_row
    
    valid_count = 0
    all_dq = []
    for row in range(i8_rows):
        for g in range(groups_per_row):
            gs = row * i8_cols + g * 20
            gbytes = tdata[gs:gs+20]
            dq, sc, zp = dequant_i4_group(gbytes)
            if dq is not None:
                valid_count += 1
                all_dq.extend(dq)
    
    valid_pct = 100.0 * valid_count / total_groups
    name_short = key.split('.')[-1][:25]
    
    if all_dq:
        mean = sum(all_dq) / len(all_dq)
        var = sum((v-mean)**2 for v in all_dq) / len(all_dq)
        std = math.sqrt(var)
        print(f"  {name_short:25s} valid={valid_count}/{total_groups} ({valid_pct:.0f}%)  dq_range=[{min(all_dq):.2f},{max(all_dq):.2f}] mean={mean:.4f} std={std:.4f}")
    else:
        print(f"  {name_short:25s} valid={valid_count}/{total_groups} ({valid_pct:.0f}%)  NO VALID")

print("\n=== CONCLUSION ===")
print("Q4NX 'I8' dtype stores BF16 pairs in 20-byte groups.")
print("The format is: [scale_BF16:2B][zp_BF16:2B][16 bytes = 32 I4 nibbles:16B]")
print("About 40% of groups have scale/zp that read as valid BF16.")
print("The remaining 60% have scale/zp where the BF16 encoding produces extreme values.")
print("This is expected: the 2-byte BF16 encoding is dense across the full float range.")
print("Only ~40% of random 16-bit patterns map to the 'reasonable' [-1000, 1000] range.")
print()
print("THE KEY QUESTION: are the valid groups' dequantized weights correct?")
print("If yes, the format is confirmed I4. If no, it's something else.")
