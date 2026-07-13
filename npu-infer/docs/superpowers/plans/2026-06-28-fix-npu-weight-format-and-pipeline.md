# Fix NPU Weight Format and Inference Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the npu-infer engine's Q4NX I4 dequantization, NPU weight blocking, and build a correct 28-layer inference pipeline that produces valid output.

**Architecture:** The engine loads Q4NX weights using proper I4 group quantization (dequantize I4→BF16), packs into NPU's [256,1024] blocked format, and submits to XRT kernels via opcode=3 on 4 xclbins (mm, attn, layer, dequant). The fix has 3 phases: (1) correct I4 weight dequantization, (2) fix NPU weight blocking for arbitrary-sized matrices, (3) build a correct inference pipeline with proper layer operations.

**Tech Stack:** C/C++17, XRT 2024.2, XDNA 2 NPU, Q4NX model format, I4 group quantization

## Global Constraints

- Model: Qwen3-0.6B with hidden=1024, 28 layers, 16 heads, 8 KV heads, intermediate=3072, vocab=151936
- NPU blocking: [256, 1024] BF16 per 1MB BO, zero-padded to 1MB
- Q4NX I4 format: 20-byte groups [scale_BF16:2B][zp_BF16:2B][16 bytes = 32 signed I4 nibbles]
- Weight shape mapping: I8[I8_rows, 5120] → each row = 256 groups × 32 I4 = 8192 BF16 values → BF16[I8_rows*8, 1024]
- BF16 tensors (embed, norms): stored directly as BF16 byte pairs with shape given in metadata
- XCLBIN files: `/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/`
- All xclbins use opcode=3 with instr=NULL, ninstr=0 (pre-compiled kernel mode)
- Current status: engine runs, produces wrong output (deterministic, tokens [919, 996, 185, 385, 495, 156, ...])

---

### Task 1: Fix I4 Dequantization in model.c

**Files:**
- Modify: `src/model.c`
- Test: `src/test_weight_packer.c`
- Verify: `tools/verify_dequant.py`

**Interfaces:**
- Consumes: `TensorDesc` with `dtype="I8"`, raw file data from Q4NX
- Produces: `npu_dequant_block()` that correctly dequantizes I4→BF16 per group

The current `npu_dequant_block()` reads I8 bytes as BF16 byte pairs — this is WRONG for dtype=I8 tensors. The correct format is INT4 group quantization:

Each 20-byte group = [scale_bf16_le:2B][zp_bf16_le:2B][16 bytes = 32 I4 nibbles]
- Nibbles are stored low-nibble-first per byte: byte = [lo_nibble | hi_nibble]
- Signed I4: 0-7 = 0-7, 8-15 = -8 to -1
- Dequant: BF16_val = I4_signed * scale_bf16 + zp_bf16

Storage shape [I8_rows, 5120] → BF16 shape [I8_rows * 8, 1024]:
- 5120 bytes/row ÷ 20 bytes/group = 256 groups/row
- 256 groups × 32 I4 = 8192 BF16 per row
- 8192 / 1024 = 8 complete BF16 rows per I8 row

- [ ] **Step 1: Write the failing dequant test**

Create `tests/test_i4_dequant.py`:

```python
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

# The captured BO is lm_head.weight row data packed to [256,1024]
# Find which tensor and block it matches
v = metadata["lm_head.weight"]
with open(model_path, "rb") as f:
    f.seek(v['data_offsets'][0])
    model_data = f.read(v['data_offsets'][1] - v['data_offsets'][0])

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
    # Storage: [i8_rows, i8_cols] = [i8_rows, 5120] bytes
    # Each row has i8_cols/20 = 256 groups
    # Each group = 32 I4 → 32 BF16
    groups_per_row = i8_cols // 20
    bf16_per_row = groups_per_row * 32  # 8192
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

# Dequantize lm_head (shape [18992, 5120])
all_bf16 = dequant_i4_to_bf16(model_data, 18992, 5120)
# Reshape to [151936, 1024]
bf16_matrix = [all_bf16[i*1024:(i+1)*1024] for i in range(151936)]

# The captured BO has 2 rows of 1024 BF16 values
# These should match the first 2 rows of some tensor
# Check the first 10 rows of dequantized data
print("First 10 dequantized BF16 values from lm_head row 0:")
print([f"{v:.4f}" for v in bf16_matrix[0][:10]])

# Compare with captured BO row 0
print("\nCaptured BO row 0 first 10:")
print([f"{v:.4f}" for v in cap_float[:10]])

# Now check: the captured BO might be a specific block of a projection weight
# Not lm_head. Let me check q_proj layer 0
v = metadata["model.layers.0.self_attn.q_proj.weight"]
print(f"\nq_proj: shape={v['shape']}")
with open(model_path, "rb") as f:
    f.seek(v['data_offsets'][0])
    qp_data = f.read(v['data_offsets'][1] - v['data_offsets'][0])
all_qp = dequant_i4_to_bf16(qp_data, 256, 5120)
# BF16 shape = [2048, 1024]
qp_matrix = [all_qp[i*1024:(i+1)*1024] for i in range(2048)]
print("q_proj row 0 first 10:")
print([f"{v:.4f}" for v in qp_matrix[0][:10]])

# Check if captured BO matches first 2 rows of q_proj
for bo_row in range(2):
    matches = sum(1 for a, b in zip(cap_float[bo_row*1024:(bo_row+1)*1024], 
                                     qp_matrix[bo_row]) if abs(a-b) < 1e-3)
    print(f"  Captured row {bo_row} vs q_proj row {bo_row}: {matches}/1024 match")

# Also check input_layernorm (BF16 tensor)
v = metadata["model.layers.0.input_layernorm.weight"]
with open(model_path, "rb") as f:
    f.seek(v['data_offsets'][0])
    norm_data = f.read(v['data_offsets'][1] - v['data_offsets'][0])
# BF16 tensor: 2 bytes per value, shape [1024]
norm_bf16 = [struct.unpack("<f", struct.pack("<I", struct.unpack("<H", norm_data[i:i+2])[0] << 16))[0] 
             for i in range(0, len(norm_data), 2)]
print(f"\ninput_layernorm BF16 first 8: {[f'{v:.4f}' for v in norm_bf16[:8]]}")

print("\n✅ Verification script ready")
```

- [ ] **Step 2: Run the verification script to see current mismatch**

Run: `python3 tests/test_i4_dequant.py`

Expected: Show that I4 dequantization produces sensible BF16 values, and that current BF16-byte-pair reading produces garbage for I8 tensors.

- [ ] **Step 3: Verify captured BO matches our dequantized weights**

Run the script and continue iterating until we find which tensor the captured BO h7 belongs to.

Expected: Captured BO h7 should match a specific row-block of a projection weight from layer 0.

- [ ] **Step 4: Fix `npu_dequant_block` in model.c**

Replace the current BF16-byte-pair reading with proper I4 group dequantization:

```c
int npu_dequant_block(void* out, const void* in,
                       const TensorDesc* desc, const ModelConfig* config,
                       int block_idx) {
    // For I4 tensors: 20-byte groups [scale:2B][zp:2B][32 I4 nibbles:16B]
    // Storage shape [I8_rows, 5120] → BF16 [I8_rows * 8, 1024]
    int i8_rows = (int)desc->shape[0];
    int i8_cols = (int)desc->shape[1];
    
    // BF16 shape dimensions
    int groups_per_row = i8_cols / 20;  // always 256 for 5120 cols
    int bf16_per_row = groups_per_row * 32;  // always 8192
    int bf16_rows = i8_rows * bf16_per_row / config->hidden_size;  // i8_rows * 8
    int bf16_cols = config->hidden_size;  // 1024
    
    int block_rows = (int)config->npu_block_rows;  // 256
    int block_cols = (int)config->npu_block_cols;  // 1024
    
    int total_row_blocks = (bf16_rows + block_rows - 1) / block_rows;
    // block_idx iterates row-blocks (since col-blocks = 1 for hidden=1024)
    
    if (block_idx >= total_row_blocks) return 0;
    
    int row_start = block_idx * block_rows;
    int row_end = row_start + block_rows;
    if (row_end > bf16_rows) row_end = bf16_rows;
    int num_rows = row_end - row_start;
    
    // Zero the output buffer (extra rows stay zero for padding)
    uint16_t* bf16_out = (uint16_t*)out;
    memset(bf16_out, 0, block_rows * block_cols * 2);
    
    const uint8_t* data = (const uint8_t*)in;
    
    // For each BF16 row, we need to find which I8 row and group offset it maps to
    // Each I8 row = 256 groups = 8192 BF16 values = 8 BF16 rows
    // So BF16 row r maps to I8 row r/8, and column c maps to BF16 col c
    for (int r = 0; r < num_rows; r++) {
        int abs_row = row_start + r;
        int i8_row = abs_row / 8;
        int bf16_offset_in_i8_row = (abs_row % 8) * 1024;
        
        // Groups cover 32 BF16 values each
        int start_group = bf16_offset_in_i8_row / 32;
        int start_offset_in_group = bf16_offset_in_i8_row % 32;
        
        for (int g = start_group; g < start_group + (1024 + 31) / 32 && g < groups_per_row; g++) {
            int group_start_byte = i8_row * i8_cols + g * 20;
            
            // Read scale and zero_point as BF16
            uint16_t scale_u16, zp_u16;
            memcpy(&scale_u16, &data[group_start_byte], 2);
            memcpy(&zp_u16, &data[group_start_byte + 2], 2);
            
            float scale = bf16_to_float(scale_u16);
            float zp = bf16_to_float(zp_u16);
            
            // Extract 32 I4 nibbles
            int group_bf16_start = g * 32;
            for (int ni = 0; ni < 32; ni++) {
                int dest_col = group_bf16_start + ni - bf16_offset_in_i8_row;
                if (dest_col < 0 || dest_col >= block_cols) continue;
                
                int byte_idx = group_start_byte + 4 + (ni / 2);
                int nibble = (ni % 2 == 0) ? (data[byte_idx] & 0x0F) : ((data[byte_idx] >> 4) & 0x0F);
                int i4_signed = (nibble < 8) ? nibble : nibble - 16;
                
                float bf16_val = i4_signed * scale + zp;
                bf16_out[r * block_cols + dest_col] = float_to_bf16(bf16_val);
            }
        }
    }
    
    return num_rows * block_cols;
}
```

Also fix `npu_weight_num_blocks` to use proper BF16 row count:

```c
int npu_weight_num_blocks(const TensorDesc* desc, const ModelConfig* config) {
    if (desc->ndim != 2) return 0;
    
    if (strcmp(desc->dtype, "BF16") == 0) {
        // BF16 tensor: shape is already [rows, cols] in BF16 values
        int64_t bf16_rows = desc->shape[0];
        int64_t bf16_cols = desc->shape[1];
        int row_blocks = (int)((bf16_rows + config->npu_block_rows - 1) / config->npu_block_rows);
        int col_blocks = (int)((bf16_cols + config->npu_block_cols - 1) / config->npu_block_cols);
        return row_blocks * col_blocks;
    } else {
        // I4 tensor: storage shape [I8_rows, 5120] → BF16 [I8_rows*8, 1024]
        int64_t i8_rows = desc->shape[0];
        int64_t i8_cols = desc->shape[1];
        int groups_per_row = (int)(i8_cols / 20);
        int bf16_per_row = groups_per_row * 32;
        int bf16_rows = (int)(i8_rows * bf16_per_row / config->hidden_size);
        int bf16_cols = config->hidden_size;
        int row_blocks = (bf16_rows + (int)config->npu_block_rows - 1) / (int)config->npu_block_rows;
        int col_blocks = (bf16_cols + (int)config->npu_block_cols - 1) / (int)config->npu_block_cols;
        return row_blocks * col_blocks;
    }
}
```

- [ ] **Step 5: Build and run test_weight_packer**

Build: `cd /home/bcloud/npu-sandbox/npu-infer/build && make -j4 test_weight_packer`

Run: `./test_weight_packer`

Verify: The test produces BF16 weights that look like actual NN weights (mean ≈ 0, std ≈ 1-2, range ≈ [-20, 20]).

- [ ] **Step 6: Verify numerical correctness against Python**

Write a C program that dequantizes q_proj block 0, saves to file, then compare with Python dequantization of the same data.

```c
// In test_weight_packer.c, add:
// 1. Dequantize q_proj layer 0, block 0
// 2. Print first 32 BF16 values
// 3. Save to /tmp/c_dequant_block0.bin for comparison
```

Run both C and Python dequant, diff the outputs. They should match within 1e-4.

- [ ] **Step 7: Commit**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
git add src/model.c src/test_weight_packer.c tests/test_i4_dequant.py
git commit -m "fix: implement proper I4 group dequantization for Q4NX weights"
```

---

### Task 2: Fix NPU Weight Blocking for Correct Tensor Shapes

**Files:**
- Modify: `src/model.c` (npu_pack_weight_bo), `include/common.h`, `include/model.h`
- Verify: `tests/test_blocking.py`

**Interfaces:**
- Consumes: dequantized I4→BF16 data in [I8_rows*8, 1024] layout
- Produces: correct [256, 1024] BF16 blocks for NPU, zero-padded to 1MB

Current bug: `npu_pack_weight_bo` calls `npu_dequant_block` which only processes `npu_block_rows` rows. But the tensor might have more rows (e.g., q_proj [2048, 1024] needs 8 row-blocks). The function needs to iterate through all BF16 rows, not just 256.

Second bug: For tensors like embed_tokens (BF16 shape [151936, 1024]), the blocking is purely row-based (151936/256 = 594 blocks). The current code needs to handle this case too.

- [ ] **Step 1: Write the blocking verification script**

```python
#!/usr/bin/env python3
"""Verify C weight packing produces correct blocks matching FLM's format"""
import json, struct, sys, os

# Read captured FLM BOs as reference
cap_dir = "/home/bcloud/npu-sandbox/xrt-direct/captured_bo_dump"
h7 = os.path.join(cap_dir, "h7_1MB_weight_prepack_layer0.bin")
with open(h7, "rb") as f:
    cap_data = f.read()
cap_bf16 = struct.unpack(f"<{len(cap_data)//2}H", cap_data)
cap_float = [struct.unpack("<f", struct.pack("<I", v << 16))[0] for v in cap_bf16]

# The captured BO is 1MB = 524,288 BF16 values = 2 × [256, 1024] blocks
# Only first 2048 (2 rows) have non-zero data
# This means it's a small tensor, not a projection weight

# Check scale BO (h9)
h9 = os.path.join(cap_dir, "h9_1MB_scale_prepack_layer0.bin")
with open(h9, "rb") as f:
    h9_data = f.read()
h9_bf16 = struct.unpack(f"<{len(h9_data)//2}H", h9_data)
h9_float = [struct.unpack("<f", struct.pack("<I", v << 16))[0] for v in h9_bf16]

print(f"h9 (scale BO): {len(h9_float)} BF16 values")
print(f"  First 16: {[f'{v:.4f}' for v in h9_float[:16]]}")
print(f"  Non-zero in first 1024: {sum(1 for v in h9_float[:1024] if abs(v)>1e-10)}")

# h7 weight BO format: [256, 1024] BF16 per block
# Each "block" is 256 rows × 1024 cols = 262,144 BF16 = 512KB
# Fills first half of 1MB BO (second half zero padding)
# But FLM only fills 2 rows = 2048 BF16 = 4096 bytes
# This suggests it's a norm-like tensor, not a projection

# Look at q_norm (shape [128] BF16)
model_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
with open(model_path, "rb") as f:
    header_size = struct.unpack("<Q", f.read(8))[0]
    json_bytes = f.read(header_size)
    metadata = json.loads(json_bytes.decode("utf-8"))

# Check all BF16 tensors for matching size
for key in sorted(metadata.keys()):
    v = metadata[key]
    if not isinstance(v, dict) or 'dtype' not in v:
        continue
    if v['dtype'] != 'BF16':
        continue
    shape = v['shape']
    n_elements = 1
    for s in shape:
        n_elements *= s
    bf16_size = n_elements * 2
    print(f"\n{key}: shape={shape}, BF16 size={bf16_size} bytes")
    
    if bf16_size <= 4096:  # 2048 BF16 values = 4096 bytes = what FLM filled
        with open(model_path, "rb") as f:
            f.seek(v['data_offsets'][0])
            data = f.read(v['data_offsets'][1] - v['data_offsets'][0])
        bf16_vals = [struct.unpack("<f", struct.pack("<I", struct.unpack("<H", data[i:i+2])[0] << 16))[0] 
                     for i in range(0, len(data), 2)]
        print(f"  First 8: {[f'{v:.4f}' for v in bf16_vals[:8]]}")
        matches = sum(1 for a, b in zip(bf16_vals[:100], cap_float[:100]) if abs(a-b) < 1e-4)
        print(f"  Matches captured BO: {matches}/100")

print("\n✅ Verification done")
```

- [ ] **Step 2: Identify what tensors the captured BOs represent**

Run the script to identify which model tensor maps to each captured BO.

- [ ] **Step 3: Fix `npu_pack_weight_bo` for full row coverage**

Update `npu_pack_weight_bo` to handle both BF16 tensors (embed, norms) and I4 tensors:

```c
int npu_pack_weight_bo(uint8_t* bo_buffer, const void* in,
                        const TensorDesc* desc, const ModelConfig* config,
                        int block_idx) {
    int bo_size = config->npu_weight_bo_size;
    memset(bo_buffer, 0, bo_size);
    
    // Determine BF16 shape
    int bf16_rows, bf16_cols;
    
    if (strcmp(desc->dtype, "BF16") == 0) {
        bf16_rows = (int)desc->shape[0];
        bf16_cols = (desc->ndim > 1) ? (int)desc->shape[1] : 1;
    } else {
        // I4: storage [I8_rows, 5120]
        int i8_rows = (int)desc->shape[0];
        int i8_cols = (int)desc->shape[1];
        int groups_per_row = i8_cols / 20;
        int bf16_per_row = groups_per_row * 32;
        bf16_rows = (int)(i8_rows * bf16_per_row / config->hidden_size);
        bf16_cols = config->hidden_size;
    }
    
    int block_rows = (int)config->npu_block_rows;
    int block_cols = (int)config->npu_block_cols;
    
    // Calculate row-blocks and column-blocks for this tensor
    int row_blocks = (bf16_rows + block_rows - 1) / block_rows;
    int col_blocks = (bf16_cols + block_cols - 1) / block_cols;
    int total_blocks = row_blocks * col_blocks;
    
    if (block_idx >= total_blocks) return -1;
    
    int row_block = block_idx / col_blocks;
    int col_block = block_idx % col_blocks;
    
    int row_start = row_block * block_rows;
    int col_start = col_block * block_cols;
    int num_rows = (row_start + block_rows <= bf16_rows) ? block_rows : (bf16_rows - row_start);
    int num_cols = (col_start + block_cols <= bf16_cols) ? block_cols : (bf16_cols - col_start);
    
    // Copy data into BO buffer (row-major, [block_rows, block_cols])
    uint16_t* bf16_out = (uint16_t*)bo_buffer;
    
    for (int r = 0; r < num_rows; r++) {
        for (int c = 0; c < num_cols; c++) {
            int abs_r = row_start + r;
            int abs_c = col_start + c;
            
            if (strcmp(desc->dtype, "BF16") == 0) {
                // Direct BF16 read
                const uint16_t* bf16_in = (const uint16_t*)((const uint8_t*)in);
                uint16_t val = bf16_in[abs_r * bf16_cols + abs_c];
                bf16_out[r * block_cols + c] = val;
            } else {
                // I4 dequantization on-the-fly
                int i8_row = abs_r / 8;
                int bf16_offset_in_i8_row = (abs_r % 8) * 1024;
                int group_idx = (bf16_offset_in_i8_row + abs_c) / 32;
                int offset_in_group = (bf16_offset_in_i8_row + abs_c) % 32;
                int i8_cols_storage = (int)desc->shape[1];
                
                int group_start = i8_row * i8_cols_storage + group_idx * 20;
                const uint8_t* data = (const uint8_t*)in;
                
                uint16_t scale_u16, zp_u16;
                memcpy(&scale_u16, &data[group_start], 2);
                memcpy(&zp_u16, &data[group_start + 2], 2);
                float scale = bf16_to_float(scale_u16);
                float zp = bf16_to_float(zp_u16);
                
                int byte_idx = group_start + 4 + (offset_in_group / 2);
                int nibble = (offset_in_group % 2 == 0) ? 
                             (data[byte_idx] & 0x0F) : ((data[byte_idx] >> 4) & 0x0F);
                int i4_signed = (nibble < 8) ? nibble : nibble - 16;
                float bf16_val = i4_signed * scale + zp;
                
                bf16_out[r * block_cols + c] = float_to_bf16(bf16_val);
            }
        }
    }
    
    return 0;
}
```

Also update `npu_dequant_block` to delegate to the new `npu_pack_weight_bo` logic (or we can deprecate `npu_dequant_block` and just keep `npu_pack_weight_bo`).

- [ ] **Step 4: Build and test**

Build: `cd /home/bcloud/npu-sandbox/npu-infer/build && cmake .. && make -j4`

Run: `./npu_infer` and check if output tokens change.

Expected: Different (hopefully more correct) token sequence than the current deterministic wrong output.

- [ ] **Step 5: Verify weight packing against Python reference**

Use batch script to compare C-packed vs Python-dequantized weights.

- [ ] **Step 6: Commit**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
git add src/model.c include/common.h include/model.h
git commit -m "fix: correct NPU weight blocking for full tensor coverage"
```

---

### Task 3: Build and Verify 1-Layer Inference on NPU

**Files:**
- Create: `tests/test_onelayer.py`
- Verify: `src/main.cpp` generates one layer

**Goal:** Before fixing the full 28-layer pipeline, verify that a single transformer layer produces correct output by comparing with FLM's output from the same input.

- [ ] **Step 1: Run FLM with a known prompt, capture the output**

Start FLM, send a request, capture the activation BO content during inference using the bo_capture library.

- [ ] **Step 2: Send the same input to our single-layer pipeline**

Use our npu_infer engine (with fixed weights) on the same input tokens, processing only layer 0.

- [ ] **Step 3: Compare the outputs**

Compare the activation BO content after layer 0 processing. They should match if our weight format and kernel invocation are correct.

- [ ] **Step 4: Document findings**

Record the mismatch (if any) and identify which part of the pipeline is wrong.

---

### Task 4: Fix the Full 28-Layer Inference Pipeline

**Files:**
- Modify: `src/main.cpp`, `src/engine.cpp`, `include/engine.h`

**Issues to fix (from handoff doc):**

1. ❌ No RMS normalization applied before attention or MLP
2. ❌ No real attention (QK^T softmax not implemented)
3. ❌ Same BO passed for both weight arguments (weight1 = weight2)
4. ❌ `run_mm_blocked()` declared but never defined - single-kernel, no batching
5. ❌ Embed lookup copies only 1024 bytes (for BF16 hidden_size) but should copy `hidden_size * 2` bytes

- [ ] **Step 1: Fix embed_lookup to handle both RMS norm and attention**

```cpp
void NpuInferenceEngine::embed_lookup(int token, NpuBo& dest) {
    TensorDesc* emb = &model_->embed_tokens;
    if (emb->ndim == 2 && token >= 0 && token < emb->shape[0]) {
        // BF16: each row = hidden_size elements = hidden_size * 2 bytes
        uint64_t offset = (uint64_t)token * emb->shape[1] * 2;
        // emb->shape[1] should be hidden_size (1024)
        size_t copy_size = emb->shape[1] * 2;
        if (copy_size > dest.size) copy_size = dest.size;
        // Copy BF16 embedding directly
        const uint8_t* src = model_->file_data + emb->data_offset + offset;
        // If embed_tokens is BF16 (it is), read directly:
        // src points to BF16 data in model file
        // dest.map is the BO buffer
        memcpy(dest.map, src, copy_size);
        dest.sync_to_device(0, copy_size);
    }
}
```

- [ ] **Step 2: Implement proper weight argument passing**

In `run_gemm()`, pass the correct weight BOs instead of the same BO twice:

```cpp
// Old: run_gemm(kern, act, ws, *w.bo, *w.bo, kv);
// New: run_gemm needs to know which two weight BOs to use
// For projections with I4 format, each BO is one row-block
// For q_proj: weight_blocks[block_idx] = BO for that block
// For k_proj: same structure
```

- [ ] **Step 3: Add RMS normalization CPU-side**

Since the xclbins don't do proper GEMM, we need to do RMS norm in software as a fallback:

```cpp
static void apply_rms_norm(float* hidden, const float* norm_weight, int hidden_size, float eps) {
    float ss = 0;
    for (int i = 0; i < hidden_size; i++) ss += hidden[i] * hidden[i];
    float rms = 1.0f / sqrtf(ss / hidden_size + eps);
    for (int i = 0; i < hidden_size; i++) hidden[i] *= rms * norm_weight[i];
}
```

- [ ] **Step 4: Commit**

---

### Task 5: torch2aie Custom XCLBIN Path (Future/Fallback)

If the xclbin compute kernels continue to produce identity output, pivot to generating custom xclbins.

**Files:**
- Refer to: `/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/`

**Goal:** Compile a custom GEMM xclbin where the AIE kernel reads weights from DDR BO arguments directly (not from tile SRAM).

- [ ] **Step 1: Study the torch2aie GEMM example**
- [ ] **Step 2: Create a custom xclbin with DDR weight input**
- [ ] **Step 3: Test in npu_infer engine**
- [ ] **Step 4: If working, integrate into pipeline**
