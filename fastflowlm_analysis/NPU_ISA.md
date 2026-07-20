# XDNA2 NPU Instruction Set Architecture (Reverse-Engineered)

Decoded from `libgemm.so` — the NPU instruction sequence generator kernel.

## Command Types (Opcode Headers)

| Opcode | Name | Cmd Class | Size (words) | Description |
|--------|------|-----------|-------------|-------------|
| `0x00` | XAIE_IO_WRITE | `npu_write_cmd` | 6 | Write a register value to a tile |
| `0x01` | XAIE_IO_BLOCKWRITE | `npu_dma_block_cmd` | 10 | N-dimensional DMA block transfer |
| `0x03` | XAIE_IO_MASKWRITE | `npu_issue_token_cmd` | 7 | Issue sync token / mask write |
| `0x81` | XAIE_IO_CUSTOM_OP_DDR_PATCH | `npu_ddr_cmd` | 10 | DDR address patching for BD |
| `0x04` | XAIE_IO_CUSTOM_OP_TCT | `npu_wait_cmd` | — | Wait for DMA completion (sync) |
| `0x18` | — | trailer | 1 | End-of-instruction marker |

## NPU Instruction Sequence Format

The `npu_sequence` is a `std::vector<uint32_t>` with this header:

```
Word 0: [npu_major(8) | npu_minor(8) | npu_dev_gen(8) | npu_rows(8)]
Word 1: [npu_cols(8) | npu_mem_tile_rows(8) | reserved(16)]
Word 2: instruction_count (number of command objects)
Word 3: instruction_lines (total uint32_t count, always << 2)
```

For NPU2 (Strix Halo / Ryzen AI MAX+):
```
Word 0: 0x00000406  (major=0, minor=1, gen=4, rows=6)
Word 1: 0x00000108  (cols=8, mem_tile_rows=1)
```

## NPU Tile Layout (6×8)

```
Col:   0   1   2   3   4   5   6   7
Row 0: IT0 IT1 IT2 IT3 IT4 IT5 IT6 IT7   (IT = Input Tile / Shim)
Row 1: MT0 MT1 MT2 MT3 MT4 MT5 MT6 MT7   (MT = Memory Tile)
Row 2: CT00 CT01 CT02 CT03 CT04 CT05 CT06 CT07  (CT = Compute Tile row 0)
Row 3: CT10 CT11 CT12 CT13 CT14 CT15 CT16 CT17  (CT = Compute Tile row 1)
Row 4: CT20 CT21 CT22 CT23 CT24 CT25 CT26 CT27  (CT = Compute Tile row 2)
Row 5: CT30 CT31 CT32 CT33 CT34 CT35 CT36 CT37  (CT = Compute Tile row 3)
```

Tile encoding: `npu_tiles = (row << 4) | col`

## Command Encodings

### 1. XAIE_IO_WRITE (0x00) — `npu_write_cmd`

Structure `npu_write_cmd` (48 bytes):
```
Offset  Size  Field
0x00    8     vtable ptr
0x08    4     row (tile row)
0x0c    4     col (tile col)
0x10    4     reg_addr (register address)
0x14    4     value (value to write)
0x18    4     could_be_push_queue (bool: 0 = rtp_write, 1 = queue push)
0x1c    8     — (padding)
0x24    4     bd_id (for queue writes)
0x28    4     repeat_count (for queue writes)
0x2c    1     issue_token (bool)
```

Serialized format (6 uint32_t words):
```
Word 0: 0x00000000   (opcode header = XAIE_IO_WRITE)
Word 1: 0x00000000   (padding)
Word 2: packed[ reg_addr | (value_hi << 20) | (bd_id << 25) ]
        bit[0:19]  = reg_addr from +0x10
        bit[20:24] = value_hi? from +0x20
        bit[25:31] = bd_id from +0x24
Word 3: 0x00000000   (padding)
Word 4: value         (the actual value to write, from +0x14)
Word 5: 0x00000018   (trailer = 24, op_lines count)
```

When `could_be_push_queue` is true (+0x2c), the instruction acts as a queue push instead of a direct rtp_write.

### 2. XAIE_IO_BLOCKWRITE (0x01) — `npu_dma_block_cmd`

Structure `npu_dma_block_cmd` (128+ bytes):
```
Offset  Size  Field
0x00    8     vtable ptr
0x08    4     row
0x0c    4     col
0x10    4     bd_id (Buffer Descriptor ID, 0-15)
0x14    4     buffer_offset (DDR byte offset for BD0)
0x18    4     buffer_length (DMA transfer length in bytes)
0x1c    4     dim0_size (step 3 size, wraps at 10 bits)
0x20    4     dim0_stride (step 3 stride, 20 bits signed)
0x24    4     dim1_size (step 2 size, 10 bits)
0x28    4     dim1_stride (step 2 stride, 20 bits signed)
0x2c    4     dim2_size (step 1 size, 10 bits)
0x30    4     dim2_stride (step 1 stride, 20 bits signed)
0x34    4     iter_size (number of iterations)
0x38    4     iter_stride (stride between iterations)
0x3c    4     next_bd_id (chained BD, 0 = end)
0x40    4     valid_bd (1 = valid)
0x44    4     cache_flag (0 = no_cache, 1 = normal, 2 = aggressive)
0x48    4     packet_enable (bool)
0x4c    4     packet_id
0x50    4     packet_type
0x54    4     out_of_order_id
0x58    4     issue_token (bool)
0x5c    4     enable_lock (bool)
0x60    4     get_lock_rel_val
0x64    4     get_lock_rel_id
0x68    4     get_lock_acq_enable
0x6c    4     get_lock_acq_val
0x70    4     get_lock_acq_id
0x74    4     shm_control_packet_id (default 15)
0x78    4     is_linear (bool, 1 = 1D transfer)
0x7c    4     use_next_bd (always 0)
```

Serialized format (10 uint32_t words):
```
Word 0: 0x00000001             (opcode = XAIE_IO_BLOCKWRITE)
Word 1: 0x00000000             (padding)

Word 2: packed BD config
        bit[0:4]   = col (from +0x0c, << 5)          [shift 5]
        bit[5:19]  = row (from +0x10, << 0x14)        [shift 0x14]
        bit[20:24] = bd_id (from +0x08, << 0x19)      [shift 0x19]
        = 0x1d000 | (col << 5) | (row << 0x14) | (bd_id << 0x19)

Word 3: 0x00000030             (something = 48)

Word 4: buffer_offset          (from +0x14)
Word 5: buffer_length          (from +0x18)

Word 6: packed dim0 (step 3) config
        bit[0:9]   = dim0_size - 1                    (from +0x1c)
        bit[10:17] = dim0_stride                       (from +0x20, << 0x18)
        bit[24:31] = field from +0x24 << 0x1e          [shift 0x1e]
        = (dim0_size-1) | (dim0_stride << 0x18) | (field_24 << 0x1e)

Word 7: packed dim1 (step 2) + dim2 (step 1)
        bit[0:9]   = dim1_size - 1                    (from +0x24, << 0x13) [shift 0x13]
        bit[10:19] = dim1_stride - 1                  (from +0x28, << 0x10) [shift 0x10]
        bit[20:29] = dim2_size - 1                    (from +0x2c, via shl 0x14)
        bit[30]    = 0 or something
        = ((dim1_size-1) << 0x13) | ((dim1_stride-1) << 0x10) | (dim2_size-1 << 0x14) | 0xc0000000

Word 8: packed iter
        bit[0:9]   = iter_size - 1                    (from +0x38, << 0x18) [shift 0x18]
        bit[10:17] = next_bd_id from +0x50            (with -1, << 0x18)
        = (iter_size-1) | (next_bd_id-1 << 0x18) | ?? from +0x7c << 0x18

Word 9: packed iter_stride config
        bit[0:9]   = iters_stride_lo - 1              (from +0x3c, dec)
        bit[10:19] = iters_stride_hi - 1              (from +0x40, dec)
        bit[20:31] = various from +0x54/+0x58 with << 0x14
        = (dim2_stride-1) | (cache_flag-1 << 0x14)

Word 10: packed lock config
         bit[0:4]   = get_lock_rel_val from +0x64 << 0x19
         bit[5:9]   = get_lock_rel_id from +0x6c << 0x0d
         bit[10:14] = get_lock_acq_val from +0x5c << 0x1b
         bit[15:19] = get_lock_acq_id from +0x68 << 0x12
         bit[20:24] = shm_control_packet_id from +0x70 << 0x0c
         bit[25:31] = get_lock_acq_enable from +0x74 << 0x05
         = (get_lock_rel_val << 0x19) | (get_lock_rel_id << 0x0d) |
           (get_lock_acq_val << 0x1b) | (get_lock_acq_id << 0x12) |
           (shm_control_packet_id << 0x0c) | (get_lock_acq_enable << 0x05) |
           is_linear (+0x78)

Word 11: 0x00000000             (trailer? or missing for is_linear)
```

For linear (1D) transfers, dims 1-2 are zeros and only dim0 matters.

### 3. XAIE_IO_CUSTOM_OP_DDR_PATCH (0x81) — `npu_ddr_cmd`

Structure `npu_ddr_cmd` (32 bytes):
```
Offset  Size  Field
0x00    8     vtable ptr
0x08    4     bd_id (buffer descriptor to patch)
0x0c    4     row
0x10    4     col
0x14    4     arg_idx (DDR buffer argument index)
0x18    4     arg_offset (byte offset into DDR)
0x1c    4     ? (maybe type flag)
```

Serialized format (10 uint32_t words):
```
Word 0: 0x00000081             (opcode = XAIE_IO_CUSTOM_OP_DDR_PATCH)
Word 1: bd_id << 2             (BD index × 4)
Word 2: 0x00000000             (padding)
Word 3: 0x00000000             (padding)
Word 4: 0x00000000             (padding)
Word 5: 0x1d004 | (row << 5) | (arg_idx << 0x14) | (col << 0x19)
Word 6: 0x00000000             (padding)
Word 7: arg_offset             (DDR byte offset)
Word 8: field_0x1c             (unknown type)
Word 9: 0x00000000             (trailer)
```

### 4. ISSUE TOKEN (0x03) — `npu_issue_token_cmd`

Structure `npu_issue_token_cmd` (32 bytes):
```
Offset  Size  Field
0x00    8     vtable ptr
0x08    4     row
0x0c    4     col
0x10    4     channel_direction (0 = MM2S, 1 = S2MM)
0x14    4     channel_id (it_channel: 0-1)
0x18    4     controller_packet_id (default 15)
0x1c    4     ? (maybe token value)
```

Serialized format (7 uint32_t words):
```
Word 0: 0x00000003             (opcode = XAIE_IO_MASKWRITE)
Word 1: 0x00000000             (padding)
Word 2: 0x1d200 | (col << 4) | (row << 8) | (channel_direction << 0x19) | (channel_id << 0x14)
        = (channel_direction << 4) | (row << 8) | (col << 8) << wait...
        = 0x1d200 | row | (channel_direction << 19) | (channel_id << 0x14) | (controller_packet_id << 0x19) 
        Actually:
        edx = [r12+0xc]  // col
        eax = [r12+0x8] << 4  // row << 4
        ecx = [r12+0x18] << 0x19  // controller_packet_id << 25
        result = (row << 4) + (col * 8) + 0x1d200
        then += (channel_id << 0x14) + (controller_packet_id << 0x19)
Word 3: 0x00000000             (padding)
Word 4: channel_direction << 8  (from +0x10, shifted by 8)
Word 5: field_0x1c             (token value)
Word 6: 0x0000001c             (trailer = 28)
```

### 5. WAIT (TCT) — `npu_wait_cmd`

Structure `npu_wait_cmd` (32 bytes):
```
Offset  Size  Field
0x00    8     vtable ptr
0x08    4     wait_row
0x0c    4     wait_col
0x10    4     channel_direction
0x14    4     wait_channel (0-1)
0x18    8     — (padding?)
```

## GEMM NPU Mapping (shim_tile_sequence_per_k_block)

The `Gemm::Impl::shim_tiles` array at address 0x15960 defines how GEMM operations are mapped onto the 6×8 NPU tile array.

The `generate_shimtile_sequence_per_k_block` template function generates the full NPU instruction sequence for a GEMM operation by:

1. **Dividing the K dimension** into blocks that fit in NPU local memory
2. **For each K-block**: scheduling DMA transfers to move A and B weight tiles from DDR to compute tiles, executing the matrix multiply on the AIE array, and writing results back
3. **Using the SHIM tiles** (row 0) for DDR-to-NPU data movement and **CT tiles** (rows 2-5) for computation

The GEMM on NPU uses:
- **A matrix**: distributed across CT tiles by columns
- **B matrix**: distributed across CT tiles by rows  
- **C matrix**: accumulated in CT tile local memory
- **Activation**: can be applied in-place (GeLU, SiLU, or none) via the `Activation_Type_t` parameter

## XRT Integration

Model `.so`'s use XRT to:
1. `xrt::device` — open NPU device
2. `xrt::xclbin` — load FPGA bitstream (program AIE array)
3. `xrt::kernel` — get kernel handle
4. `xrt::bo` — allocate device buffers (weights, inputs, outputs)
5. `xrt::run` — execute instruction sequence
6. `npu_sequence::dump()` — serialize commands → send to device via `xrt::bo::sync()`
