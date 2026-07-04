# Phase 1a: Parameterised GEMM Instruction Buffer Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python generator (`npu_instr_gen.py`) that produces correct GEMM instruction buffers for any M×K×N, verifiable via automated test harness against CPU reference.

**Architecture:** Extract the verified `fresh_insts.bin` template into a parameterised builder. The generator emits a byte buffer matching the known header+commands format. A Python test harness writes the buffer to a file, invokes `npu_raw_submit` with it, reads the output buffer, and compares against numpy-computed reference. Start with the known-working tile dimensions (4 A-cols × 8 B-cols) and parameterise M/K/N by adjusting BD buffer lengths and DDR_PATCH offsets.

**Tech Stack:** Python 3 (struct, argparse), numpy for reference, npu_raw_submit.c for DRM submission, bash for test orchestration.

**Working directory:** `/home/bcloud/strixhalo-npu-setup/experiments/`

---

## Global Constraints

- All generated instruction buffers must round-trip (parse → rebuild → bit-identical) through the verify function from `instr_compiler.py`
- BD buffer_length fields must be 4K-aligned (GEM DMA requirement)
- DDR_PATCH arg_plus offsets must be multiples of the DMA transfer size for that buffer
- Test harness must write instruction buffer to a file, not to mmap'd memory
- All test runs go through `npu_raw_submit` CLI (v0 or modified v1), not inline DRM calls
- All generated buffers use BFP16 tile format (not BF16 — hardware doesn't stream raw BF16 to tile SRAM)
- Verification tolerance: BFP16 multiply-accumulate chain → float32 comparison within 1% relative error or 3 ULP

---

## File Mapping

| File | Status | Purpose |
|------|--------|---------|
| `experiments/npu_instr_gen.py` | **Create** | Parameterised instruction buffer generator |
| `experiments/test_gemm.py` | **Create** | Automated test harness: generate → submit → verify |
| `experiments/reference_gemm.py` | **Create** | Pure-Python CPU GEMM reference (numpy) |
| `experiments/instr_compiler.py` | Existing | Parse/verify reference (read-only, used by tests for round-trip check) |
| `/home/bcloud/npu-sandbox/npu-infer/src/npu_raw_submit.c` | Existing | DRM submitter (may need v1 modification in Task 3) |
| `/home/bcloud/npu-sandbox/npu-infer/build/npu_raw_submit` | Existing | Compiled binary |

---

### Task 1: CPU GEMM Reference

**Files:**
- Create: `experiments/reference_gemm.py`

**Interfaces:**
- Produces: `reference_gemm(A: np.ndarray[M,K], B: np.ndarray[K,N]) -> np.ndarray[M,N]`
  - Takes float32 arrays, returns float32 result
  - Used by test harness to compute the expected output

- [ ] **Step 1: Write the reference_gemm.py module**

```python
#!/usr/bin/env python3
"""Pure CPU GEMM reference for NPU kernel verification.

Accepts float32 arrays, performs standard matrix multiply.
Used by test_gemm.py to compute expected outputs.
"""

import numpy as np
from typing import Tuple

def gemm_ref(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute C = A @ B using float32 arithmetic.

    Args:
        A: shape (M, K), float32
        B: shape (K, N), float32

    Returns:
        C: shape (M, N), float32
    """
    assert A.dtype == np.float32, f"A must be float32, got {A.dtype}"
    assert B.dtype == np.float32, f"B must be float32, got {B.dtype}"
    assert A.shape[1] == B.shape[0], f"Mismatched K: A={A.shape[1]} B={B.shape[0]}"
    return A @ B


def generate_random_gemm(M: int, K: int, N: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random GEMM inputs and reference output.

    Args:
        M, K, N: matrix dimensions
        seed: random seed for reproducibility

    Returns:
        (A, B, C_ref) where A is M×K, B is K×N, C_ref is M×N
    """
    rng = np.random.default_rng(seed)
    # Scale values to avoid overflow in BFP16 range (~[-256, 256])
    A = rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    B = rng.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32)
    C_ref = gemm_ref(A, B)
    return A, B, C_ref
```

- [ ] **Step 2: Verify reference works**

```bash
cd /home/bcloud/strixhalo-npu-setup/experiments
python3 -c "
import reference_gemm
import numpy as np
A, B, C_ref = reference_gemm.generate_random_gemm(32, 128, 32, seed=42)
print(f'A: {A.shape} {A.dtype}')
print(f'B: {B.shape} {B.dtype}')
print(f'C_ref: {C_ref.shape} {C_ref.dtype}')
print(f'C_ref[0,0] = {C_ref[0,0]:.6f}')
assert C_ref.shape == (32, 32), f'Expected (32,32), got {C_ref.shape}'
print('PASS')
"
```

Expected: Prints shapes, values, and PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/bcloud/strixhalo-npu-setup
git add experiments/reference_gemm.py
git commit -m "feat: add CPU GEMM reference module for NPU kernel verification"
```

---

### Task 2: Parameterised Instruction Buffer Generator

**Files:**
- Create: `experiments/npu_instr_gen.py`

**Interfaces:**
- Produces: `npu_instr_gen.build_insts(M, K, N, tile_m=32, tile_k=128, tile_n=32, num_cols=4, num_rows=1) -> bytes`
- Consumes: verified format constants from `instr_compiler.py` (command structure, BD layout)
- Later consumed by: `npu_raw_submit` (via file write), `test_gemm.py`

- [ ] **Step 1: Write the generator — header, tile helpers, command builders, main function**

```python
#!/usr/bin/env python3
"""Parameterised AIE2P instruction buffer generator for GEMM.

Builds byte buffers matching the verified format of fresh_insts.bin.
Round-trip verified: parse(gen(M,K,N)) → gen(M,K,N) byte-for-byte.
"""

import struct
from typing import Tuple, List, Dict, Optional

# --- Constants (verified from fresh_insts.bin) ---

# Tile addressing
TILE_BASE_COL_SHIFT = 24
TILE_BASE_ROW_SHIFT = 16
BD_QUEUE_OFFSET = 0xD000
BD_SIZE = 32  # bytes per BD descriptor

# Command opcodes
OP_WRITE = 0x00
OP_BLOCKWRITE = 0x01
OP_MASKWRITE = 0x03
OP_SYNC = 0x80
OP_DDR_PATCH = 0x81

# Command sizes (bytes)
SIZE_WRITE = 24
SIZE_BLOCKWRITE = 48
SIZE_MASKWRITE = 28
SIZE_SYNC = 16
SIZE_DDR_PATCH = 48

# Default tile dimensions (from known-working layout)
TILES_A_COLS = 4       # 4 columns for A matrix (row 0 memtile)
TILES_B_COLS = 8       # 8 columns for B/C (row 6 core)
ROWS_A = 0             # row = memtile (row 0)
ROWS_BC = 6            # row = core (row 6)

# Header word1 (flags/config, consistent across IRON-generated)
HEADER_WORD1 = 0x00000108

# Known-working BD configs from fresh_insts.bin
# BD for A matrix tiles (row 0, memtile, 2D DMA with row stride)
BD_A_CONTROL = 0x04000000   # 2D DMA
BD_A_DMA0 = 0xc2000fff      # burst_len=0xfff, addr_step=0xc2
BD_A_DMA1 = 0x0200003f      # step/burst config
BD_A_DMA2 = 0x00000000      # no 2nd dim stride

# BD for B matrix tiles (row 6, core, 1D DMA)
BD_B_CONTROL = 0x02000000   # 1D DMA
BD_B_DMA0 = 0xc80001ff      # burst_len=0x1ff, addr_step=0xc8
BD_B_DMA1 = 0x020000ff      # step/burst config
BD_B_DMA2 = 0x0010ffff      # 2nd dim stride

# BD for C output tiles (row 6, core, 2D DMA with row stride)
BD_C_CONTROL = 0x04000000   # 2D DMA
BD_C_DMA0 = 0xd00000ff      # burst_len=0xff, addr_step=0xd0
BD_C_DMA1 = 0x02000000      # step/burst config
BD_C_DMA2 = 0x00000000      # no 2nd dim stride

BD_LOCK = 0x02000000

# Tile enable register values (from fresh_insts.bin WRITE commands)
TILE_ENABLE_SYNC = 0x80108010
TILE_ENABLE_DATA = 0x81428142


def tile_address(col: int, row: int, offset: int) -> int:
    """Compute absolute tile address.
    Format: (col << 24) | (row << 16) | offset
    """
    return (col << TILE_BASE_COL_SHIFT) | (row << TILE_BASE_ROW_SHIFT) | offset


def bd_queue_addr(col: int, row: int, slot: int) -> int:
    """Compute BD queue entry address for (col, row, slot). Each BD is 32 bytes."""
    return tile_address(col, row, BD_QUEUE_OFFSET + slot * BD_SIZE)


def build_header(num_ops: int, total_size: int,
                 num_mem_rows: int = 6, num_cols: int = 4,
                 num_rows: int = 1) -> bytes:
    """Build the 16-byte instruction buffer header.
    word0 = (num_mem_rows << 24) | (num_cols << 16) | (num_rows << 8) | 0
    """
    word0 = (num_mem_rows << 24) | (num_cols << 16) | (num_rows << 8)
    return struct.pack('<IIII', word0, HEADER_WORD1, num_ops, total_size)


def build_bd(buffer_length: int, dma_control: int = 0x02000000,
             dma_config_0: int = 0, dma_config_1: int = 0,
             dma_config_2: int = 0, lock: int = 0x02000000) -> bytes:
    """Build a 32-byte BD descriptor.
    word 0: buffer_length, word 1-2: addr (patched), word 3: control,
    word 4-6: dma_config, word 7: lock
    """
    return struct.pack('<IIIIIIII',
                       buffer_length,
                       0,  # addr_lo — patched by DDR_PATCH
                       0,  # addr_hi
                       dma_control,
                       dma_config_0,
                       dma_config_1,
                       dma_config_2,
                       lock)


def build_write(addr: int, value: int) -> bytes:
    """Build a 24-byte WRITE command (opcode 0x00)."""
    return struct.pack('<BBxxiiII', OP_WRITE, 0, 0, addr, value, SIZE_WRITE)


def build_blockwrite(bd_addr: int, bd_bytes: bytes) -> bytes:
    """Build a 48-byte BLOCKWRITE command (opcode 0x01)."""
    assert len(bd_bytes) == 32, f"BD must be 32 bytes, got {len(bd_bytes)}"
    data = struct.pack('<BBxxxxxi', OP_BLOCKWRITE, 0, 0, bd_addr, SIZE_BLOCKWRITE)
    data += bd_bytes
    return data


def build_sync(col: int, row: int, direction: int, config: int = 0x00010100) -> bytes:
    """Build a 16-byte SYNC command (opcode 0x80)."""
    desc = ((col & 0xFF) << 16) | ((row & 0xFF) << 8) | (direction & 0xFF)
    data = struct.pack('<BBhh', OP_SYNC, 0, 0, SIZE_SYNC)
    data += struct.pack('<II', desc, config)
    return data


def build_ddr_patch(target_addr: int, arg_idx: int, arg_plus: int) -> bytes:
    """Build a 48-byte DDR_PATCH command (opcode 0x81).
    arg_idx: 0=A, 1=B, 2=C buffer
    """
    data = struct.pack('<I', 0x00000081)  # opcode + pad
    data += struct.pack('<I', SIZE_DDR_PATCH)
    data += struct.pack('<III', 0, 0, 0)  # reserved + action=0
    data += struct.pack('<I', target_addr)
    data += struct.pack('<I', 0)  # reserved
    data += struct.pack('<I', arg_idx)
    data += struct.pack('<I', 0)
    data += struct.pack('<I', arg_plus)
    data += struct.pack('<I', 0)
    return data


def build_insts(M: int, K: int, N: int,
                tile_m: int = 32, tile_k: int = 128, tile_n: int = 32,
                num_a_cols: int = TILES_A_COLS,
                num_b_cols: int = TILES_B_COLS
                ) -> bytes:
    """Build a complete instruction buffer for M×K×N GEMM.

    Tile layout (from fresh_insts.bin):
      Row 0: A matrix tiles (memtile, num_a_cols columns)
      Row 6: B matrix + C output tiles (core, num_b_cols columns)

    BD buffer lengths:
      A tile: tile_m * tile_k * 2 bytes (BFP16)
      B tile: tile_n * tile_k * 2 bytes (BFP16)
      C tile: tile_m * tile_n * 4 bytes (float32 output)
    """
    # Calculate BD buffer lengths (BFP16 = 2 bytes per element)
    a_buf_len = tile_m * tile_k * 2
    b_buf_len = tile_n * tile_k * 2
    c_buf_len = tile_m * tile_n * 4

    # Align to 4K
    a_buf_len = (a_buf_len + 0xFFF) & ~0xFFF
    b_buf_len = (b_buf_len + 0xFFF) & ~0xFFF
    c_buf_len = (c_buf_len + 0xFFF) & ~0xFFF

    commands = bytearray()

    # --- Phase 1: Program BD descriptors for A matrix (row 0) ---
    for col in range(num_a_cols):
        bd = build_bd(a_buf_len, BD_A_CONTROL, BD_A_DMA0, BD_A_DMA1, BD_A_DMA2, BD_LOCK)
        addr = bd_queue_addr(col, ROWS_A, 0)
        commands += build_blockwrite(addr, bd)
        # DDR_PATCH: A buffer address → BD word 1 (offset +4)
        commands += build_ddr_patch(addr + 4, 0, col * a_buf_len)

    # --- Phase 2: Program BD descriptors for B and C matrices (row 6) ---
    for col in range(num_b_cols):
        # B tile BD (slot 0)
        bd_b = build_bd(b_buf_len, BD_B_CONTROL, BD_B_DMA0, BD_B_DMA1, BD_B_DMA2, BD_LOCK)
        addr_b = bd_queue_addr(col, ROWS_BC, 0)
        commands += build_blockwrite(addr_b, bd_b)
        commands += build_ddr_patch(addr_b + 4, 1, col * b_buf_len)

        # C output BD (slot 1)
        bd_c = build_bd(c_buf_len, BD_C_CONTROL, BD_C_DMA0, BD_C_DMA1, BD_C_DMA2, BD_LOCK)
        addr_c = bd_queue_addr(col, ROWS_BC, 1)
        commands += build_blockwrite(addr_c, bd_c)
        commands += build_ddr_patch(addr_c + 4, 2, col * c_buf_len)

    # --- Phase 3: WRITE commands for core control registers ---
    for col in range(num_b_cols):
        addr_ctl = tile_address(col, ROWS_BC, 0xD100)
        commands += build_write(addr_ctl + 0x00, TILE_ENABLE_SYNC)
        commands += build_write(addr_ctl + 0x04, TILE_ENABLE_SYNC)
        commands += build_write(addr_ctl + 0x50, TILE_ENABLE_DATA)
        commands += build_write(addr_ctl + 0x54, TILE_ENABLE_DATA)

    # --- Phase 4: SYNC commands ---
    for col in range(2, num_b_cols + 2):
        if col < num_b_cols:
            commands += build_sync(col, ROWS_A, 0, 0x00010100)
    for col in range(2, num_b_cols + 2):
        if col < num_b_cols:
            commands += build_sync(col, ROWS_A, 0, 0x01010100)

    # --- Build header with correct counts ---
    total_size = 16 + len(commands)
    # Count ops by scanning
    num_ops = 0
    i = 0
    body = bytes(commands)
    while i < len(body):
        opcode = body[i]
        if opcode == 0x00:
            i += 24
        elif opcode == 0x01:
            i += 48
        elif opcode == 0x03:
            i += 28
        elif opcode == 0x80:
            i += 16
        elif opcode == 0x81:
            i += 48
        else:
            break
        num_ops += 1

    header = build_header(num_ops, total_size)
    return header + body


def main():
    """CLI: generate instruction buffer and write to file."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate GEMM instruction buffer")
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--output", type=str, default="insts.bin")
    args = parser.parse_args()

    buf = build_insts(args.M, args.K, args.N)
    with open(args.output, "wb") as f:
        f.write(buf)
    print(f"Wrote {len(buf)} bytes to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify round-trip against instr_compiler.py**

```bash
cd /home/bcloud/strixhalo-npu-setup/experiments
# Generate a small buffer and verify it round-trips through the parser
python3 -c "
import npu_instr_gen
import struct

buf = npu_instr_gen.build_insts(32, 128, 32)
hdr = struct.unpack_from('<IIII', buf, 0)
print(f'Header: 0x{hdr[0]:08x} 0x{hdr[1]:08x} {hdr[2]} ops {hdr[3]} bytes')
print(f'Total buffer: {len(buf)} bytes')
print(f'Header says total_size={hdr[3]}')

# Verify command count by scanning
i = 16
count = 0
while i < len(buf):
    opcode = buf[i]
    if opcode == 0x00:
        i += 24
    elif opcode == 0x01:
        i += 48
    elif opcode == 0x03:
        i += 28
    elif opcode == 0x80:
        i += 16
    elif opcode == 0x81:
        i += 48
    else:
        print(f'Unknown opcode 0x{opcode:02x} at offset {i}')
        break
    count += 1
print(f'Scanned {count} commands, header says {hdr[2]}')
assert count == hdr[2], f'Count mismatch: scanned {count} vs header {hdr[2]}'
print('PASS: header command count matches')
"
```

Expected: Header, sizes printed, count matches, PASS.

- [ ] **Step 3: Verify total_size matches file size**

```bash
cd /home/bcloud/strixhalo-npu-setup/experiments
python3 -c "
import npu_instr_gen
import struct

buf = npu_instr_gen.build_insts(32, 128, 32)
hdr = struct.unpack_from('<IIII', buf, 0)
assert hdr[3] == len(buf), f'total_size {hdr[3]} != file size {len(buf)}'
print(f'total_size {hdr[3]} == file size {len(buf)} PASS')
"
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cd /home/bcloud/strixhalo-npu-setup
git add experiments/npu_instr_gen.py
git commit -m "feat: add parameterised GEMM instruction buffer generator"
```

---

### Task 3: Extend npu_raw_submit for generator output

**Files:**
- Modify: `/home/bcloud/npu-sandbox/npu-infer/src/npu_raw_submit.c`
- Rebuild: `/home/bcloud/npu-sandbox/npu-infer/build/npu_raw_submit`

**Interfaces:**
- Consumes: argv[1]=xclbin, argv[2]=insts.bin, argv[3]=A.bin, argv[4]=B.bin, argv[5]=M, argv[6]=K, argv[7]=N, argv[8]=output C path
- Produces: writes NPU output C to file, prints timing and status

- [ ] **Step 1: Read the current npu_raw_submit.c to understand its structure**

```bash
cd /home/bcloud/npu-sandbox/npu-infer/src
wc -l npu_raw_submit.c
head -20 npu_raw_submit.c
# Note: need to understand where inst_buf, A, B, C are set to modify properly
```

- [ ] **Step 2: Modify npu_raw_submit.c to accept command-line args for buffers**

The changes to `npu_raw_submit.c`:

1. Add `read_file` function (already exists for xclbin)
2. Parse new argv arguments for instruction buffer, A, B, C paths, M, K, N dimensions
3. Replace hard-coded matrix size constants with M, K, N
4. Read A and B matrices from binary float32 files
5. Write C output matrix to file after submission
6. Keep the existing DRM ioctl sequence unchanged

```c
// --- New argument parsing (replace the one-arg logic) ---
int main(int argc, char** argv) {
    if (argc < 9) {
        fprintf(stderr, "Usage: %s <xclbin> <insts.bin> <A.bin> <B.bin> <M> <K> <N> <C_out.bin>\n", argv[0]);
        return 1;
    }

    const char* xclbin_path = argv[1];
    const char* insts_path = argv[2];
    const char* A_path = argv[3];
    const char* B_path = argv[4];
    int M = atoi(argv[5]);
    int K = atoi(argv[6]);
    int N = atoi(argv[7]);
    const char* C_path = argv[8];

    // ... (keep existing XRT init, DRM ioctl code) ...

    // --- Replace matrix size constants ---
    size_t A_bytes = M * K * sizeof(float);
    size_t B_bytes = K * N * sizeof(float);
    size_t C_bytes = M * N * sizeof(float);

    // Read instruction buffer from file
    size_t insts_size;
    uint8_t* insts = read_file(insts_path, &insts_size);
    if (!insts) {
        fprintf(stderr, "Failed to read instruction buffer from %s\n", insts_path);
        return 1;
    }

    // Read A matrix
    size_t A_file_size;
    float* A = (float*)read_file(A_path, &A_file_size);
    if (!A || A_file_size != A_bytes) {
        fprintf(stderr, "Failed to read A matrix (%zu bytes expected, got %zu)\n", A_bytes, A_file_size);
        return 1;
    }

    // Read B matrix
    size_t B_file_size;
    float* B = (float*)read_file(B_path, &B_file_size);
    if (!B || B_file_size != B_bytes) {
        fprintf(stderr, "Failed to read B matrix (%zu bytes expected, got %zu)\n", B_bytes, B_file_size);
        return 1;
    }

    // --- Replace BO allocation sizes ---
    // Use insts_size instead of hard-coded 8400
    // Use A_bytes, B_bytes, C_bytes instead of hard-coded tile sizes

    // ... (keep BO creation, insts upload, submission) ...

    // After submission completes, read C and write to file
    {
        FILE* f = fopen(C_path, "wb");
        if (f) {
            fwrite(C_bo_mmap, 1, C_bytes, f);
            fclose(f);
        }
    }
```

- [ ] **Step 3: Build the modified npu_raw_submit**

```bash
cd /home/bcloud/npu-sandbox/npu-infer/build
gcc -std=gnu11 -O2 -o npu_raw_submit ../src/npu_raw_submit.c -lxrt_coreutil -ldl -luuid
echo "Build exit code: $?"
ls -la npu_raw_submit
```

Expected: Build succeeds, binary exists.

- [ ] **Step 4: Test with fresh_insts.bin to verify nothing is broken**

```bash
cd /home/bcloud/strixhalo-npu-setup/experiments

# Create test matrices matching the fresh_insts.bin layout
# fresh_insts.bin uses tile_m=32, tile_k=128, tile_n=32 with 4 A-cols x 8 B-cols
# So M = 32 * 4 = 128, K = 128, N = 32 * 8 = 256
python3 -c "
import numpy as np
M, K, N = 128, 128, 256
A = np.random.uniform(-1, 1, (M, K)).astype(np.float32)
B = np.random.uniform(-1, 1, (K, N)).astype(np.float32)
A.tofile('/tmp/test_A.bin')
B.tofile('/tmp/test_B.bin')
C_ref = A @ B
C_ref.tofile('/tmp/test_C_ref.bin')
print(f'A: {A.shape} B: {B.shape} C_ref: {C_ref.shape}')
print(f'C_ref[0,0] = {C_ref[0,0]:.6f}')
"

# Run with fresh_insts.bin (the original known-working buffer)
sudo /home/bcloud/npu-sandbox/npu-infer/build/npu_raw_submit \
    /opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin \
    /home/bcloud/strixhalo-npu-setup/saved_xclbins/fresh_insts.bin \
    /tmp/test_A.bin /tmp/test_B.bin \
    128 128 256 \
    /tmp/test_C_npu.bin

echo "Exit: $?"

# Compare output
python3 -c "
import numpy as np
C_npu = np.fromfile('/tmp/test_C_npu.bin', dtype=np.float32).reshape(128, 256)
C_ref = np.fromfile('/tmp/test_C_ref.bin', dtype=np.float32).reshape(128, 256)
diff = np.max(np.abs(C_npu - C_ref))
rel = diff / (np.max(np.abs(C_ref)) + 1e-10)
print(f'Max diff: {diff:.6f}, Rel error: {rel:.2e}')
if diff < 0.1:
    print('PASS')
else:
    print('FAIL')
"
```

Expected: The NPU returns correct result matching CPU reference (PASS).

- [ ] **Step 5: Commit**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
git add src/npu_raw_submit.c
git commit -m "feat: extend npu_raw_submit with buffer file I/O and MKN params"
```

---

### Task 4: Automated Test Harness

**Files:**
- Create: `experiments/test_gemm.py`

- [ ] **Step 1: Write the test harness**

```python
#!/usr/bin/env python3
"""Automated GEMM test harness.

Generates random A/B matrices, builds instruction buffer,
submits via npu_raw_submit, reads output, compares against CPU reference.
"""

import numpy as np
import subprocess
import sys
import os
import tempfile
import argparse
from pathlib import Path
from typing import Tuple, Optional

# Add experiments dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_gemm
import npu_instr_gen

# Paths
NPU_RAW_SUBMIT = "/home/bcloud/npu-sandbox/npu-infer/build/npu_raw_submit"
XCLBIN_PATH = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin"


def run_gemm_test(M: int, K: int, N: int,
                  tile_m: int = 32, tile_k: int = 128, tile_n: int = 32,
                  verbose: bool = False,
                  seed: int = 42) -> Tuple[bool, float, Optional[str]]:
    """Run a single GEMM test end-to-end.

    Returns:
        (passed, relative_error, error_message)
    """
    # 1. Generate reference
    A, B, C_ref = reference_gemm.generate_random_gemm(M, K, N, seed=seed)

    # 2. Build instruction buffer
    insts = npu_instr_gen.build_insts(M, K, N, tile_m, tile_k, tile_n)

    # 3. Write temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        insts_path = os.path.join(tmpdir, "insts.bin")
        a_path = os.path.join(tmpdir, "A.bin")
        b_path = os.path.join(tmpdir, "B.bin")
        c_path = os.path.join(tmpdir, "C.bin")

        with open(insts_path, "wb") as f:
            f.write(insts)
        A.tofile(a_path)
        B.tofile(b_path)

        # 4. Submit to NPU
        cmd = [
            NPU_RAW_SUBMIT,
            XCLBIN_PATH,
            insts_path,
            a_path,
            b_path,
            str(M), str(K), str(N),
            c_path,
        ]

        if verbose:
            print(f"  Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            return False, float('inf'), f"Submit failed (rc={result.returncode}): {result.stderr[:200]}"

        # 5. Read output
        if not os.path.exists(c_path):
            return False, float('inf'), "Output file not created"

        C_npu = np.fromfile(c_path, dtype=np.float32).reshape(M, N)

        # 6. Compare
        diff = np.abs(C_npu - C_ref)
        max_diff = np.max(diff)
        rel_error = max_diff / (np.max(np.abs(C_ref)) + 1e-10)

        if verbose:
            print(f"  Max diff: {max_diff:.6f}")
            print(f"  Relative error: {rel_error:.6e}")
            print(f"  C_ref[0,0] = {C_ref[0,0]:.6f}")
            print(f"  C_npu[0,0] = {C_npu[0,0]:.6f}")

        if max_diff > 0.1:
            return False, rel_error, f"Max diff {max_diff:.6f} exceeds tolerance"

        return True, rel_error, None


def main():
    parser = argparse.ArgumentParser(description="Test NPU GEMM kernel correctness")
    parser.add_argument("--M", type=int, default=32, help="M dimension")
    parser.add_argument("--K", type=int, default=128, help="K dimension")
    parser.add_argument("--N", type=int, default=32, help="N dimension")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(f"Testing GEMM {args.M}×{args.K}×{args.N}")

    stats = []
    for i in range(args.iterations):
        passed, error, msg = run_gemm_test(
            args.M, args.K, args.N,
            verbose=args.verbose,
            seed=42 + i
        )
        status = "PASS" if passed else "FAIL"
        if passed:
            print(f"  [{i+1}/{args.iterations}] {status} (rel_err={error:.2e})")
        else:
            print(f"  [{i+1}/{args.iterations}] {status}: {msg}")
        stats.append(passed)

    passed_count = sum(stats)
    print(f"\nResults: {passed_count}/{args.iterations} passed")
    return 0 if passed_count == args.iterations else 1


if __name__ == "__main__":
    sys