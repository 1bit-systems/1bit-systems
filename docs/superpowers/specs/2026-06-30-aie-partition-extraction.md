# AIE Partition Extraction — Complete Reference (FINAL)

**Target:** `attn.xclbin` (Qwen3-8B-NPU2 model) from FastFlowLM  
**Date:** 2026-06-30  
**Location:** `/opt/fastflowlm/share/flm/xclbins/Qwen3-8B-NPU2/attn.xclbin`

---

## 1. xclbin Format (AXLF v2)

| Field | Offset | Size | Value |
|---|---|---|---|
| Magic | 0x000 | 8 | `xclbin2\0` |
| Header | 0x008 | 440 B | xclbin v2 header |
| Section table | 0x1C0 | 7 × 40 B | 7 sections |

**7 Sections:**

| Section | Type | Size | Description |
|---|---|---|---|
| mem_topology | 6 | 40 B | Memory bank topology |
| **aie_partition** | **32** | **310504 B** | **AIE partition (PDI + metadata)** |
| EMBEDDED_METADATA | 2 | 778 B | FastFlowLM kernel metadata (JSON) |
| IP_LAYOUT | 8 | 40 B | IP core layout |
| MEM_TOPOLOGY | 7 | 40 B | Memory topology (runtime) |
| GROUP_CONNECTIVITY | 27 | 40 B | Group connectivity |
| GROUP_TOPOLOGY | 26 | 40 B | Group topology |

---

## 2. aie_partition Metadata

| Field | Offset | Value |
|---|---|---|
| Schema version | 0x0C | 0 |
| Partition name | 0xB8 | (empty) |
| Operations/cycle | 0xCC | 2048 |
| Inference fingerprint | 0xD0 | `0x5B7F` |
| Pre/post fingerprint | 0xD4 | `0x3039` |
| Column width | 0xD8 | 8 |
| Start columns | 0xDA | (none) |
| PDI UUID | 0x24 | `28d0238d-8ca3-45bc-810f-81effcc87988` |
| DPU kernel ID | 0xB6 | `0x901` |
| CDO group name | 0xB0 | `"DPU"` |

---

## 3. Complete PDI Structure (310,464 bytes)

```
 0x000000 - 0x00014F  │   336 B │ IDPP flatbuffer + partition_info
 0x000150 - 0x013025  │ 77,526 B │ CDO V2 container (BOOTGEN format)
    ├─ 0x150: hdrlen(4) = 4
    ├─ 0x154: ident(4) = "CDO\0" (0x004F4443)
    ├─ 0x158: version(4) = 0x00000200 (V2)
    ├─ 0x15C: data_size(4) = 77526 (total from 0x150)
    ├─ 0x160: checksum(4) = 0xFFAF8AE2
    └─ 0x164: commands(77506 B) = CDO V2 command stream
 0x013026 - 0x013035  │     16 B │ Gap/padding (not NOPs, binary data)
 0x013036 - 0x04BABF  │232,074 B │ Pure AIE2P VLIW kernel
    └─ 14,504 × 128-bit bundles (rem=10 B padding)
 0x04BAC0 - 0x04BCBF  │    512 B │ CDO post-amble: tile ENABLE
    └─ 32 × MASK_WRITE: MEM+CORE+SHIM+NOC × 8 columns
    └─ Format: [addr(4)][mask=1(4)][val=1(4)][hdr=0x00030102(4)]
```

### 3.1 CDO V2 Container Format (BOOTGEN)

From `/home/bcloud/mlir-aie/third_party/bootgen/cdo-binary.c`:

```
CDO V2 header:
  word[0] = hdrlen (uint32, must be >= 4)
  word[1] = identification (for V2: 0x004F4443 = "CDO\0")
  word[2] = version (0x00000200 = V2.0)
  word[3] = data_size (total bytes from word[0] to end of commands)
  word[hdrlen] = checksum of words[0..hdrlen-1]
  words[hdrlen+1..] = CDO V2 commands

CDO V2 command format:
  header_word = (args_len << 16) | cmd_id
  if (args_len == 255): next word is true args_len (long header)
  followed by args_len × uint32 arguments
  cmd_id in bits 0-15, args_len in bits 16-23

Known V2 commands:
  CMD2_MASK_WRITE    = 0x0102  (args=3: addr, mask, val)
  CMD2_WRITE         = 0x0103  (args=2: addr, val)
  CMD2_DMA_WRITE     = 0x0105  (args>=2: addr_hi, addr_lo, data...)
  CMD2_NOP           = 0x0111  (args=0+)
  CMD2_END_MARK      = 0x0100  (args=0)
```

### 3.2 CDO Command Count (Corrected: 77506 B of commands)

| Type | Count | Bytes | Description |
|---|---|---|---|
| MASK_WRITE (0x102) | 96 | 1,536 B | Masked register write |
| WRITE (0x103) | 20 | 240 B | Direct register write |
| DMA_WRITE (0x105) | 155 | 73,620 B | Block data write (VLIW init data) |
| NOP (0x111) | 152 | 608 B | Alignment/padding |
| **Total** | **423** | **78,004 B** | (including overhead) |

DMA_WRITE data payload contains VLIW kernel code for tile program memory loading:  
~62,256 B of VLIW init data across 8 columns, with 2 kernel variants (A=485 bundles for even cols, B=487 for odd cols).

### 3.3 Tile Address Format

```
addr = 0x00200000 + col * 0x01000000 + type * 0x10000 + offset
```

**4 Tile Types per Column:**

| Type Byte | Abbr | Function |
|---|---|---|
| `0x23` | MEM | Memory tile (L2 shared SRAM, DMA) |
| `0x33` | CORE | Compute tile (AIE core, VLIW execution) |
| `0x43` | SHIM | Shim tile (PL/DMA interface to host) |
| `0x53` | NOC | NOC tile (interconnect router) |

**8 Columns:** 0, 2, 4, 6, 8, 10, 12, 14  
**32 Tiles Total:** 8 × 4

---

## 4. AIE2P VLIW Kernel Analysis

### 4.1 Core Stats

| Metric | Value |
|---|---|
| Bundles | **14,504** (128-bit each) |
| Bundles with 0xBA prefix | 2,108 (14.5%) |
| Kernel variants | 2 (A=485, B=487 bundles) |
| Variant difference | ~20 B (data section offset) |
| Algorithm | BF16 inner-product GEMM |
| Inner loop iterations | 12 |
| Inner loop stride | 16 elements |
| Inner dimension | 60 (`r1 = 0x3C`) |
| Accumulation | FP32 from BF16 multiplies |

### 4.2 Identified Registers

| Class | Registers |
|---|---|
| Vector | `x0` – `x9` (256-bit BF16 vectors) |
| BF16 accumulators | `bmll0` – `bmll2`, `wl5` – `wl9` |
| Pointers | `p0` – `p3` |
| Scalars | `r0` – `r24` |
| Loop control | `lc` (count), `le` (end), `ls` (start) |
| Special | `vaddsign0`, `crrnd`, `cml2` |

### 4.3 Disassembly Highlights

**Prologue** (bundles 0-4, 80 B):
```
vsel.16       x5, x2, x10, r17
vst           wl5, [p0], #0x20
vconv.bf16.fp32 x7, cml2
vlt.bf16      r20, x7, x5
extend.u16    r21, r20
vsel.16       x9, x3, x5, r21
vst           wl9, [p1], #0x20
ret           lr
```

**Main Loop Setup** (bundles 7-9):
```
lda.s16       r3, [p1], #0x2
movxm         p3, #0x73c24
lda.s8        r2, [p3, #0x0]
movx          r24, #0x0
lda.s16       r4, [p2], #0x2
movx          r0, #0x10              ; stride = 16
vbcst.16      x0, r24
movx          r1, #0x3c              ; inner dim = 60
vinsert.16    x1, x0, #0, r1
movxm         le, #0x500
movx          lc, #0xc               ; 12 iterations
mov           vaddsign0, #0x1
```

**Inner Loop Body** (pointer-chase MAC):
```
lda.s16       r4, [p2], #0x2         ; weight LUT
ashl          r6, r4, r0             ; offset = addr × 16
lda.s16       r3, [p1], #0x2         ; activation LUT
ashl          r5, r3, r0
vinsert.32    x2, x1, #0, r6         ; sparse gather
vinsert.32    x3, x0, #0, r5
vmov          bmll1, x2              ; BF16 MAC low
vmov          bmll0, x3
vsub.f        dm2, dm0, dm1, r1      ; FP32 accumulate
vconv.bf16.fp32 wl6, bmll2
vextract.16   r7, x6, #0x0, vaddsign0
vinsert.16    x4, x0, #0, r7
```

### 4.4 0xBA Chess-Extended Encoding

**14.5%** of bundles (2,108/14,504) use the Chess/AMD proprietary `0xBA` prefix to encode extended VLIW instructions. These:
- Encode **6 operations per bundle** (beyond standard 4-slot AIE2P spec)
- Use **10-byte extended slots** (0xBA + 9 payload bytes)
- Crash LLVM disassembler (`AIE2PGenAsmWriter.inc:5096: Bits != 0`)
- Require proprietary AMD xchesscc to disassemble

---

## 5. CDO Post-Amble: Tile ENABLE Sequence

Located at PDI offset `0x4BAC0` (512 B, 32 commands):

```
For each column C in {0, 2, 4, 6, 8, 10, 12, 14}:
  MASK_WRITE 0x{C}232000  mask=1 val=1  → MEM ENABLE
  MASK_WRITE 0x{C}332000  mask=1 val=1  → CORE ENABLE
  MASK_WRITE 0x{C}432000  mask=1 val=1  → SHIM ENABLE
  MASK_WRITE 0x{C}532000  mask=1 val=1  → NOC ENABLE
```

Each command is a 16-byte aligned block: `[addr(4)][mask(4)][val(4)][hdr=0x00030102(4)]`

---

## 6. Execution Pipeline

```
FastFlowLM Runtime:
  register_xclbin(path) → XRT load
  create_app(kernel_id) → NPU app context
  gen_layer_seq(...)    → tile instruction seq
  cmds2seq(ops)         → serialize → seq_data
  _setup_kernel(seq)    → map → DRM buffer
  DRM_IOCTL_XDNA_SUBMIT → submit to AMD XDNA driver
```

---

## 7. Key Artifacts

| File | Size | Description |
|---|---|---|
| `pdi_image.bin` | 310,464 B | Raw PDI container from aie_partition |
| `cdo_payload.bin` | 77,506 B | CDO V2 commands (423 cmds, BOOTGEN format) |
| `cdo_decoded.txt` | ~53 KB | Full CDO command decode |
| `vliw_kernel_pure.bin` | **232,074 B** | Corrected VLIW kernel (14,504 bundles) |
| `cdo_trailer.bin` | **512 B** | CDO post-amble (32 tile ENABLE commands) |
| `extraction_summary.json` | ~6 KB | Comprehensive metadata |
| `disasm_prologue.txt` | 2.5 KB | Clean disassembly (~370 instructions) |
| `vliw_prologue.elf` | 576 B | ELF-wrapped VLIW for llvm-objdump |

All files in `/tmp/fastflowlm_extracted/`

---

## 8. Source Code References

| File | Contents |
|---|---|
| `/home/bcloud/mlir-aie/third_party/bootgen/cdo-binary.c` | CDO V2 format implementation |
| `/home/bcloud/mlir-aie/third_party/bootgen/cdo-command.h` | CDO command definitions |
| `/home/bcloud/mlir-aie/third_party/bootgen/cdo-binary.h` | CDO data structures |
| `/home/bcloud/mlir-aie/.venv/bin/llvm-objdump` | AIE2P disassembler |

---

## 9. Full AIE2P Disassembly (COMPLETE)

Using the Chess darts disassembler from the AMD AIE toolchain (`xchesscc +d`), 100% of the VLIW kernel was successfully decoded — including all 14.5% of 0xBA-extended bundles.

### Command
```
XILINXD_LICENSE_FILE=/home/bcloud/Downloads/Xilinx.lic \
/home/bcloud/torch2aie/toolchain/bin/xchesscc -p me -C Release_LLVM \
  -D__AIENGINE__ -D__AIE_ARCH__=21 -D__AIEARCH__=21 \
  -I $AIETOOLS/include -P $AIETOOLS/data/aie2p/lib \
  +W darts,-allow-segment-padding +d -f kernel_full.elf
```

### Results
| Metric | Value |
|---|---|
| Total instruction lines | **40,298** |
| Lines with extended encoding | **All decoded** (14.5% 0xBA bundles) |
| Output file | `disasm_full.txt` (7.0 MB) |
| Unique operations | **91** |
| Kernel code offset | 7,546 B into VLIW area (data tables precede) |
| Kernel replication | ~8,480 B per copy × 26+ copies =

### Key Instructions Identified
| Instruction | Count | Purpose |
|---|---|---|
| `VLDA.CONV.fp32.bf16` | 691 | BF16→FP32 load with conversion |
| `VMAC.f` | 1,009 | FP32 multiply-accumulate |
| `VMUL.f` | 858 | FP32 multiply |
| `VST.SRS.4x` | 1,807 | Store with shift-round-saturate (4 lanes) |
| `VST.CONV.bf16.fp32` | 250 | Store with BF16 down-conversion |
| `VLDB.128` | 1,544 | 128-bit vector load (BF16 data) |
| `LDA.s16` / `LDA.u16` | 522+ | 16-bit scalar load (index/pointer arithmetic) |
| `NOPX` | 14,511 | NOP slot padding (pipeline hazards) |
| `JL` | 409 | Jump/branch |

## 10. Remaining Open Items

1. **Cross-compare attn.xclbin variants** — 7 size groups across 36 models
2. **IRON toolchain VLIW decode** — check `/home/bcloud/mlir-aie/npu2_40_toolchain/python/iron/`
