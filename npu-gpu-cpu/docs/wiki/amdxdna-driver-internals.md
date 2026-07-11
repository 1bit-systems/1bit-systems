# amdxdna Staging Driver Internals — Task 1.1

## Overview

The `amdxdna` staging driver (`drivers/staging/amdxdna/` upstream, now at
`drivers/accel/amdxdna/` in kernel 7.0+) manages the AMD XDNA NPU through
a DRM accel device (`/dev/accel/accel0`). AMD has upstreamed the driver
through DRM accelerators framework (drivers/accel/). On Strix Halo (NPU4),
it uses the `aie2_*` AIE2 backend (`aie2_pci.c`, `aie2_message.c`, etc.).

## Key Files

| File | Purpose |
|------|---------|
| `aie2_pci.c` | PCI probe, device init, firmware loading, SMU/PSP lifecycle |
| `aie2_pci.h` | Core data structures: `amdxdna_dev_hdl`, `mgmt_mbox_chann_info` |
| `aie2_message.c` | Mailbox message protocol (send/wait) for firmware commands |
| `aie2_msg_priv.h` | Opcode definitions for all firmware messages |
| `aie2_hwctx.c` | Hardware context creation/destruction (column allocation) |
| `aie2_ctx.c` | Context management + gpu_scheduler integration |
| `aie2_ctx_runqueue.c` | Context scheduling runqueue (priorities) |
| `amdxdna_mailbox.c` | Low-level mailbox transport (shared SRAM rings) |
| `npu4_regs.c` | NPU4 register definitions (BAR layout, SRAM offsets) |
| `aie2_psp.c` | PSP firmware loading for NPU |
| `aie2_smu.c` | SMU power management (DPM, clock, power on/off) |

## Device Init Sequence (aie2_hw_start in aie2_pci.c)

```
1. pci_enable_device()
2. pci_set_master()
3. BAR mapping (SRAM, mailbox, PSP, SMU regs)
4. aie_psp_start()         — Load NPU firmware via PSP
5. aie_smu_init()          — SMU power management init (FIXED: PSP before SMU)
6. aie2_get_mgmt_chann_info() — Poll FW_ALIVE_OFF → read mgmt_mbox_chann_info
7. xdna_mailbox_start_channel(mgmt_chann)
8. aie2_mgmt_fw_init() → runtime_cfg, pasid, time_quantum, xdna_reset
9. aie2_mgmt_fw_query() → firmware version
10. aie2_query_aie_version() + aie2_query_aie_metadata()
```

## MMIO Layout

The NPU has 3-4 PCI BARs depending on generation:

- **BAR0 (SRAM)**: Main SRAM region — firmware alive pointer at `SRAM_REG_OFF(FW_ALIVE_OFF)`,
  management mailbox channel info at variable SRAM address (written by firmware).
  Mailbox ring buffers live in SRAM at `MBOX_CHANN_OFF + slot_id × CHAN_SLOT_SZ (8KB)`.
- **BAR1 (Mailbox)**: Doorbell registers for mailbox signalling
  - `X2I_TAIL` / `X2I_HEAD` — host→fifo write pointer
  - `I2X_TAIL` / `I2X_HEAD` — fifo→host write pointer
  - `DOORBELL_OFFSET` — ring doorbell to trigger firmware IRQ
- **BAR2 (PSP)**: PSP control registers for firmware loading
- **BAR3 (SMU)**: SMU power management registers (DPM levels, clock control)

NPU4-specific offset tables in `npu4_regs.c`:
```c
struct aie_bar_off_pair npu4_sram_offs[] = {
    {MBOX_CHANN_OFF, 0x800},   // Mailbox channel SRAM offset
    {FW_ALIVE_OFF,   0x1FFC},  // Firmware alive pointer
};
```

## Firmware ABI (Mailbox Protocol)

### Message Format

Messages are fixed-size (64 bytes) packets written to the management channel:
```
struct xdna_mailbox_msg {
    u32 opcode;        // MSG_OP_* from aie2_msg_priv.h
    u32 is_response;   // 0 = request, 1 = response
    u32 state;         // Channel state
    u32 sender;        // 0 = host, 1 = firmware
    u64 data[6];       // Up to 48 bytes payload
};
```

### Key Firmware Opcodes

| Opcode | Name | Purpose |
|--------|------|---------|
| 0x02 | MSG_OP_CREATE_CONTEXT | Allocate NPU column range + create hw context |
| 0x03 | MSG_OP_DESTROY_CONTEXT | Release hw context |
| 0x07 | MSG_OP_SYNC_BO | Synchronize buffer ownership |
| 0x0C | MSG_OP_EXECUTE_BUFFER_CF | Execute command (control-flow format) |
| 0x10 | MSG_OP_EXEC_DPU | Execute DPU kernel |
| 0x11 | MSG_OP_CONFIG_CU | Configure compute unit |
| 0x17 | MSG_OP_EXEC_NPU | Execute NPU (ELF-based) kernel |
| 0x18 | MSG_OP_CHAIN_EXEC_NPU | Chained multi-kernel NPU execution |
| 0x103 | MSG_OP_ASSIGN_MGMT_PASID | Assign IOMMU PASID |

### Context Creation Flow

```
amdxdna_drm_ioctl(CREATE_HWCTX)
  → aie2_hwctx_start()
     → xrs_reserve_columns()         — Column range solver
     → aie2_create_context()         — Mailbox MSG_OP_CREATE_CONTEXT
     → xdna_mailbox_create_channel() — Per-context mailbox channel
     → aie2_config_cu()              — Configure AIE cores for this context
```

## Ring Protocol / Command Submission

Command submission uses a two-layer approach:

1. **DRM gpu_scheduler**: Uses `drm_sched_entity` + `drm_sched_job` for
   queue management. The `aie2_cmd_submit()` callback feeds jobs to the
   firmware.

2. **Mailbox channel**: Each context gets a dedicated mailbox channel with
   its own ring buffer in SRAM. The host writes command packets into the
   ring buffer, then rings a doorbell register. The firmware processes
   commands and writes completion status back.

3. **Execution messages**: The actual execution payload is packed through
   `aie2_exec_msg_ops` function pointer table:
   - `init_cu_req()` — Build CU config request
   - `init_dpu_req()` — Build DPU execution request
   - `fill_cf_slot()` — Fill control-flow command slot
   - `fill_elf_slot()` — Fill ELF kernel load slot

## Memory Management (GEM)

- Buffers allocated as `amdxdna_gem_obj` (GEM objects)
- Two paths: **carved-out** (pre-allocated reserved memory) and **CMA** (contiguous allocator)
- Can import dma-buf from amdgpu for zero-copy sharing
- IOMMU SVA (Shared Virtual Addressing) for user-space pointer support
- Buffer sync via `MSG_OP_SYNC_BO`

## Key Data Structures

```c
struct amdxdna_dev_hdl {          // NPU device handle
    sram_base, mbox_base,         // Mapped BARs
    psp_hdl, smu_hdl,             // PSP/SMU handles
    mgmt_info,                    // Mailbox channel info (from firmware)
    mgmt_fw_version,              // Firmware version
    total_col,                    // Number of AIE columns
    mgmt_chann,                   // Management mailbox channel
    ctx_rq,                       // Context runqueue
    tdr,                          // Task recovery
};

struct mgmt_mbox_chann_info {     // Written by firmware at FW_ALIVE_OFF
    x2i_tail, x2i_head,           // Host→fifo ring pointers
    x2i_buf, x2i_buf_sz,          // Host→fifo ring buffer address + size
    i2x_tail, i2x_head,           // Fifo→host ring pointers
    i2x_buf, i2x_buf_sz,          // Fifo→host ring buffer
    magic (0x55504e5f = "_NPU"),  // Magic number
    msi_id, prot_major, prot_minor,
};
```

## Integration Points for amdgpu NPU IP Block

The `amdgpu_npu.c` patch must replicate this init sequence:

```
amdgpu_npu_early_init():
  → PCI function discovery (PCI slot function 1)
  → BAR resource sizing

amdgpu_npu_sw_init():
  → Firmware request (amdgpu_ucode_request)
  → SRAM/mailbox BAR mapping
  → Ring init (amdgpu_ring pattern)
  → IOMMU SVA setup

amdgpu_npu_hw_init():
  → pci_enable_device(npu_pdev)
  → pci_set_master(npu_pdev)
  → aie_psp_start(npu->psp) — Load NPU firmware via PSP
  → aie_smu_init(npu->smu) — Init SMU
  → Poll FW_ALIVE_OFF until non-zero
  → Read mgmt_mbox_chann_info from SRAM
  → xdna_mailbox_start_channel(mgmt_chann)
  → aie2_mgmt_fw_init() — Runtime config + calibration
  → aie2_mgmt_fw_query() — Version check
  → Query AIE version + metadata
```

The FIRMWARE (binary blob) performs its own init — we just need to:
1. Power on the NPU (PSP → SMU)
2. Wait for firmware to write the alive pointer
3. Read the mailbox channel info
4. Start communicating through the mailbox

This is exactly what amdxdna does in `aie2_hw_start()`, and our
`amdgpu_npu_hw_init()` stub must replicate it.
