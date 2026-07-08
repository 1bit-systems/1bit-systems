# NPU Unlock — SmokelessRuntimeEFIPatcher (SREP)

**Purpose:** Unlocks the AMD XDNA2 NPU on consumer Strix Halo (Ryzen AI Max+ 395) systems. Without this patch, AMD's AGESA firmware returns the NPU in a locked/disabled state and the inference engine cannot access it.

## How It Works

A single `FastPatch` in SREP's config flips the NPU lockdown check:

```
Op FastPatch
Pattern: 32C0488B5C2408488B7C2410C3
B001
Op Exec
```

| Hex | Assembly | Meaning |
|-----|----------|---------|
| `32 C0` | `xor eax, eax` | Return 0 — NPU locked |
| → `B0 01` | `mov al, 1` | Return 1 — NPU unlocked |
| `48 8B 5C 24 08` | `mov rbx, [rsp+8]` | Epilogue |
| `48 8B 7C 24 10` | `mov rdi, [rsp+10]` | Epilogue |
| `C3` | `ret` | Return |

The remaining bytes are function epilogue — left untouched.

## Files

- `BOOTX64.efi` — Main EFI bootloader (unsigned, boot from USB)
- `SmokelessRuntimeEFIPatcher(020).efi` — SREP v0.20 runtime patcher
- `Oniguruma.efi` — Regex library dependency
- `SREP_Config.cfg` — Patch configuration (above)
- `startup.nsh` — UEFI shell boot script

## Usage

1. Format a USB drive as FAT32
2. Copy the files to the root
3. Boot from the USB drive (UEFI mode)
4. SREP applies the patch at runtime — no BIOS flash needed
5. Reboot into Linux — `xrt-smi examine` should show the NPU

## Requirements

- AMD Strix Halo (Ryzen AI Max+ 395) system
- UEFI boot mode (no CSM/Legacy)
- XRT kernel driver loaded (`amdxdna.ko`)

## Why This Exists

AMD ships the XDNA2 NPU disabled on consumer Strix Halo silicon. The hardware is fully capable — 50 TOPS INT8, 32 AIE2P tiles — but the AGESA firmware gates it behind a software check intended for enterprise licensing. This patch removes that gate, allowing the open-source fused layer engine to drive the NPU at 291 tok/s (30 KB binary).

## References

- [SmokelessRuntimeEFIPatcher](https://github.com/SmokelessCPU/SmokelessRuntimeEFIPatcher)
- [1bit.systems performance benchmarks](performance.md)
