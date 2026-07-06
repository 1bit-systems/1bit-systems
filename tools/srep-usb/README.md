# NPU Unlock USB — SmokelessRuntimeEFIPatcher

**Purpose:** Unlocks the AMD XDNA2 NPU on Strix Halo (Ryzen AI Max+ 395).

Burn these files to a FAT32 USB drive, boot from it (UEFI mode), and SREP patches the AGESA NPU lockdown at runtime. No BIOS flash needed.

## Files

| File | Size | Purpose |
|------|------|---------|
| `BOOTX64.efi` | 1.0 MB | EFI bootloader (unsigned, loads SREP) |
| `SmokelessRuntimeEFIPatcher.efi` | 108 KB | SREP v0.20 runtime patcher |
| `Oniguruma.efi` | 434 KB | Regex library dependency |
| `SREP_Config.cfg` | 96 B | Patch config — flips NPU lockdown check |
| `startup.nsh` | 158 B | UEFI shell boot script |

## Usage

```bash
# Prepare USB (from Linux)
sudo dd if=/dev/zero of=/dev/sdX bs=1M count=1    # WARNING: check device!
sudo mkfs.vfat -F 32 /dev/sdX1
sudo mount /dev/sdX1 /mnt
sudo cp -r tools/srep-usb/* /mnt/
sudo umount /mnt

# Boot from USB
# 1. Insert USB, restart
# 2. Press F12/ESC/DEL for boot menu
# 3. Select USB drive (UEFI mode)
# 4. SREP runs automatically — black screen, ~2 seconds
# 5. Remove USB, reboot into Linux

# Verify NPU is unlocked
xrt-smi examine
# → Should show RyzenAI-npu5
xrt-smi validate
# → Should show TOPS: 51.0 PASS (with firmware 255.0.11.71)
```

## What It Patches

The config flips a single function in AGESA:
```asm
; Original (disabled):
xor  eax, eax    → return 0 (NPU locked)

; Patched (enabled):
mov  al, 1       → return 1 (NPU unlocked)
```

## Requirements

- AMD Strix Halo (Ryzen AI Max+ 395) — e.g., Sixunited AXB35 BeyondMax
- UEFI boot mode (no CSM)
- XRT kernel driver loaded (`amdxdna.ko`)

## See Also

- [`docs/npu-unlock-srep.md`](../../docs/npu-unlock-srep.md) — Technical documentation
- [`docs/npu-full-unlock-plan.md`](../../docs/npu-full-unlock-plan.md) — Complete unlock roadmap
