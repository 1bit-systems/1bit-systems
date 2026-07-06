# NPU Unlock — From Linux (No USB Needed)

Two tools to unlock the AMD XDNA2 NPU on Strix Halo from userspace:

- **`npu_unlock.c`** — C binary (can embed in daemon)
- **`npu-unlock.py`** — Python script (for testing/discovery)

## How It Works

The NPU lockdown is a single byte in the UEFI NVRAM `AmdSetup` variable
(written by AMI's `SetupUtilityApp`). Flip that byte, the NPU stays
unlocked across reboots. No USB. No SREP. No firmware mod.

## Usage

```bash
# Check if NPU is already unlocked
sudo ./npu_unlock status

# Unlock (one-time, persists across reboots)
sudo ./npu_unlock unlock

# Dump AmdSetup for offset discovery
sudo ./npu_unlock dump
```

## Finding the NPU Offset (For New BIOS Versions)

If your BIOS version isn't in the known offsets database:

```bash
# 1. Dump current state (NPU unlocked)
sudo ./npu_unlock dump > unlocked.txt

# 2. Reboot, disable NPU in BIOS, dump again
sudo ./npu_unlock dump > locked.txt

# 3. Diff to find the byte that changed
diff locked.txt unlocked.txt
```

The NPU enable is typically a `00→01` transition at a single byte in
the `AmdSetup` variable (offsets 0x20-0xC0 range).

Once found, contribute the offset:
```c
{ "1.07", "AmdSetup", AMD_GUID, 0x??, 0xFF, 0x00, 0x01 },
```

## Embedding in Your Binary

The `npu_unlock.c` file is self-contained with no dependencies beyond
libc. Compile it into your daemon:

```c
// Call this at startup — if NPU is locked, unlock it
if (npu_status() == LOCKED) {
    npu_unlock_known_offset();
}
```

Or compile standalone:
```bash
gcc -O2 -o npu_unlock npu_unlock.c
sudo ./npu_unlock unlock
```

## See Also

- [`tools/srep-usb/`](../srep-usb/) — USB/SREP method (fallback)
- [`docs/npu-unlock-srep.md`](../../docs/npu-unlock-srep.md) — Technical details
- [`setup_var.efi`](https://github.com/datasone/setup_var.efi) — Alternative UEFI shell tool
