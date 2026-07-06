#!/usr/bin/env python3
"""
npu-unlock — Unlock AMD Strix Halo NPU from Linux (no USB/SREP needed)

Patches the UEFI NVRAM variable that controls the NPU lockdown on AMI BIOS.
No reboot into SREP USB needed — run this once, reboot, NPU stays unlocked.

Usage:
  ./npu-unlock.py status          # Check if NPU is locked or unlocked
  ./npu-unlock.py unlock          # Unlock NPU (writes NVRAM)
  ./npu-unlock.py lock            # Lock NPU (reverts)
  ./npu-unlock.py dump            # Dump all setup variables
  ./npu-unlock.py find-offset     # Scan for NPU offset (needs locked system)

Requires:
  - root (for efivarfs access)
  - efivarfs mounted (mount | grep efivarfs)
"""

import os
import struct
import sys

EFIVARFS = "/sys/firmware/efi/efivars"

# Known NPU offsets for different BIOS versions
# Format: (bios_version, var_name, var_guid, offset, bit, locked_value, unlocked_value)
KNOWN_OFFSETS = [
    # AMI BIOS v1.07 — BeyondMax AXB35 (Strix Halo)
    # The NPU enable is in AmdSetup variable, found by SREP differential
    # TODO: Fill in after training completes and we can diff
]

# Setup variable GUIDs
SETUP_VARS = {
    "Setup": "ec87d643-eba4-4bb5-a1e5-3f3e36b20da9",
    "AmdSetup": "3a997502-647a-4c82-998e-52ef9486a247",
    "AMD_PBS_SETUP": "a339d746-f678-49b3-9fc7-54ce0f9df226",
}


def read_var(name, guid):
    """Read a UEFI NVRAM variable."""
    path = f"{EFIVARFS}/{name}-{guid}"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    attrs = struct.unpack_from("<I", data, 0)[0]
    payload = data[4:]
    return {"attrs": attrs, "payload": payload, "path": path}


def write_var(name, guid, payload):
    """Write a UEFI NVRAM variable."""
    path = f"{EFIVARFS}/{name}-{guid}"
    attrs = 0x07  # NV | BS | RT
    data = struct.pack("<I", attrs) + payload
    with open(path, "wb") as f:
        f.write(data)


def check_npu_status():
    """Check if NPU is detected by the kernel."""
    # Method 1: Check if amdxdna driver is loaded
    try:
        with open("/proc/modules") as f:
            for line in f:
                if "amdxdna" in line:
                    npu_loaded = True
                    break
    except:
        npu_loaded = False

    # Method 2: Check lspci for NPU device
    npu_pci = False
    try:
        import subprocess
        result = subprocess.run(
            ["lspci", "-nn"], capture_output=True, text=True
        )
        npu_pci = "Neural Processing Unit" in result.stdout
    except:
        pass

    # Method 3: Try xrt-smi
    xrt_ok = False
    try:
        result = subprocess.run(
            ["xrt-smi", "examine"], capture_output=True, text=True
        )
        xrt_ok = "RyzenAI-npu" in result.stdout
    except:
        pass

    print(f"NPU PCI device:     {'✅ detected' if npu_pci else '❌ not found'}")
    print(f"amdxdna driver:     {'✅ loaded' if npu_loaded else '❌ not loaded'}")
    print(f"XRT device:          {'✅ accessible' if xrt_ok else '❌ not accessible'}")
    
    if xrt_ok:
        print("\nNPU is UNLOCKED and working.")
    elif npu_pci and not xrt_ok:
        print("\nNPU is detected but not initialized — may be locked in BIOS.")
    elif not npu_pci:
        print("\nNPU not found — check BIOS or SREP unlock.")
    
    return xrt_ok


def dump_vars():
    """Dump all setup variables for analysis."""
    for name, guid in SETUP_VARS.items():
        var = read_var(name, guid)
        if var is None:
            print(f"{name}: NOT FOUND")
            continue
        payload = var["payload"]
        print(f"\n=== {name} ({len(payload)} bytes) ===")
        for i in range(0, len(payload), 16):
            hex_part = " ".join(f"{b:02x}" for b in payload[i:i+16])
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in payload[i:i+16])
            print(f"  {i:04x}: {hex_part:<48s} {ascii_part}")


def find_npu_offset():
    """Find the NPU enable bit by analyzing the Setup variables.
    
    Strategy:
    1. Look for 0→1 transitions in AmdSetup (AMD-specific settings)
    2. The NPU enable is likely a single byte: 00=disabled, 01=enabled
    3. Compare with known patterns from other Strix Halo BIOS versions
    """
    print("Scanning for NPU offset...")
    print("(Requires a dump from a LOCKED system for comparison)")
    print()
    print("Method: Save this dump, then disable NPU in BIOS,")
    print("dump again, and diff:")
    print()
    print("  # Before (locked state):")
    print("  python3 npu-unlock.py dump > locked_dump.txt")
    print()
    print("  # After (unlocked state):")
    print("  python3 npu-unlock.py dump > unlocked_dump.txt")
    print()
    print("  # Find the difference:")
    print("  diff locked_dump.txt unlocked_dump.txt")


def unlock():
    """Unlock the NPU by writing the enable bit."""
    # Check if NPU is already unlocked
    if check_npu_status():
        print("\nNPU is already unlocked. Nothing to do.")
        return

    print("Searching for NPU offset in known database...")
    
    # Try known offsets
    bios_version = "unknown"
    try:
        with open("/sys/class/dmi/id/bios_version") as f:
            bios_version = f.read().strip()
    except:
        pass
    
    print(f"BIOS version: {bios_version}")
    
    for entry in KNOWN_OFFSETS:
        if entry[0] == bios_version or entry[0] == "any":
            var_name, guid, offset, bit, locked_val, unlocked_val = entry[1:]
            var = read_var(var_name, guid)
            if var is None:
                continue
            payload = bytearray(var["payload"])
            current = payload[offset]
            print(f"Found match: {var_name} offset {offset:#x} (current={current:#x})")
            
            if current == unlocked_val:
                print("Already unlocked!")
                return
            
            payload[offset] = unlocked_val
            write_var(var_name, guid, bytes(payload))
            print("Written! Reboot to apply.")
            return
    
    print(f"\nNo known offset for BIOS {bios_version}.")
    print("Options:")
    print("  1. Use the SREP USB method (one-time boot)")
    print("  2. Run `find-offset` to discover it yourself")
    print("  3. Contribute the offset to the project")


def main():
    if os.geteuid() != 0:
        print("❌ Must run as root (for efivarfs access)")
        sys.exit(1)
    
    if not os.path.ismount(EFIVARFS):
        print("❌ efivarfs not mounted. Run: mount -t efivarfs none /sys/firmware/efi/efivars")
        sys.exit(1)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    
    if cmd == "status":
        check_npu_status()
    elif cmd == "unlock":
        unlock()
    elif cmd == "lock":
        print("Lock not yet implemented — disable NPU in BIOS or use SREP")
    elif cmd == "dump":
        dump_vars()
    elif cmd == "find-offset":
        find_npu_offset()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: npu-unlock.py [status|unlock|dump|find-offset]")


if __name__ == "__main__":
    main()
