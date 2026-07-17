# Boot Configuration — 40-Column NPU Unlock with Secure Boot

> How to enable the full 40 XDNA 2 AIE columns on Strix Halo while keeping
> Secure Boot enabled. Verified on **Bosgame BeyondMax** (Ryzen AI Max+ 395,
> BIOS 1.09, AMI Aptio 5).

## The Goal

The stock amdxdna driver defaults to **8 columns** (32 tiles). The NPU has
**40 columns** (160 tiles) physically available, gated by the kernel module
parameter `aie2_max_col=40`. The challenge is getting this parameter applied
at boot while Secure Boot is ON — because `modprobe.d` parameters won't work
if the module is built-in or loaded by initramfs before userspace applies them.

## Verified State (2026-07-16)

| Check | Status | How to Verify |
|---|---|---|
| **Secure Boot** | ✅ Enabled | `mokutil --sb-state` |
| **Kernel Lockdown** | ✅ None (most permissive) | `cat /sys/kernel/security/lockdown` → `none [integrity] confidentiality` |
| **NPU 40 columns** | ✅ Active | `sudo cat /sys/module/amdxdna/parameters/aie2_max_col` → `40` |
| **NPU device** | ✅ Present | `ls -la /dev/accel/accel0` |
| **NPU firmware** | ✅ Loaded | `cat /sys/class/accel/accel0/device/fw_version` → `1.0.0.166` |

## How It Works

The trick: **pass `amdxdna.aie2_max_col=40` via the kernel command line in
GRUB**, not via `/etc/modprobe.d/`.

Why the cmdline approach works with Secure Boot:

1. The kernel command line is **signed as part of the GRUB+shim chain** —
   Secure Boot verifies the signature on the bootloader and kernel, not the
   individual cmdline parameters.
2. Module parameters passed via cmdline are applied **during module init**,
   before the module becomes available to userspace — so they work even for
   built-in or initramfs-loaded modules.
3. `/etc/modprobe.d/` parameters only apply when `modprobe` loads the module
   from userspace — too late for modules compiled into the kernel or loaded
   by initramfs.

## Step-by-Step Replication

### 1. Set the kernel parameter in GRUB

```bash
# Edit GRUB config
sudo nano /etc/default/grub
```

Find the line `GRUB_CMDLINE_LINUX_DEFAULT` and append the NPU parameters:

```bash
GRUB_CMDLINE_LINUX_DEFAULT="...existing params... amdxdna.aie2_max_col=40 amdgpu.gfxoff=0"
```

**Full example from working config:**

```bash
GRUB_CMDLINE_LINUX_DEFAULT="amdgpu.no_system_mem_limit=1 ttm.pages_limit=31457280 amdxdna.aie2_max_col=40 amdgpu.gfxoff=0"
```

Additional recommended params:
- `amdgpu.no_system_mem_limit=1` — allow GPU to use all available system memory
- `ttm.pages_limit=31457280` — increase TTM page limit for large model allocations
- `amdgpu.gfxoff=0` — disable GFX off to prevent hangs under sustained NPU/GPU load

### 2. Update GRUB

```bash
sudo update-grub
```

### 3. Install the signed DKMS module (if using Secure Boot)

If you're running the **stock Ubuntu kernel** with Secure Boot, the amdxdna
module is already signed by Ubuntu's kernel signing key. No extra steps needed.

If you're building a **custom module** or applying patches that aren't signed:

```bash
# Option A: Sign the module with your own MOK key
sudo mokutil --import /path/to/module-signing-key.der
# Reboot, enroll the key in MOK Manager, then:
sudo kmodsign sha512 /path/to/mok.key /path/to/mok.crt /lib/modules/$(uname -r)/kernel/drivers/accel/amdxdna/amdxdna.ko

# Option B: Use the Ubuntu build system to keep stock signed module
# (recommended — stock amdxdna v0.7.0 supports aie2_max_col=40)
```

With the **stock Ubuntu kernel** (7.0.0-27-generic in this case) the stock
`amdxdna.ko` already understands `aie2_max_col=40` — no patching needed.

### 4. Reboot

```bash
sudo reboot
```

### 5. Verify

```bash
# Check Secure Boot is still enabled
mokutil --sb-state
# → SecureBoot enabled

# Check kernel lockdown is still permissive
cat /sys/kernel/security/lockdown
# → none [integrity] confidentiality

# Check the 40-column parameter took effect
sudo cat /sys/module/amdxdna/parameters/aie2_max_col
# → 40

# Confirm boot cmdline
cat /proc/cmdline
# → ... amdxdna.aie2_max_col=40 ...

# Check NPU is alive
ls -la /dev/accel/accel0
cat /sys/class/accel/accel0/device/fw_version
```

## What NOT to Do

| Approach | Why It Fails |
|---|---|
| `/etc/modprobe.d/amdxdna.conf` with `options amdxdna aie2_max_col=40` | Works for manually-loaded modules but not if amdxdna is built-in or loaded by initramfs before userspace runs |
| `modprobe -r amdxdna && modprobe amdxdna aie2_max_col=40` | Only lasts until next reboot — and requires root + a clean unload (which may not work if NPU is busy) |
| Disabling Secure Boot | Unnecessary — the cmdline approach works fine with Secure Boot enabled. Disabling it weakens your boot chain for no benefit |
| `amdxdna.fw_patches_enable=1` | Kernel ignores this parameter (cosmetic only). Present in some earlier configs but has no functional effect |

## Boot Order (efibootmgr -v)

The verified boot chain on this hardware:

```
Boot0000* Ubuntu → shimx64.efi → grubx64.efi → kernel (with amdxdna.aie2_max_col=40)
```

Secure Boot chain: **BIOS → shim (MOK manager) → GRUB → kernel**
- shim is signed by Microsoft
- GRUB is signed by shim MOK
- Kernel is signed by Ubuntu/DKMS key

The driver parameter is just text on the kernel command line — it doesn't
affect the signature chain.

## Troubleshooting

### NPU device missing after reboot

```bash
# Check module is loaded
lsmod | grep amdxdna
# → amdxdna 172032 0

# Check dmesg for init errors
sudo dmesg | grep amdxdna
# → [drm] Load firmware amdnpu/17f0_11/npu_7.sbin
# → [drm] Initialized amdxdna_accel_driver 0.7.0 for 0000:c6:00.1

# If not loaded, try manually
sudo modprobe amdxdna
```

### Secure Boot blocking module load

```bash
# Check if module is signed
modinfo amdxdna | grep sig
# Should show: sig_id: PKCS#7, signer: ...

# If unsigned, enroll a MOK key and sign it (see Step 3 above)
```

### 40 columns not taking effect

```bash
# Check the actual module parameter value
sudo cat /sys/module/amdxdna/parameters/aie2_max_col
# If this shows 8 (not 40), the cmdline parameter isn't being passed

# Verify /etc/default/grub contains the param
grep aie2_max_col /etc/default/grub

# Check that update-grub ran successfully
grep aie2_max_col /boot/grub/grub.cfg | head -5
# Should show: options amdxdna aie2_max_col=40
```

## Hardware Reference

This was verified on:

| Component | Detail |
|---|---|
| **System** | Bosgame BeyondMax AXB35-02 |
| **BIOS** | AMI Aptio 5, version 1.09 |
| **CPU** | AMD Ryzen AI Max+ 395 (Strix Halo) |
| **NPU** | AMD XDNA 2, NPU5/VE2, PCI 0000:c6:00.1 |
| **GPU** | Radeon 8060S (gfx1151, RDNA 3.5) |
| **Memory** | 128 GB unified LPDDR5X |
| **Kernel** | Ubuntu 7.0.0-27-generic |
| **Driver** | amdxdna v0.7.0 (stock) |
| **Firmware** | amdnpu/17f0_11/npu_7.sbin v1.0.0.166 |
