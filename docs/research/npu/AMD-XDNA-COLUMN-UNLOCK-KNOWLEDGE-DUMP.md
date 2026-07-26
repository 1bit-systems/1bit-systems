# AMD XDNA NPU Column Unlock — Knowledge Dump

## Overview

AMD XDNA NPUs are built from **AI Engine (AIE) tiles** arranged in a 2D grid of **columns × rows**. Each column contains compute, memory, and shim tiles. The NPU's TOPS scales linearly with column count:

```
TOPS ≈ 4096 * total_col * clock_freq / 1000000
```

The "40 column unlock" refers to getting the driver/firmware to expose all physically available AIE columns on a given die, rather than the 4–8 that ship by default.

## Hardware Variants

| Code | NPU # | PCI ID | Generation | Columns (default) | Example APU |
|------|-------|--------|------------|-------------------|-------------|
| NPU1 | XDNA 1 | 1502 | AIE2 | 4 | Ryzen 7040 (Phoenix) |
| NPU3 | XDNA 1 | 17f0 rev < 0x10 | AIE4 | 3 | Ryzen 8040 (Hawk Point) |
| NPU4 | XDNA 2 | 17f0 rev 0x10 | AIE2 | 8 | Ryzen AI 300 (Strix Point) |
| NPU5 | XDNA 2 | 17f0 rev 0x11 | VE2 | 8 | Ryzen AI MAX 395 (Strix Halo) |
| NPU6 | XDNA 2 | ? | ? | ? | Future |

## Repository

**Primary repo:** https://github.com/amd/xdna-driver
- `main` branch — active development
- `ve2_upstream` branch — VE2 (NPU5/Strix Halo) path
- Branches `1.4` through `1.8` — release tags
- `vai_6.1`, `vai_6.2` — Vitis AI integration branches

**Upstream kernel (torvalds/linux):**
- `drivers/accel/amdxdna/` — in-tree driver (lags behind out-of-tree)

## Driver Architecture — Column Count Control

### Path A: AIE2 (`src/driver/amdxdna/aie2_pci.c`)

Used by: NPU1 (Phoenix), NPU4 (Strix Point)

**Module parameter:**
```c
#define AIE2_MAX_COL 128
uint aie2_max_col = AIE2_MAX_COL;
module_param(aie2_max_col, uint, 0600);
MODULE_PARM_DESC(aie2_max_col, "Maximum column could be used");
```

**Column assignment:**
```c
ndev->total_col = min(aie2_max_col, ndev->metadata.cols);
```

`ndev->metadata.cols` comes from `aie2_query_aie_metadata()` — a firmware mailbox query. The driver cannot exceed what the firmware reports.

---

### Path B: AIE4 (`src/driver/amdxdna/aie4_pci.c`)

Used by: NPU3 (Hawk Point / VF mode), some NPU4

**Module parameter:**
```c
#define AIE4_MAX_COL 128
uint aie4_max_col = AIE4_MAX_COL;
module_param(aie4_max_col, uint, 0600);
MODULE_PARM_DESC(aie4_max_col, " Maximum column could be used");
```

**Column assignment:**
```c
ndev->total_col = min(aie4_max_col, ndev->metadata.cols);
```

But wait — the **upstream kernel** (`drivers/accel/amdxdna/aie4_pci.c`) has this hardcoded:
```c
#define AIE4_TOTAL_COLUMN  3
// ...
ndev->total_col = min(AIE4_TOTAL_COLUMN, ndev->aie.metadata.cols);
req.partition_col_count = AIE4_TOTAL_COLUMN;  // also hardcoded in partition request
```

This is the key upstreaming gap — needs `AIE4_TOTAL_COLUMN` replaced with a module parameter.

---

### Path C: VE2 / NPU5 (`src/driver/amdxdna/ve2_of.c` + `ve2_hwctx.c`)

Used by: Strix Halo (Ryzen AI MAX 395, PCI `1022:17f0` rev `0x11`)

**Module parameters:**
```c
// In ve2_hwctx.c:
int max_col;
module_param(max_col, int, 0644);
MODULE_PARM_DESC(max_col, "Max column supported by this driver");

int start_col;
module_param(start_col, int, 0644);
MODULE_PARM_DESC(start_col, "Test only option: lead column set to start_col");

int partition_size;
int ve2_hwctx_limit;
```

**Column assignment (in ve2_of.c):**
```c
ret = aie_get_device_info(&xdna_hdl->aie_dev_info);
XDNA_INFO(xdna, "AIE device: %d columns, %d rows",
          xdna_hdl->aie_dev_info.cols, xdna_hdl->aie_dev_info.rows);

if (max_col > 0 && start_col >= 0 &&
    (max_col + start_col) <= xdna_hdl->aie_dev_info.cols) {
    xrs_cfg.total_col = max_col;
    XDNA_INFO(xdna, "Using module parameter: max_col=%d, start_col=%d",
              max_col, start_col);
} else {
    xrs_cfg.total_col = xdna_hdl->aie_dev_info.cols;
}
```

The VE2 path already supports override! The bottleneck is `aie_dev_info.cols` — which comes from `aie_get_device_info()`. This queries the Xilinx AI Engine driver/device tree for the hardware columns. If the firmware/device tree reports fewer columns than physically present, that's the hard limit.

## Key Source Files

| File | Purpose |
|------|---------|
| `src/driver/amdxdna/aie2_pci.c` | AIE2 driver (NPU1/NPU4) — has `aie2_max_col` param |
| `src/driver/amdxdna/aie4_pci.c` | AIE4 driver (NPU3/NPU4 classic) — has `aie4_max_col` param |
| `drivers/accel/amdxdna/aie4_pci.c` | Upstream AIE4 — **hardcoded to 3** ⚠️ |
| `drivers/accel/amdxdna/aie2_pci.c` | Upstream AIE2 — has `aie2_max_col` param |
| `src/driver/amdxdna/ve2_of.c` | VE2 init — reads device info, applies `max_col`/`start_col` |
| `src/driver/amdxdna/ve2_hwctx.c` | VE2 hwctx — defines `max_col`/`start_col` module params |
| `src/driver/amdxdna/ve2_mgmt.c` | VE2 management — partition creation, scheduling, handshake |
| `src/driver/amdxdna/ve2_mgmt.h` | VE2 header — address layout, handshake struct |
| `drivers/accel/amdxdna/aie4_pci.h` | AIE4 header — struct amdxdna_dev_hdl with `total_col` field |
| `src/driver/amdxdna/aie2_pci.h` | AIE2 header (same) |
| `src/driver/amdxdna/aie4_solver.c` | AIE4 resource solver — uses `total_col` |
| `src/driver/amdxdna/aie2_solver.c` | AIE2 resource solver |
| `src/driver/amdxdna/ve2_res_solver.c` | VE2 resource solver |
| `drivers/accel/amdxdna/amdxdna_sensors.c` | Sensor query — caps column util reporting at 8 cols |

## Key Commits

| SHA | Date | Message | Relevance |
|-----|------|---------|-----------|
| `584b8df` | Dec 2025 | "Removed VE2_MAX_COL and XRS_MAX_COL macro (#890)" | Removed hardcoded upper limits |
| `da55975` | Nov 2025 | "Adding start col support for VE2 (#856)" | Added `start_col` feature |
| `cb44600` | Feb 2026 | "Modified ve2_get_total_col function to fetch rows/cols (#1106)" | Dynamic column query |
| `27516b2` | Mar 2026 | "accel/amdxdna: add per-context DPM level reference counting" | Added `col_opc` per-platform (2048 NPU1, 4096 NPU4-6) |
| PR #1458 | Jun 2026 | "fix: reorder NPU init — PSP before SMU on Strix Halo" | **Critical fix** for Strix Halo boot |
| `211f213` | Jul 11 2026 | "Allocating cacheable buffers when cacheable flag is set" | Latest `main` HEAD |
| PR #1486 | Jul 10 2026 | "AIE4 firmware logging and event tracing (DPT)" | Open PR, mentions "40/40 on Strix" |
| PR #1485 | Jul 10 2026 | "aie4 async error support" | Open PR |

## How to Override Columns (All Methods)

### Method 1: Module Parameter (VE2/NPU5 — Strix Halo)
```bash
sudo modprobe -r amdxdna
sudo modprobe amdxdna max_col=40
```
Or via `/etc/modprobe.d/amdxdna.conf`:
```
options amdxdna max_col=40 start_col=0
```

### Method 2: Module Parameter (AIE2 — Phoenix/Strix Point)
```bash
sudo modprobe amdxdna aie2_max_col=40
```

### Method 3: Module Parameter (AIE4 — Hawk Point)
```bash
sudo modprobe amdxdna aie4_max_col=40
```

### Method 4: Patch Driver Source
Edit `drivers/accel/amdxdna/aie4_pci.c`:
```c
// Change:
#define AIE4_TOTAL_COLUMN 3
// To:
#define AIE4_TOTAL_COLUMN 40
```
Or better, replace with a module param like the out-of-tree version does.

## Checking Current Column Count

```bash
# Via sysfs (if exposed)
cat /sys/class/accel/accel*/device/columns 2>/dev/null || echo "not exposed"

# Via xrt-smi (if XRT installed)
sudo xrt-smi examine

# Via flm (FastFlowLM)
sudo flm validate

# Via dmesg (look for AIE device info)
dmesg | grep -i "AIE device"
```

## The Real Bottlenecks

1. **Firmware (`npu.sbin`)** — loaded from `/lib/firmware/amdnpu/XXXX_XX/npu_*.sbin`. The firmware reports `metadata.cols` via the mailbox protocol. If it reports fewer cols than physically present, the driver respects it. Custom firmware or a firmware hook is needed.

2. **Partition request** — in `aie4_partition_init()` partition is created with `AIE4_TOTAL_COLUMN` columns. If you increase the total column count but don't also increase the partition size, it'll still only use 3.

3. **AIE Device Info** — for VE2, `aie_get_device_info()` queries the Xilinx AIE driver/platform. If the device tree or AIE driver reports limited columns, that gates everything.

4. **Power/thermal** — more columns = more power. The SMU/DPM firmware may refuse to power on extra columns if they exceed the thermal budget.

5. **Sensor reporting** — `amdxdna_sensors.c` caps per-column utilization reporting at 8 columns (`min_t(u32, total_col, 8)`), but this is cosmetic, not functional.

---

## Secure Boot / MOK / Module Signing — THE ACTIVE BLOCKER (2026-07-13)

This section is the live log. The driver-architecture stuff above is solved; **this is what is actually preventing the 40-column module from loading.** Read this before doing anything else so we stop repeating the same loop.

### Status snapshot (verified this session)

| Check | Value | Source |
|-------|-------|--------|
| Secure Boot | **ENABLED** | `mokutil --sb-state` |
| Kernel lockdown | **Active** (from EFI Secure Boot) | `dmesg`: "Kernel is locked down from EFI Secure Boot mode" |
| `amdxdna` loaded? | **NO** — no `/sys/class/accel/*`, not in `lsmod` | `ls /sys/class/accel/` empty |
| `modprobe amdxdna` result | **`Key was rejected by service`** | live attempt |
| Our MOK enrolled? | **NO** | `mokutil --test-key MOK.der` → "not enrolled" |
| Enrolled keys | **1** = Canonical Ltd. Master CA (the Ubuntu distro key) | `mokutil --list-enrolled` |
| MOK queue (`--list-new` / `--list-delete`) | **EMPTY** | nothing staged |

### The two `amdxdna.ko` per kernel (why modprobe fails even though a valid signed module exists)

```
/lib/modules/7.0.0-27-generic/
  kernel/drivers/accel/amdxdna/amdxdna.ko.zst   ← in-tree, signed "Build time autogenerated kernel key" (VALID, would load)
  updates/dkms/amdxdna.ko.zst                     ← DKMS override, signed "strixhalo Secure Boot Module Signature key" (REJECTED)
```
`modprobe` prefers `updates/dkms/` (higher priority), so it tries our patched module, fails the key check, and aborts — it does **not** fall back to the in-tree one.

DKMS state (both kernels):
```
amdxdna/7.0.0-rc1+git20260310.6b13cb8f4, 7.0.0-22-generic: installed
amdxdna/7.0.0-rc1+git20260310.6b13cb8f4, 7.0.0-27-generic: installed   ← running
```
DKMS signing framework (`/etc/dkms/framework.conf`) is **empty** → DKMS signs with the default pair `/var/lib/shim-signed/mok/MOK.{priv,der}`.

modprobe.d override (`/etc/modprobe.d/amdxdna.conf`):
```
options amdxdna aie2_max_col=40 fw_patches_enable=1
```

### What we tried (the loop we kept repeating)

```
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der
sudo mokutil --list-new        ← key appears staged
sudo reboot
  → blue MOK Manager screen
  → enter enrollment password  ← ACCEPTED ("I did enroll it")
  → MOK Manager: "reboot now?"
  → pick reboot
  → HANG / LOCKUP right here
  → hard power-cycle to recover
  → key never persisted (MOK-list-RT is volatile; commit to MOK-list happens during the same reset cycle that hung)
mokutil --test-key MOK.der → STILL "not enrolled"
→ repeat
```
Also attempted: `sudo mokutil --reset` (twice) — clears the staging queue but does not fix the underlying issue.

### Three definitive findings

**Finding 1 — The MOK keypair is out of sync (most likely root cause of the circle).**

```
/var/lib/shim-signed/mok/MOK.der   2026-07-08 00:05  927 bytes   (public cert, OLD)
/var/lib/shim-signed/mok/MOK.priv   2026-07-13 12:38 1675 bytes  (private key, REGENERATED TODAY)
/lib/modules/.../updates/dkms/amdxdna.ko.zst   built/signed 2026-07-13 13:18 (AFTER the priv regen)
```
The cert (`MOK.der`) and the private key (`MOK.priv`) were **not generated together** — the `.priv` was overwritten 5 days after the `.der`. Because `/etc/dkms/framework.conf` is empty, DKMS signed the module with `MOK.priv` + `MOK.der`, i.e. a **mismatched pair**. Consequences:
- The module's signature will not validate against `MOK.der`'s public key.
- Therefore **even a successful enrollment of `MOK.der` would still leave the module rejected.**
- This is why every lap of the loop ends the same way regardless of enrollment.

Cert subject matches (so it *looks* right, which is why it fooled us):
```
MOK.der subject = CN=strixhalo Secure Boot Module Signature key
module signer   = strixhalo Secure Boot Module Signature key
```
CN is a chosen string, not derived from the key — matching CN ≠ matching key.

**To confirm:** regenerate a clean pair together and compare the cert's Subject Key Identifier against the module's `sig_key` (`47:52:80:9B:07:9E:6A:90:E7:FA:17:C1:45:9A:71:B4:FA:42:CC:CA`). Current `MOK.der` SKID is `C1:93:55:C1:A7:D8:23:00:61:A7:81:27:A3:44:2F:8F:89:8F:9D:47`, which does not match the module's `sig_key` — already strong evidence of the mismatch.

> **✅ FIXED 2026-07-13 PM:** regenerated a clean matching pair (`openssl req -x509 -newkey rsa:2048 -outform DER`). Pubkey MD5 matches (priv == cert == `277f1aa1…`). New cert SKID `3A:FD:A3:B8:8C:79:27:C8:1C:3A:33:3B:F9:E5:A9:F6:6A:DB:16:C8`. Module rebuilt + re-signed via `dkms install --force`; modinfo signer = `strixhalo Secure Boot Module Signature key`. **Keypair mismatch eliminated** — remaining blocker is purely Findings 2/3 (enrollment persistence / hang).

**Finding 2 — Enrollment never actually persisted.**
`mokutil --test-key` says our key is "not enrolled"; the only enrolled key is Canonical's distro key. Every blue-screen enrollment that *appeared* to succeed was rolled back because the reboot that MOK Manager triggers to commit `MOK-list-RT` → `MOK-list` hung (Finding 3), and the hard reset needed to recover discards the volatile staging list.

**Finding 3 — The lockup point is the MOK-Manager-triggered reboot, not the NPU.**
User account: enrolled, password accepted, "reboot" → instant hang. `efibootmgr` is healthy afterward and Ubuntu boots fine, so this is **not** permanent NVRAM corruption and **not** a boot-time NPU wedge (the box recovers cleanly). It is specifically the warm-reset that MOK Manager issues after committing the key.

**Hardware correction:** the board is a **Bosgame BeyondMax AXB35-02** (a.k.a. **Bosgame M5**), AMI Aptio 5, BIOS **1.07** (2025-09-12) — *not* GMKtec. On this board the UEFI `ResetSystem()` warm-reset path appears to hang when issued from inside the EFI MOK Manager app (a known class of AMD-SoC firmware issue). **Evidence:** `journalctl --list-boots` shows boot `-1` ending 13:56:10 and the next boot not starting until 14:29:21 — a **33-minute gap** = hung at the MOK commit reboot until a cold power-cycle. That cold cycle is exactly what discards the staged enrollment.

### Decoupling the two risks before doing anything else

We have been conflating two independent problems. Separate them:

1. **Signing risk** — can we get our key trusted under Secure Boot? (Findings 1–3)
2. **Firmware-patch risk** — when `fw_patches_enable=1 aie2_max_col=40` actually loads, does it wedge the NPU PSP / hang the box? **Completely untested.**

Do NOT attempt MOK enrollment again until both are isolated, otherwise a hang is un-diagnosable (is it the signing? the reset? the fw patch?).

### Immediate next step (diagnostic, no enrollment, no BIOS change)

Load the **distro-signed in-tree** module to prove the NPU/SoC is healthy under Secure Boot with the stock driver:

```bash
# Move the rejected DKMS override aside so modprobe falls through to the in-tree one
sudo mv /lib/modules/$(uname -r)/updates/dkms/amdxdna.ko.zst /root/amdxdna.ko.zst.dkmsoff
sudo depmod -a
sudo modprobe amdxdna
ls /sys/class/accel/                # expect accel0
sudo dmesg | grep -iE 'AIE device|columns|amdxdna|psp' | tail
```
Expected: `accel0` appears, dmesg shows AIE device info with the **stock** column count (8 for NPU5). This confirms the NPU boots fine with a trusted signature and isolates the patched-module behavior as a separate variable.

### Path to actually run the 40-column patch (pick one)

**Path A — Disable Secure Boot (pragmatic, recommended to make progress today).**
Boot into the Bosgame BIOS (DEL at POST) → disable Secure Boot → boot. Kernel lockdown lifts, the self-signed DKMS module loads regardless of MOK state. Then test `aie2_max_col=40 fw_patches_enable=1`. Once the column unlock is **verified to actually work and not wedge the NPU**, decide whether to re-enable SB + fix MOK. This removes the entire signing side-quest so we can find out if the firmware patch itself is viable — the real unknown.

**Path B — Fix MOK properly (only if SB must stay on).**
1. Regenerate a **matching** pair together:
   ```bash
   sudo openssl req -new -x509 -newkey rsa:2048 -keyout /var/lib/shim-signed/mok/MOK.priv \
     -out /var/lib/shim-signed/mok/MOK.der -nodes -days 36500 \
     -subj '/CN=strixhalo Secure Boot Module Signature key/'
   sudo chmod 600 /var/lib/shim-signed/mok/MOK.priv
   ```
2. Confirm the new cert's Subject Key Identifier equals the module `sig_key` after re-signing.
3. Re-sign + rebuild the DKMS module with the matching pair:
   ```bash
   sudo dkms remove amdxdna/7.0.0-rc1+git20260310.6b13cb8f4 --all
   sudo dkms install amdxdna/7.0.0-rc1+git20260310.6b13cb8f4 -k $(uname -r)
   ```
4. Enroll `MOK.der`. **Before** the MOK-triggered reboot hang bites: flash the **Bosgame BIOS 1.09** update (manufacturer-supplied; see session log below — may fix the `ResetSystem` hang), then verify persistence with `mokutil --test-key`.

**Do not repeat the bare `mokutil --import → reboot` loop** — it has failed the same way ~5 times and cannot succeed while Finding 1 stands.

### Key files / locations

| Path | Purpose |
|------|---------|
| `/var/lib/shim-signed/mok/MOK.der` / `MOK.priv` | MOK keypair (REGENERATED clean 2026-07-13 PM; `.broken-20260713` backups kept) |
| `/etc/modprobe.d/amdxdna.conf` | `aie2_max_col=40 fw_patches_enable=1` |
| `/etc/dkms/framework.conf` | empty → uses default MOK path |
| `/lib/modules/<k>/updates/dkms/amdxdna.ko.zst` | patched module, REJECTED under SB |
| `/lib/modules/<k>/kernel/drivers/accel/amdxdna/amdxdna.ko.zst` | in-tree, distro-signed, VALID |
| `/var/lib/dkms/amdxdna/original_module/` | DKMS backup of original |
| `/home/bcloud/xdna-driver`, `/home/bcloud/amdxdna-dkms` | build trees |
| `/lib/firmware/amdnpu/17f0_11/npu.dev.sbin` | VE2/Strix-Halo firmware blob (in-tree loads `npu_7.sbin` symlink → `npu.sbin.1.1.2.65.zst`) |

### Session log — 2026-07-13 (PM)

**NPU health CONFIRMED (signing risk isolated).** Moved the rejected DKMS override aside (`/root/amdxdna.ko.zst.dkmsoff`), `depmod -a`, `modprobe amdxdna` → in-tree distro-signed module loaded under Secure Boot: `accel0` at `0000:c6:00.1`, firmware `amdnpu/17f0_11/npu_7.sbin`, driver v0.7.0 (AIE2 path). Key lines:
```
[ 773.474171] amdxdna: unknown parameter 'fw_patches_enable' ignored   ← confirms fw_patches_enable is DKMS-only
[ 773.476088] amdxdna 0000:c6:00.1: [drm] Load firmware amdnpu/17f0_11/npu_7.sbin
[ 773.586164] [drm] Initialized amdxdna_accel_driver 0.7.0 for 0000:c6:00.1
```
→ NPU/PSP/SoC is healthy under SB with a trusted signature. **Signing risk and NPU-wedge risk are now decoupled.**

**Two independent unlock approaches identified (both in flight):**

| # | Approach | Mechanism | Status |
|---|----------|-----------|--------|
| 1 | **CBS firmware menu** | AMD CBS → NBIO Common Options → XDNA Configuration → Active AIE Columns. Reached via **Smokeless UMAF** (custom form browser) OR **SmokelessRuntimeEFIPatcher** `FastPatch` on stock `SetupUtilityApp` (cfg `SREP_Config_Framework.cfg` flips `xor al,al`→`mov al,1` on the suppress check `32C0488B5C2408488B7C2410C3`). Sets the column count at the firmware/device-tree level = the *real* fix. | Patch run 11:26 with **default** cfg (SuppressIFPatcher+Loader), outcome unconfirmed (log saved `~/SREP.log.20260713-1126`). Framework cfg not yet run. |
| 2 | **DKMS `fw_patches_enable`** | Patched out-of-tree driver rewrites the fw blob at load. | Module built+signed; blocked on MOK enrollment (Findings 2/3). |

Note: no Bosgame BIOS release (0.x–1.06, per `Release_Note.txt`) ever mentions NPU/XDNA/AIE — so the official CBS unlock isn't in *any* BIOS; it always needs UMAF/RuntimeEFIPatcher. **1.09 is for general fixes + the likely MOK-hang fix + a clean baseline before patching** (patcher byte-patterns are BIOS-version-specific, so patch the *final* BIOS).

**Keypair fixed, module re-signed (see Finding 1 ✅).** `dkms install --force` rebuilt for 7.0.0-27-generic; module is back at `updates/dkms/amdxdna.ko.zst`, correctly signed.

**BIOS 1.09 obtained from Bosgame; flash USB prepped.**
- Package: `Downloads/AXB35-02_BOSGAME_SW1.09_PCBV2.0X_20260508.zip` (ROM `AXB3502109.bin`, 32 MiB, sha256 `b5e99129…`).
- Manufacturer EFI command: `AfuEfix64.efi ../ROM/AXB3502109.bin /p /b /n /r /k /l /x /capsule /q`.
- USB (SanDisk Cruzer, `/dev/sda`) loaded with `Shell/`, `ROM/`, root `.bin`, + `FLASH_INSTRUCTIONS.txt`. `AfuEfix64.efi` is **unsigned → needs Secure Boot OFF** (built-in shell or built-in Instant Flash).

**Ordered plan going forward:**
1. **Flash Bosgame BIOS 1.09** (DEL → built-in Instant Flash preferred, keeps SB on; else AfuEfi with SB off). First boot: Load Optimized Defaults.
2. Re-test the **MOK enrollment** on 1.09 — if the `ResetSystem` hang is fixed, `mokutil --import MOK.der` → blue screen → enroll → `mokutil --test-key` = enrolled. Then DKMS module loads under SB.
3. **NPU unlock:** re-run the RuntimeEFIPatcher on 1.09 (may need an updated `SREP_Config` if the byte pattern shifted) → set Active AIE Columns in CBS → verify `aie_dev_info.cols` returns 40. Falls back to the DKMS `fw_patches` path if the CBS route is blocked.

**Backups / artifacts created this session:** `MOK.der.broken-20260713`, `MOK.priv.broken-20260713`, `~/SREP.log.20260713-1126`, `Downloads/npu-unlock-usb/SREP.log.20260713-1126`, flash USB payload.

### Session log — 2026-07-13 (post-1.09 boot, ~17:18)

**BIOS 1.09 FLASH CONFIRMED.** `dmidecode -t bios`: Vendor AMI, **Version 1.09**, Release Date **05/08/2026** — matches the Bosgame `AXB3502109.bin` target (was 1.07). Boot clean: 3-min uptime at first check, 0 failed systemd units, only the 2 harmless KHO/gkr-pam warnings.

**NPU HEALTH RE-CONFIRMED ON 1.09.** In-tree distro-signed module loaded under SB: `accel0` @ `0000:c6:00.1`, FW `1.1.2.65` (`npu_7.sbin`), driver 0.7.0, xrt-smi sees `RyzenAI-npu5`. `flm validate` → **8 columns (stock, of 40)**. Δ = 32 cols locked ≈ 5× TOPS headroom.

**Signing mismatch RE-EMERGED (Finding 1 is back, new form).** Keypair `MOK.{der,priv}` are mutually consistent (both mtime 14:31; `MOK.der` SKID `3A:FD:A3:B8:8C:79:27:C8:1C:3A:33:3B:F9:E5:A9:F6:6A:DB:16:C8`, = the "✅ FIXED" cert from earlier). BUT the currently-installed DKMS module `sig_key` = `1E:1B:E0:65:33:F9:84:BA:B2:FC:24:66:FC:7A:4D:E2:00:38:86:D6` — a **third** key, neither the old broken cert (`C1:93:55:C1…`) nor the fixed one. The `dkms install --force` at 14:32 did **NOT** sign with the fixed `MOK.priv`. → **Even a successful enrollment of `MOK.der` would still reject this module.** MOK path is currently a guaranteed dead end; do not pursue it until the module is re-signed with the matching key AND `sig_key` == `MOK.der` SKID is verified.

**DECISION: Path A (disable Secure Boot) is the unambiguous next step.** Because (a) the MOK path is dead until re-sign + verify, and (b) the firmware-patch wedge risk (`fw_patches_enable=1 aie2_max_col=40`) is STILL UNTESTED and must be isolated before spending any reboot on signing. Disabling SB lifts lockdown so the self-signed DKMS module loads regardless of which key signed it.

**Armed for the test (safe).** DKMS module restored to `/lib/modules/.../updates/dkms/amdxdna.ko.zst`; `/etc/modprobe.d/amdxdna.conf` still set to `aie2_max_col=40 fw_patches_enable=1`; `softdep amdxdna pre: amdgpu` in place. Confirmed **nothing auto-loads amdxdna at boot** (`/etc/modules-load.d/amdxdna.conf` = "removed — must be loaded manually") → the untested fw-patch can only load on an explicit `modprobe`, so it cannot wedge the boot path, only the deliberate test.

**Next-action sequence (handed to user):**
1. Reboot → DEL → `Secure Boot` = `Disabled` (AMI Aptio 5: may need `Secure Boot Mode` = `Custom` first) → F10. Lockdown lifts.
2. `sudo dmesg -w` in one pane; `sudo modprobe amdxdna` in another.
3. `flm validate` → expect **40 columns** (unlock works) or NPU/PSP hang (cold-cycle; pivot to CBS/UMAF SREP approach).
4. If 40 cols: real `flm`/`xrt-smi` inference sanity check to prove not soft-wedged.
5. **Re-secure LAST:** re-sign DKMS module with matching `MOK.priv` (fix `sig_key`==SKID), re-enable SB, enroll `MOK.der` on 1.09 (this also tests whether 1.09 fixed the Finding-3 `ResetSystem` hang).

**Baseline state now:** trusted in-tree module currently loaded (8-col). DKMS override back in slot, armed.

---

## Open Questions for Further Investigation

1. **What does `aie_dev_info.cols` actually return on Strix Halo?** Does the device tree/firmware report 40 or fewer? (Blocked: can't read until a trusted module loads — see "Immediate next step" above.)
2. **What happens if you pass `max_col=40` / `fw_patches_enable=1` on a Strix Halo that reports 40 cols?** Does the firmware accept a 40-column partition, or does `fw_patches_enable=1` wedge the PSP and hang the box? **UNTESTED** — this is the real risk and the reason to prefer Path A (disable SB) so it can be tested cleanly.
3. **Does `fw_patches_enable=1` even apply on VE2/NPU5?** The param name (`aie2_max_col`, `fw_patches_enable`) suggests an AIE2 path; Strix Halo is VE2/NPU5 which uses `max_col`/`start_col`. Confirm which module params this DKMS build actually honors before trusting the column count.
4. **Bosgame BIOS 1.09:** does it fix the EFI `ResetSystem()` warm-reset hang inside MOK Manager (Finding 3)? And does flashing it shift the `SetupUtilityApp` byte pattern that `SREP_Config_Framework.cfg` patches (`32C0488B5C2408488B7C2410C3`→`B001`), requiring a new patcher config?
5. **For the upstream kernel** — has anyone submitted a PR to replace `AIE4_TOTAL_COLUMN = 3` with a module param yet?
6. **Is there a way to bypass firmware column reporting** by patching the firmware blob or adding a kernel quirk?

### Session log — 2026-07-14 (evening, continuation of the 2026-07-13 PM session)

**Starting state:** patched DKMS module (`updates/dkms/amdxdna.ko.zst`) loaded, `aie2_max_col=40 fw_patches_enable=1` active per `/etc/modprobe.d/amdxdna.conf`, driver reporting `NPU: 40 total_col`. Secure Boot state unknown at session start.

#### Traced the EINVAL crash all the way to its actual root cause

Reproduced the "all-zero tokens, hangs" bug from `1bit-systems`' README — it does not hang. It crashes cleanly with `SIGABRT`:
```
DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22): Invalid argument
```
dmesg: `aie2_xrs_load: create context failed, ret -22` → `aie2_alloc_resource: Allocate AIE resource failed` → `aie2_hwctx_init: Alloc hw resource failed`.

Traced the call chain in `~/xdna-driver/drivers/accel/amdxdna/`:
- `aie2_alloc_resource()` (`aie2_ctx.c:597`) builds `xrs_req` from `hwctx->col_list`/`num_col`, calls `xrs_allocate_resource()`.
- `xrs_allocate_resource()` (`aie2_solver.c:310`) runs `sanity_check()` first — checks `cdop->ncols > xrs->cfg.total_col`. **`XRS_MAX_COL` is 128** (`aie2_solver.h:9`), not 8 — the software-side resource bitmap is not the limiter. A 40-column request passes this check fine.
- Actual failure happens one level deeper, in a real firmware mailbox call: `aie_send_mgmt_msg_wait: command opcode 0x2 failed, status 0x2000003`. Opcode `0x2` = `MSG_OP_CREATE_CONTEXT` (`aie2_msg_priv.h:10`). The request struct (`create_ctx_req`, built in `aie2_create_context()`, `aie2_message.c`) is constructed correctly — `req.num_col = hwctx->num_col` (a `__u8`, correctly set to 40).
- **Status `0x2000003` decodes exactly to `AIE2_STATUS_MGMT_ERT_NOAVAIL`** (`aie2_msg_priv.h`, the `AIE2_STATUS_MGMT_ERT_*` block starts at `0x2000001`). "Resource not available" — from the firmware's own management/scheduler layer, not the Linux driver.

**Conclusion: the driver and everything above it (compiler, xclbin, resource-solver software) is correct end-to-end. The closed firmware itself refuses to grant a partition wider than its own internal resource tables allow**, regardless of what `aie2_max_col`/`metadata.cols` claims. This is a firmware-binary-level constraint, not fixable via driver source changes.

Useful firmware strings pulled from `npu.dev.sbin` (`strings -n 6`, matched against the error path) — likely markers of the actual resource table / validation logic:
```
Invalid column count: %u >= %u
[MGT-ERT]: Column index out of range, Col: %u (Num Columns: %u)
Max cols used for partition %d = %d
Invalid Start Column/Numcols
assign_aie2_columns:: Failed to map TLB for col %d
```
Not yet correlated to a specific firmware address — next step for whoever continues this is finding which of these strings' code path is actually hit during a 40-col `MSG_OP_CREATE_CONTEXT`, the same disassembly workflow already proven in `data/PATCH_README.md` (the serialization-gate patch).

#### Built a real 40-column xclbin and hardware-tested it — disproved the earlier hypothesis

Adapted AMD's proven `n2_core_placed.py` bf16 template (the one `STEP5-INT8-32TILE-PLAN.md` used for the 8-col/32-core work) to `n_aie_cols=40`. Found and fixed three latent 8-column assumptions: a `range(8)` loop only wiring 8 of 40 B-matrix fifos, a `core_tiles[row][0:8]` slice only reaching 8 of 40 compute tiles for the A-matrix broadcast, and `c_base_idx = group_idx * 8` (should scale with `n_aie_cols`). Also found the GEMM shape needs `N` divisible by `n * n_aie_cols` or `num_col_tile` underflows to zero via integer division and the task-generation loop silently does nothing (empty instruction file, no error).

Compiled clean through the patched NPU2-40 toolchain (`~/mlir-aie/npu2_40_toolchain/`): **`final_40col_v2.xclbin`, 421,440 bytes, real 10,976-byte instruction stream** — genuinely further than the compiler-design doc's own "Next Steps" ever reached. Two "loop count overflow" backend warnings persisted through compilation without blocking it — flagged as a possible correctness risk, not investigated further.

Rebuilt the test harness (`test_mt_gemm3.cpp`'s approach — its dependency headers, `helper.h`/`gemm_atb_layout.h`, had been deleted from the original `torch2aie/examples/...` location but intact copies existed in `~/mlir-aie/programming_examples/ml/block_datatypes/`).

**Ran it on real hardware: identical `EINVAL`/`NOAVAIL` failure as the original bug, even with a properly-built 40-column xclbin.** This disproves the working hypothesis from earlier tonight (that the crash was 8-column-built xclbins colliding with a 40-column-configured driver) — the blocker is confirmed to be the firmware resource-allocation layer itself, independent of how the xclbin was compiled. No hang, box stayed fully responsive both times.

Artifacts preserved: `~/npu-sandbox/npu-infer/bf16_kernel_dev/col40_gemm/` (fixed template, xclbin, insts, test harness).

#### Discovered BOTH module paths had been tampered, not just DKMS

After being told about two prior kernel panics, tried the documented "load the safe in-tree module" recovery (`mv` the DKMS override aside, `modprobe amdxdna`). **The "in-tree" module was not actually pristine either** — same `NPU UNLOCK: overriding metadata.cols from 8 to 40` message fired from it too, and `dpkg -S` confirmed the file at `kernel/drivers/accel/amdxdna/amdxdna.ko.zst` (mtime Jul 8) was not the one the installed package actually shipped — it had been manually replaced at some point, outside package management. **Do not trust "in-tree vs DKMS" as a safety distinction on this box going forward — check dpkg provenance, not just the load path.**

#### Successful, verified restoration to stock — the real safe baseline, remotely achievable

`sudo apt-get install --reinstall linux-modules-7.0.0-27-generic` restores the genuine, package-verified `amdxdna.ko.zst` (confirmed via `dpkg -L`, mtime Jun 18 — predates all tampering). Reload result:
- Driver version **0.7.0** (patched ones were 0.15.0)
- Firmware loaded: **`npu_7.sbin`** (stock), not `npu.dev.sbin` (dev/patched)
- **No "NPU UNLOCK" message at all**
- `xrt-smi examine` / `flm validate` confirm: **`NPU: /dev/accel/accel0 with 8 columns`**

This is now the proven, fast (~1 minute, no reboot needed), remotely-achievable path back to a known-good state — **no live USB required**, contrary to the earlier assumption that recovery needed physical console access. Use this if anything regresses: `sudo modprobe -r amdxdna && sudo apt-get install --reinstall linux-modules-$(uname -r) && sudo depmod -a && sudo modprobe amdxdna`.

**Current verified state (end of this session): Secure Boot enabled, stock driver 0.7.0, stock firmware, 8 columns, fully stable.**

#### Where this leaves the 40-column effort

Three layers, three different states:
1. **Compiler** (MLIR-AIE → xclbin targeting 40 cols): ✅ solved and hardware-tested tonight.
2. **Driver metadata reporting** (`aie2_max_col` override): ✅ solved (the DKMS/tampered-in-tree patch achieves this) — but this alone doesn't get you a working 40-col context.
3. **Firmware resource allocation**: ❌ the real, final, remaining blocker — `AIE2_STATUS_MGMT_ERT_NOAVAIL` inside the closed firmware, independent of anything upstream.

Actually unlocking this requires patching the firmware binary itself (finding and widening whatever internal table the firmware checks `num_col` against before granting a context) — the same class of work as the already-successful serialization-gate patch in `data/PATCH_README.md`, using the same proven `ps1p`/`ipu_disasm` tooling in `~/npu_re_workspace/`. Not started this session beyond a first-pass string search. This carries the same risk class that produced the two earlier kernel panics — the rollback path above makes recovery fast, but doesn't eliminate the risk of needing it again, or of something going wrong that the rollback can't cleanly fix (e.g. mid-flash corruption). Treat as a deliberate, focused effort with the JetKVM console attended, not a background task.

### Firmware disassembly pass — 2026-07-14, column-limit search (candidates only, not patched)

Searched `npu.dev.sbin` (copied to `data/npu_current_dev.sbin`) for code references to the column-limit error strings, using the same method that found the serialization gate (scan code section 0x220-0x1c000 for 16-bit halfwords matching the string's file offset).

**String offsets found:**
- `"Invalid column count: %u >= %u"` — file offset `0x1d6d1`
- `"Max cols used for partition %d = %d"` — file offset `0x1d520`
- `"[MGT-ERT]: Column index out of range, Col: %u (Num Columns: %u)"` — file offset `0x20bdd`
- `"Invalid Start Column/Numcols"` — file offset `0x4e495`

**Code-reference scan:** only `"Max cols used for partition"` (`0x1d520`) produced hits — its low-16-bit value (`0xd520`) appears at 4 code offsets: `0x00cc6`, `0x00cf4`, `0x01f02`, `0x07306`. The other three strings produced zero matches with this method — either referenced via a different addressing scheme (indexed/computed, not a direct immediate load) or not reached by this particular scan window.

**Disassembled ±0x20-0x40 bytes around each hit** using `tools/ipu_disasm_v2.py --start/--end`. All 4 windows contain `cmp`/`beq`/`bne` instructions in the general vicinity, but **none can be confidently identified as *the* column-count gate** — this is the same heuristic disassembler used for the serialization-gate patch, and its own opcode map (`data/isa_analysis.md`) explicitly marks multiple opcodes (`0xa`, `0xb`, `0xd`, `0xe`) as **"Unclear"**. Several of the decoded instructions in these windows fall into unclear-opcode territory, meaning the disassembly around these candidates is lower-confidence than the serialization-gate case was.

**Decision: did not patch.** Picking the wrong branch to NOP here isn't a clean failure like the driver-level EINVAL — it's undefined behavior in firmware running on real hardware, the same risk class that already produced two kernel panics tonight. Not worth guessing.

**What would actually raise confidence before attempting a patch:**
1. Cross-reference the "Unclear" opcodes against a second data point — e.g. compare against any published/leaked documentation for a related AIE2/VE2 firmware ISA, or diff this firmware against a different firmware version (if one exists) to see which bytes change between versions that are known to differ only in column-limit config.
2. Narrow the 4 candidates by checking which one is actually reachable from the `MSG_OP_CREATE_CONTEXT` handler specifically (trace forward from the mailbox dispatch table instead of backward from string references) — `0x00cc6`/`0x00cf4` are close together and near the start of the code section (plausibly early init/config code), `0x01f02` and `0x07306` are further in and may be more relevant to a runtime request-handling path. Worth checking call graph proximity to the message-dispatch entry point before picking one.
3. If available, a way to single-step or trace the firmware during an actual failing `MSG_OP_CREATE_CONTEXT` call (even coarse-grained, e.g. via any debug/trace registers the firmware exposes) would confirm which code path actually executes, rather than guessing from static analysis alone.

Candidates preserved here for whoever picks this up with more time/better ISA confidence.

---

## Appendix: 40-Column Unlock — Verified Working (2026-07-16)

On **2026-07-16**, the full 40-column driver-level unlock was achieved and verified.

### The Two Kernel Patches

Two source files were modified in the local xdna-driver checkout. These are the patches that make `amdxdna-40col.ko` work:

#### Patch 1: AIE2 Path — `drivers/accel/amdxdna/aie2_pci.c`

```c
/* NPU column unlock: if aie2_max_col > fw-reported cols, override metadata */
if (aie2_max_col > ndev->aie.metadata.cols) {
    XDNA_WARN(ndev->aie.xdna,
              "NPU UNLOCK: overriding metadata.cols from %d to %d",
              ndev->aie.metadata.cols, aie2_max_col);
    ndev->aie.metadata.cols = aie2_max_col;
}
ndev->total_col = min(aie2_max_col, ndev->aie.metadata.cols);
XDNA_WARN(ndev->aie.xdna, "NPU: %d total_col (aie2_max_col=%d, metadata.cols=%d)",
          ndev->total_col, aie2_max_col, ndev->aie.metadata.cols);
```

#### Patch 2: VE2 Path — `src/driver/amdxdna/ve2_of.c`

```c
/* NPU column unlock: override aie_dev_info cols if max_col is larger */
if (max_col > 0 && max_col > (int)xdna_hdl->aie_dev_info.cols) {
    XDNA_INFO(xdna, "NPU UNLOCK: overriding aie_dev_info.cols from %d to %d",
              xdna_hdl->aie_dev_info.cols, max_col);
    xdna_hdl->aie_dev_info.cols = (u32)max_col;
}
```

### Exact Loading Sequence

| Time | Action | Result |
|------|--------|--------|
| 09:27:41 | Boot with kernel params `aie2_max_col=40 fw_patches_enable=1` | In-tree 0.7.0 loads, 8 cols |
| 10:09:50 | `sudo modprobe -r amdxdna` | Unload in-tree module |
| 10:10:00 | `sudo insmod amdxdna-40col.ko aie2_max_col=40` | FAIL: unknown symbol `amd_pmf_get_npu_data` |
| 10:10:03 | `sudo modprobe gpu-sched && sudo modprobe amd-pmf` | Load dependencies |
| 10:10:03 | `sudo insmod amdxdna-40col.ko aie2_max_col=40` (2nd attempt) | **SUCCESS — 40 columns!** |
| 10:10:22 | HW context creation attempt | FW `AIE2_STATUS_MGMT_ERT_NOAVAIL` |
| 10:10:29 | `sudo modprobe -r amdxdna` | Unload custom module |
| 10:10:30 | `sudo modprobe amdxdna aie2_max_col=40` (in-tree) | Back to 8 cols (safe baseline) |

### Key Artifacts

| Artifact | Path |
|----------|------|
| Custom module | `/home/bcloud/amdxdna-40col.ko` (10.9 MB, v0.1) |
| Dev firmware | `/lib/firmware/amdnpu/17f0_11/npu.dev.sbin` (430 KB) |
| Patched src (AIE2) | `/home/bcloud/xdna-driver/drivers/accel/amdxdna/aie2_pci.c` |
| Patched src (VE2) | `/home/bcloud/xdna-driver/src/driver/amdxdna/ve2_of.c` |
| 40-col xclbin | `/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/col40_gemm/final_40col_v2.xclbin` |

### Current Blocker

Creating a 40-col HW context fails with firmware status `0x2000003` (`AIE2_STATUS_MGMT_ERT_NOAVAIL`). The firmware's internal resource allocation tables are still hardcoded for 8 columns. Fix requires patching the firmware binary `npu.dev.sbin` — same methodology as the serialization-gate patch (see `npu_re_workspace/data/PATCH_README.md`).

### Firmware RE Candidates

Strings from `npu.dev.sbin` pointing to the column-limit validation code:
- `"Invalid column count: %u >= %u"` — file offset `0x1d6d1`
- `"Max cols used for partition %d = %d"` — file offset `0x1d520` (hits at code offsets `0x00cc6`, `0x00cf4`, `0x01f02`, `0x07306`)
- `"[MGT-ERT]: Column index out of range, Col: %u (Num Columns: %u)"` — file offset `0x20bdd`
- `"Invalid Start Column/Numcols"` — file offset `0x4e495`

### Verbatim dmesg from the unlock

```
amdxdna 0000:c6:00.1: [drm] Load firmware amdnpu/17f0_11/npu.dev.sbin
amdxdna 0000:c6:00.1: [drm] aie2_mgmt_fw_query: NPU UNLOCK: overriding metadata.cols from 8 to 40
amdxdna 0000:c6:00.1: [drm] aie2_mgmt_fw_query: NPU: 40 total_col (aie2_max_col=40, metadata.cols=40)
[drm] Initialized amdxdna_accel_driver 0.15.0 for 0000:c6:00.1 on minor 0
```

**Important clarification (added 2026-07-25):** the "SUCCESS — 40 columns!" line above means the driver-level metadata override worked — it does not mean a real 40-column compute context ever ran. The very next row in the table (`HW context creation attempt` → `NOAVAIL`) shows the actual firmware gate rejecting real use immediately. This distinction matters: every later session that "re-achieves" the 40-column driver override is repeating this same, already-solved half of the problem, not making new progress on the actual blocker. See the next appendix for the full firmware-patch effort against that gate, none of which has succeeded yet.

---

## Appendix: Firmware Patch Attempts v5/v5a/v5b/v5c (2026-07-21 through 2026-07-25) — still blocked

Following on from the previous appendix's "Current Blocker," this section tracks the actual attempt to defeat the firmware-side column-count gate (`AIE2_STATUS_MGMT_ERT_NOAVAIL`) via binary patching, plus the driver-side rebuild work needed to keep testing it across a kernel upgrade.

### The firmware patches

On 2026-07-21, static analysis of `npu.dev.sbin` identified the actual gate: a comparison constant at file offset `0x31E0` (hardcoded `8`) and a conditional branch to an error path at `0x31E4`. Four patched variants were built (all 429680 bytes, distinct hashes):

| Variant | Patch applied | Result (as tested) |
|---|---|---|
| `npu_patched_40col_v5.sbin` | offset `0x31E0` only (8→0x28) | `fw return error 0x5` (`aie2_smu_exec: smu cmd 4 timed out`, `Power off failed, ret -110`) |
| `npu_patched_40col_v5a.sbin` | same patch, alternate build | `fw return error 0x5` (same signature; v5/v5a not individually distinguished in the surviving logs — the staged file from that test was overwritten before it could be hashed) |
| `npu_patched_40col_v5b.sbin` | offset `0x31E4` only (branch NOP'd) | `fw return error 0x63` (`failed to validate fw, ret -5`, `aie2_hw_start: failed to start psp`) — confirmed 2026-07-25 06:24 |
| `npu_patched_40col_v5c.sbin` | both changes combined | `fw return error 0x63` (same as v5b) — confirmed 2026-07-24 |

**All four fail PSP firmware validation before a context can even be attempted.** `0x5` and `0x63` are different PSP-side rejection codes — neither variant gets past firmware signature/integrity checking to reach the actual column-count logic. This is a **harder failure than the 2026-07-16 baseline** (`NOAVAIL`, which at least reached context creation) — the current patches trip PSP validation first.

**Hardware hazard confirmed twice**: loading a firmware image that fails PSP validation this way wedges the SMU, which on this Strix Halo box is shared between the NPU and the GPU — the GPU goes down too (`SMU: No response`, `Failed to disable gfxoff!`, eventual `GPU Recovery Failed`). Only a full cold power-cycle reliably clears it; a warm reboot has been observed to *not* clear the wedge. Treat any live firmware-load test on this class of patch as needing to be attended, with the user aware a hard power-cycle may be required afterward.

### Driver-side blocker (separate from the firmware issue, now fixed)

Rebuilding the cols-only out-of-tree driver (`unlock-cols-only-cleanctx` branch, `~/xdna-driver`) after the box moved from kernel `7.0.0-27` to `7.0.0-28` hit an unrelated blocker: `insmod` failed with `Unknown symbol amd_pmf_get_npu_data (err -2)`. Root cause: `drivers/accel/amdxdna/config_kernel.h` was auto-generated against `7.0.0-27` and never regenerated. Fix:

```
sh drivers/accel/tools/configure_kernel.sh    # regenerates config_kernel.h for the running kernel
cd ~/xdna-driver/drivers/accel/amdxdna && make -C /lib/modules/$(uname -r)/build M=$(pwd) \
    CFLAGS_MODULE="-DAMDXDNA_DEVEL" OFT_CONFIG_AMDXDNA_PCI=y OFT_CONFIG_AMDXDNA_OF=n modules
```

Note the explicit `OFT_CONFIG_AMDXDNA_PCI=y OFT_CONFIG_AMDXDNA_OF=n` — invoking `make` without them skips this repo's own `Kbuild` gating and silently drops the entire AIE2/AIE4/NPU-regs object group, producing a different-looking (but spurious) undefined-symbol storm. Validated live 2026-07-25 05:56: the rebuilt module loaded cleanly (no symbol error), reached the same driver-level "40 total_col" milestone as 2026-07-16, then immediately hit a fourth, distinct failure: `aie2_error_worker: Did not get error column` repeated ~32 times, then the boot went silent (no panic/oops logged) — not yet investigated (likely `aie2_ctx.c`/`aie2_error.c` async error-reporting sized for 8 columns).

**This driver-side rebuild is unrelated to the firmware PSP-validation problem above** — fixing it only restores the ability to reach the driver-level milestone on the current kernel; it does not touch why v5/v5a/v5b/v5c fail PSP validation.

### Kernel version is not the blocker

It might look like downgrading back to `7.0.0-27` (the kernel in use during the 2026-07-16 baseline) would help, since the rebuild friction above was triggered by the `.27`→`.28` upgrade. It would not: `NOAVAIL` (2026-07-16, on `.27`) and `0x5`/`0x63` (2026-07-24/25, on `.28`) are all firmware/PSP-side rejections, independent of the host kernel version. The `.27`/`.28` distinction only affects whether the out-of-tree **driver module** loads at all (vermagic/symbol matching); it has no bearing on what the **PSP** does with a given firmware image.

### Current state (2026-07-25 ~06:30)

- Both confirmed-bad firmware files renamed out of the driver's search path: `npu.dev.sbin.bad_v5b_0x63`, `npu.dev.sbin.bad_v5c_0x63` (under `/lib/firmware/amdnpu/17f0_11/`).
- System restored to stock: in-tree signed driver (`sig_id: PKCS#7`, matches package), stock firmware (`npu_7.sbin`, hash-verified identical to the packaged `npu.sbin.1.1.2.65.zst`), no active `aie2_max_col` override in `/etc/modprobe.d/` or GRUB.
- **Untested-fresh candidates**: none remain among v5/v5a/v5b/v5c — all four have now been live-tested and all fail PSP validation. Continuing this approach would mean either a different patch offset/strategy, or accepting the GPU-inference-only path (measured 371–441 tok/s) as the near-term outcome — an option raised during the 2026-07-24 GPU/SMU-wedge incident as a pragmatic fallback if the firmware patch route stalls.

---

## Appendix: PSP signature validation is the actual gate (2026-07-25) — work paused here

After v5/v5a/v5b/v5c all failed (see above), it's worth being explicit about *why*, because it changes what "continuing this research" means.

### Why byte-patching the column constant can't work as attempted

`npu.dev.sbin` is not loaded directly by the NPU — it's handed to the PSP (Platform Security Processor), a separate on-die ARM core that acts as AMD's hardware root of trust. Standard PSP firmware-loading flow validates the image against a signature chain anchored in on-die fused keys before anything in the image is allowed to execute. That signature covers the image as a whole (or large signed regions of it), not just the specific bytes any single patch touched.

This matches what was actually observed, not just architectural theory:
- The 2026-07-16 baseline patch (driver-side `aie2_max_col` override only, firmware untouched) reached real context creation and failed on `AIE2_STATUS_MGMT_ERT_NOAVAIL` — a firmware **logic** rejection, well past validation.
- Every firmware **binary** patch since (v5/v5a's `0x5`, v5b/v5c's `0x63`) fails earlier and differently — `fw return error`, `failed to validate fw, ret -5`, PSP refusing to even start the image. These are validation-stage codes, not the column-count logic the patches were targeting. Changing *where* in the file the patch lands (offset `0x31E0` vs `0x31E4` vs both) changed the specific rejection code but never got past this stage — consistent with any modification invalidating a signature check, regardless of which bytes changed.

### What "PSP signature unlock" would actually require

To get a patched image past this, one of two things would be needed:
1. **A legitimately re-signed image** — i.e., AMD's private signing key for this PSP firmware family, which is not available to us and has no known public leak for this hardware generation. Not something this project can obtain.
2. **An actual vulnerability in the PSP's signature-verification path** that lets an unsigned or modified image load anyway. This is a distinct, much larger security-research problem — PSP firmware exploitation — unrelated to the column-count patch this effort has been iterating on. It would mean auditing PSP's own boot/validation code (which is itself closed and only reachable via the very interface it's gatekeeping), not adjusting offsets in an NPU firmware blob. Publicly known PSP exploits (where they exist, for other AMD platform generations) came out of dedicated PSP-focused research, not incremental patching of a downstream firmware image.

Neither path is a natural continuation of the v5/v5a/v5b/v5c work — patching different offsets in `npu.dev.sbin` will keep tripping the same signature check no matter which byte moves. **This is why the column-unlock effort is paused as of 2026-07-25, not just low on untested variants.**

### Status

Secure Boot has been re-enabled on the test box (it was only disabled to permit loading the unsigned out-of-tree `amdxdna` driver module during testing — unrelated to the PSP signature question, but no longer needed with this line of attack closed). System verified back to a clean stock baseline: in-tree signed driver, stock firmware, no errors.

If this is picked up again, scope it explicitly as one of the two options above rather than another firmware-byte-patch variant — variant 6, 7, 8 of the same offset-guessing approach will hit the same wall as v5 through v5c.
