# DKMS NPU Driver Build & Load Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and load the out-of-tree DKMS `amdxdna.ko` module against the clean NPU firmware, enabling post-validation firmware patches without corrupting PSP state.

**Architecture:** The DKMS module at `/home/bcloud/amdxdna-dkms/` already has the correct design: PSP validates the stock firmware (valid RSA signature passes), then `aie2_apply_fw_patches()` zeros function pointer slots in the IPU firmware's scheduler dispatch table *in the DMA buffer* before `PSP_START` copies it to NPU SRAM. The key insight is that the pre-built `.ko` (built Jun 29 14:53) was loaded against **patched** firmware at that time — now the firmware is restored to clean stock, so the same `.ko` will pass PSP validation. We just need to load it correctly.

**Tech Stack:** Linux kernel module (DKMS), PCI driver for `17f0_11` NPU (XDNA2 NPU5), PSP firmware interface, SMU power management

## Global Constraints

- All source at `/home/bcloud/amdxdna-dkms/`
- Pre-built binary: `/home/bcloud/amdxdna-dkms/src/amdxdna/amdxdna.ko` (source timestamp Jun 29 14:53)
- Firmware files: `/lib/firmware/amdnpu/17f0_11/npu.sbin.1.1.2.65.zst` (clean restored backup)
- Kernel: `7.0.0-27-generic` on Strix Halo
- DKMS package name: `amdxdna-dkms`
- MUST NOT leave the NPU in an inconsistent state — always verify after load
- DO NOT patch the firmware binary file itself — patches happen in-memory via the kernel module

---

### Task 1: Verify pre-built module is compatible with current kernel

**Files:**
- Examine: `/home/bcloud/amdxdna-dkms/src/amdxdna/amdxdna.ko`
- Examine: `/lib/modules/7.0.0-27-generic/kernel/drivers/accel/amdxdna/amdxdna.ko.zst`

**Interfaces:**
- Consumes: The pre-built binary at `amdxdna-dkms/src/amdxdna/amdxdna.ko`
- Produces: Confirmation that the module can be loaded (or must be rebuilt)

- [ ] **Step 1: Check vermagic compatibility**

Run:
```bash
modinfo /home/bcloud/amdxdna-dkms/src/amdxdna/amdxdna.ko | grep vermagic
modinfo amdxdna 2>/dev/null | grep vermagic
```

The vermagic must match exactly. If they don't match, we need to rebuild.

- [ ] **Step 2: Check for strong module dependency conflicts**

Run:
```bash
modinfo /home/bcloud/amdxdna-dkms/src/amdxdna/amdxdna.ko | grep depends
```

The dependencies (`gpu-sched,amd-pmf`) must match the currently loaded versions.

- [ ] **Step 3: Verify the firmware files the module will use**

Run:
```bash
# Check what firmware the module requests
modinfo /home/bcloud/amdxdna-dkms/src/amdxdna/amdxdna.ko | grep firmware
# Confirm the files exist on disk
for fw in $(modinfo -F firmware /home/bcloud/amdxdna-dkms/src/amdxdna/amdxdna.ko 2>/dev/null); do
  [ -f "/lib/firmware/$fw" ] || [ -f "/lib/firmware/$fw.zst" ] || echo "MISSING: $fw"
done
```

Expected: All firmware files present. Critical: `amdnpu/17f0_11/npu_7.sbin.zst` resolved through symlink must point to clean restored firmware.

- [ ] **Step 4: Rebuild if vermagic/dependencies differ**

If vermagic differs from the in-tree kernel module, rebuild from source:

```bash
cd /home/bcloud/amdxdna-dkms/src/amdxdna
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
```

Expected: Clean build, no errors, new `amdxdna.ko` created.

---

### Task 2: Unload the in-tree amdxdna module and load the DKMS module

**Files:**
- Load: `/home/bcloud/amdxdna-dkms/src/amdxdna/amdxdna.ko`

**Interfaces:**
- Consumes: Task 1's verified `.ko`
- Produces: Running DKMS module with patched firmware on NPU

**Critical path:** The order of PSP operations is:
1. Driver loads firmware into DMA buffer (stock firmware with valid RSA sig)
2. `PSP_VALIDATE` — PSP checks RSA signature → PASSES (clean firmware)
3. `aie2_apply_fw_patches()` — zeroes 4 function pointers in DMA buffer
4. `PSP_START` with `PSP_START_COPY_FW` — copies DMA buffer (with patches) to NPU SRAM
5. SMU power on → mailbox init → firmware goes live

- [ ] **Step 1: Unload the in-tree module**

```bash
# Check for active users
lsmod | grep amdxdna
# Unload
sudo rmmod amdxdna
# Verify it's gone
lsmod | grep amdxdna || echo "Module unloaded successfully"
```

Expected: `amdxdna` no longer in `lsmod` output.

- [ ] **Step 2: Load the DKMS module**

```bash
# Load the out-of-tree module
sudo insmod /home/bcloud/amdxdna-dkms/src/amdxdna/amdxdna.ko
# Check if loaded
lsmod | grep amdxdna
```

Expected: Module loads without errors. `lsmod` shows `amdxdna` with `OE` (Out-of-tree, unsigned) flag.

- [ ] **Step 3: Verify no kernel errors**

```bash
# Check for any errors in kernel log
dmesg 2>/dev/null | grep -E "(amdxdna|PSP|npu)" | tail -10 || journalctl --no-pager -k 2>/dev/null | grep -E "(amdxdna|PSP|npu)" | tail -10
```

Expected: Should show "Applied 4 firmware patches after PSP validation" and NO "ERROR" lines. If any ERROR lines appear with `ret -110` or `PSP is not ready`, the firmware validation failed.

- [ ] **Step 4: Verify the NPU accel device appeared**

```bash
ls -la /dev/accel/
cat /sys/kernel/debug/dri/0/name 2>/dev/null || cat /sys/class/accel/accel0/device/device 2>/dev/null
```

Expected: `/dev/accel/accel0` exists. If it doesn't, the probe failed.

---

### Task 3: Verify NPU is functional and patches are applied

**Files:**
- Read-only: `/sys/kernel/debug/dri/0/amdxdna*` or similar debugfs entries
- Read: dmesg/journalctl for "Applied firmware patches" message

**Interfaces:**
- Consumes: Task 2's loaded module
- Produces: Confirmed working NPU with patched firmware

- [ ] **Step 1: Check for the patch confirmation message**

```bash
journalctl --no-pager -k 2>/dev/null | grep -i "firmware patch" | tail -5
```

Expected: Should see `Applied 4 firmware patches after PSP validation`.

- [ ] **Step 2: Check XRT/runtime can access the NPU**

```bash
# If XRT is installed
which xbutil 2>/dev/null && xbutil examine 2>/dev/null
# Or check the device
ls -la /dev/dri/render* 2>/dev/null
cat /sys/class/accel/accel0/device/vendor 2>/dev/null
cat /sys/class/accel/accel0/device/device 2>/dev/null
```

Expected: XRT device visible if installed, or accel device and render node present.

- [ ] **Step 3: Check no systemd services are restarting**

```bash
systemctl --user list-units --state=failed 2>/dev/null
systemctl --user is-active mlx-engine.service 2>/dev/null
systemctl --user is-active mlx-unified-proxy.service 2>/dev/null
```

Expected: No failed units, both mlx services `inactive` or `not-found`.

- [ ] **Step 4: Test basic NPU interaction with aiebu-dump**

```bash
aiebu-dump 2>&1 | head -5
```

Expected: Should not crash. May show help text or version info. If it crashes with SIGSEGV, the NPU hardware is still in bad state.

---

### Task 4: Make the DKMS module load persistent (optional)

**Files:**
- Consider: `/etc/modules-load.d/amdxdna.conf` (load at boot)
- Consider: `/etc/depmod.d/amdxdna.conf` (override built-in module)

**Interfaces:**
- Consumes: Task 2's working module
- Produces: Persistent DKMS module loading across reboots

- [ ] **Step 1: Create a modprobe config to prefer the DKMS module**

```bash
# Build a proper DKMS .deb package
cd /home/bcloud/amdxdna-dkms
dpkg-buildpackage -us -uc -b 2>&1 | tail -20
```

- [ ] **Step 2: Install the package**

```bash
sudo dpkg -i /home/bcloud/amdxdna-dkms_*.deb 2>&1
```

- [ ] **Step 3: Regenerate module dependencies**

```bash
sudo depmod -a
```

- [ ] **Step 4: Verify the installed module is the DKMS version**

```bash
modinfo amdxdna | grep -E "filename|srcversion"
```

Expected: `filename` should show `/lib/modules/.../updates/.../amdxdna.ko` and `srcversion` should match the DKMS build.
