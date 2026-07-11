#!/bin/bash
# apply-and-build.sh — Patch the kernel amdgpu driver with NPU support and rebuild
#
# Run this OUTSIDE the sandbox (on the host directly) with:
#   sudo bash apply-and-build.sh
#
# What it does:
#   1. Extracts matching kernel source
#   2. Adds NPU source files (amdgpu_npu.c, .h, _mgr.c, _sched.c)
#   3. Patches amdgpu for Strix Halo NPU PCI IDs (0x17f0, 0x17f1, 0x17f2)
#   4. Adds AMD_IP_BLOCK_TYPE_NPU to the IP block enum
#   5. Registers NPU as an IP block in amdgpu_device.c
#   6. Adds NPU DRM IOCTL definitions
#   7. Rebuilds only the amdgpu module
#   8. Installs the new module
#   9. Rebinds the NPU device from amdxdna to amdgpu

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_VER="${KERNEL_VER:-$(uname -r)}"
BUILD_DIR="/lib/modules/${KERNEL_VER}/build"
OUTPUT_DIR="/tmp/amdgpu-npu-build"
KERNEL_SRC="${OUTPUT_DIR}/linux-source"
KERNEL_MAJOR="${KERNEL_VER%%%.*}"  # "7" from "7.0.0-27-generic"

NPU_PCI_IDS=( "0x17f0" "0x17f1" "0x17f2" )
NPU_PCI_DEV="${NPU_PCI_DEV:-0000:c6:00.1}"

echo "=============================================="
echo "  amdgpu NPU patch — build & install"
echo "  Kernel: ${KERNEL_VER}"
echo "  NPU device: ${NPU_PCI_DEV}"
echo "=============================================="

# ── Step 1: Get kernel source ───────────────────────────────────────
echo ""
echo "=== Step 1: Getting kernel source ==="
mkdir -p "${OUTPUT_DIR}"

SRC_TARBALL=$(ls /usr/src/linux-source-${KERNEL_MAJOR}*.tar.bz2 2>/dev/null | head -1)
EXTRACTED_SRC=$(ls -d /usr/src/linux-source-${KERNEL_MAJOR}*/ 2>/dev/null | grep -v "\.tar\." | head -1)

if [ -n "$EXTRACTED_SRC" ]; then
    echo "Using already-extracted kernel source at ${EXTRACTED_SRC} (symlink)"
    rm -rf "${KERNEL_SRC}"
    ln -sf "$EXTRACTED_SRC" "${KERNEL_SRC}"
elif [ -n "$SRC_TARBALL" ]; then
    echo "Extracting kernel source from ${SRC_TARBALL}..."
    rm -rf "${KERNEL_SRC}"
    tar -xf "$SRC_TARBALL" -C "${OUTPUT_DIR}"
    # Handle the source directory prefix (linux-source-*)
    SRC_PREFIX=$(ls "${OUTPUT_DIR}/" 2>/dev/null | head -1)
    if [ -d "${OUTPUT_DIR}/${SRC_PREFIX}" ]; then
        mv "${OUTPUT_DIR}/${SRC_PREFIX}" "${KERNEL_SRC}"
    fi
elif [ -d /lib/modules/${KERNEL_VER}/source ]; then
    echo "Using /lib/modules/${KERNEL_VER}/source"
    rm -rf "${KERNEL_SRC}"
    cp -a "/lib/modules/${KERNEL_VER}/source" "${KERNEL_SRC}"
else
    echo "ERROR: Cannot find kernel source."
    echo "Install with: sudo apt install linux-source-${KERNEL_VER%%.*}.0"
    exit 1
fi

# Copy .config and Module.symvers from the build tree
echo "Copying build config and symvers..."
cp "${BUILD_DIR}/.config" "${KERNEL_SRC}/"
cp "${BUILD_DIR}/Module.symvers" "${KERNEL_SRC}/" 2>/dev/null || true
ln -sf "${BUILD_DIR}/scripts" "${KERNEL_SRC}/scripts" 2>/dev/null || true

# ── Step 2: Copy NPU source files ────────────────────────────────────
echo ""
echo "=== Step 2: Copying NPU source files ==="
AMDGPU_DIR="${KERNEL_SRC}/drivers/gpu/drm/amd/amdgpu"
mkdir -p "${AMDGPU_DIR}"

for f in amdgpu_npu.c amdgpu_npu.h amdgpu_npu_mgr.c amdgpu_npu_sched.c; do
    if [ -f "${SCRIPT_DIR}/${f}" ]; then
        cp -v "${SCRIPT_DIR}/${f}" "${AMDGPU_DIR}/${f}"
    else
        echo "WARNING: ${f} not found in ${SCRIPT_DIR}"
    fi
done

# ── Step 3: Patch kernel files ───────────────────────────────────────
echo ""
echo "=== Step 3: Patching kernel source ==="

# Helper: add a line after a matching pattern if it's not already there
add_line_after() {
    local file="$1" pattern="$2" new_line="$3"
    if grep -qF "${new_line}" "${file}" 2>/dev/null; then
        echo "  ✓ already present: ${new_line}"
    else
        sed -i "/${pattern}/a\\${new_line}" "${file}"
        echo "  + added: ${new_line}"
    fi
}

# Helper: add a line before a matching pattern
add_line_before() {
    local file="$1" pattern="$2" new_line="$3"
    if grep -qF "${new_line}" "${file}" 2>/dev/null; then
        echo "  ✓ already present: ${new_line}"
    else
        sed -i "/${pattern}/i\\${new_line}" "${file}"
        echo "  + added: ${new_line}"
    fi
}

# 3a. Add NPU PCI IDs to amdgpu_drv.c
DRV="${AMDGPU_DIR}/amdgpu_drv.c"
echo "  Patching amdgpu_drv.c (PCI IDs)..."
for id in "${NPU_PCI_IDS[@]}"; do
    if grep -q "${id}" "${DRV}" 2>/dev/null; then
        echo "    ✓ PCI ID ${id} already present"
    else
        # Find the last {0, 0, 0} entry which terminates the list
        sed -i "/{0, 0, 0}/i\\\t{ PCI_VDEVICE(AMD, ${id}), CHIP_STRIX_HALO }," "${DRV}"
        echo "    + added PCI ID ${id}"
    fi
done

# 3b. Add NPU .o files to Makefile
MAK="${AMDGPU_DIR}/Makefile"
echo "  Patching Makefile..."
add_line_after "${MAK}" "amdgpu_ttm\\.o" "	amdgpu_npu.o \\\\"
add_line_after "${MAK}" "amdgpu_npu\\.o" "	amdgpu_npu_mgr.o \\\\"
add_line_after "${MAK}" "amdgpu_npu_mgr\\.o" "	amdgpu_npu_sched.o \\\\"

# 3c. Add npu field to amdgpu.h
AMDH="${AMDGPU_DIR}/amdgpu.h"
echo "  Patching amdgpu.h..."
if grep -q "struct amdgpu_npu.*npu" "${AMDH}" 2>/dev/null; then
    echo "    ✓ npu field already present"
else
    sed -i "/struct amdgpu_jpeg.*jpeg;/a\\\n\t/* NPU (XDNA) engine */\n\tstruct amdgpu_npu\t\t*npu;" "${AMDH}"
    echo "    + added npu field"
fi

# 3d. Add #include "amdgpu_npu.h" and npu_ip_block to amdgpu_device.c
DEVC="${AMDGPU_DIR}/amdgpu_device.c"
echo "  Patching amdgpu_device.c..."
add_line_after "${DEVC}" "#include.*amdgpu_psp\\.h" "#include \"amdgpu_npu.h\""
if grep -q "npu_ip_block" "${DEVC}" 2>/dev/null; then
    echo "    ✓ npu_ip_block registration already present"
else
    sed -i "/amdgpu_device_ip_block_add.*npu_ip_block/d" "${DEVC}" 2>/dev/null || true
    # Add after the first ip_block_add call or before discovery
    sed -i "/amdgpu_discovery_set_ip_blocks/i\\\tamdgpu_device_ip_block_add(adev, \&npu_ip_block);" "${DEVC}"
    echo "    + added npu_ip_block registration"
fi

# 3e. Add AMD_IP_BLOCK_TYPE_NPU to amdgpu_ip.h
IPH="${AMDGPU_DIR}/amdgpu_ip.h"
echo "  Patching amdgpu_ip.h..."
add_line_before "${IPH}" "AMD_IP_BLOCK_TYPE_NUM" "	AMD_IP_BLOCK_TYPE_NPU,"

# 3f. Add NPU DRM IOCTL definitions to amdgpu_drm.h
DRM="${KERNEL_SRC}/include/uapi/drm/amdgpu_drm.h"
echo "  Patching amdgpu_drm.h (NPU IOCTLs)..."
if grep -q "AMDGPU_NPU_CTX\|DRM_AMDGPU_NPU" "${DRM}" 2>/dev/null; then
    echo "    ✓ NPU IOCTLs already present"
else
    # Add DRM_AMDGPU_NPU_CTX and EXEC constants
    sed -i "/DRM_AMDGPU_CTX_UPDATE/a\\#define DRM_AMDGPU_NPU_CTX            0x20\n#define DRM_AMDGPU_NPU_EXEC           0x21" "${DRM}"
    # Add IOCTL definitions after FENCE_TO_HANDLE
    sed -i "/DRM_IOCTL_AMDGPU_FENCE_TO_HANDLE/a\\#define DRM_IOCTL_AMDGPU_NPU_CTX       DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_NPU_CTX, struct drm_amdgpu_npu_ctx)\n#define DRM_IOCTL_AMDGPU_NPU_EXEC      DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_NPU_EXEC, struct drm_amdgpu_npu_exec)" "${DRM}"
    echo "    + added NPU IOCTL definitions"
fi

echo ""
echo "=== Patches applied. ==="

# ── Step 4: Build amdgpu module ──────────────────────────────────────
echo ""
echo "=== Step 4: Building amdgpu module ==="
cd "${KERNEL_SRC}"

# Create necessary symlinks
ln -sf "${BUILD_DIR}/tools" "${KERNEL_SRC}/tools" 2>/dev/null || true

# Build the module
make -C "${KERNEL_SRC}" M=drivers/gpu/drm/amd/amdgpu modules -j$(nproc) 2>&1 | tail -20

echo ""
echo "=== Build exit code: ${PIPESTATUS[0]} ==="

if [ ! -f "${AMDGPU_DIR}/amdgpu.ko" ]; then
    echo "ERROR: Build failed — amdgpu.ko not produced."
    echo "Check build output above for errors."
    exit 1
fi

# ── Step 5: Install module ──────────────────────────────────────────
echo ""
echo "=== Step 5: Installing new amdgpu module ==="
make -C "${KERNEL_SRC}" M=drivers/gpu/drm/amd/amdgpu modules_install
depmod -a

echo ""
echo "=== Module installed ==="

# ── Step 6: Rebind NPU device ────────────────────────────────────────
echo ""
echo "=== Step 6: Rebinding NPU from amdxdna → amdgpu ==="

# Unbind from amdxdna
if [ -e "/sys/bus/pci/drivers/amdxdna/${NPU_PCI_DEV}" ]; then
    echo "  Unbinding ${NPU_PCI_DEV} from amdxdna..."
    echo "${NPU_PCI_DEV}" > /sys/bus/pci/drivers/amdxdna/unbind
    echo "  ✓ Unbound"
else
    echo "  NOTICE: ${NPU_PCI_DEV} not bound to amdxdna (or already unbound)"
fi

# Set driver_override so amdgpu picks it up
echo "  Setting driver_override for ${NPU_PCI_DEV}..."
echo "amdgpu" > /sys/bus/pci/devices/${NPU_PCI_DEV}/driver_override

# Bind to amdgpu
echo "  Binding ${NPU_PCI_DEV} to amdgpu..."
echo "${NPU_PCI_DEV}" > /sys/bus/pci/drivers/amdgpu/bind 2>/dev/null || \
    echo "  (bind may need a full module reload — see instructions below)"

# Reload amdgpu to pick up the new device
echo ""
echo "  Reloading amdgpu module..."
modprobe -r amdgpu || true
modprobe amdgpu

echo ""
echo "=============================================="
echo "  DONE"
echo "=============================================="
echo ""
echo "Verify with:"
echo "  ls /dev/accel*"
echo "  xrt-smi examine -r host"
echo "  lspci -s ${NPU_PCI_DEV} -vv | grep 'Kernel driver'"
echo ""
echo "If the device rebind fails, reboot and re-run this script."
