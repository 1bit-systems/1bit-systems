#!/usr/bin/env bash
# build_aie.sh — scaffold wrapper for the BitNet-1.58 NPU xclbin build.
#
# STATUS (2026-04-24): SCAFFOLD ONLY. Do not run on CI. Running it locally
# assumes the ironenv venv is present + mlir-aie built + peano installed;
# see `docs/wiki/NPU-AIE2P.md` §"Install path for developers".
#
# The real BitNet-1.58 NPU kernel is a three-stage pipeline — unpack →
# stock mm.cc matmul → scale — authored via IRON + MLIR-AIE.  This script
# is the eventual `make`-replacement that wraps the upstream Makefile
# pattern from:
#     mlir-aie/programming_examples/basic/matrix_multiplication/
#         single_core/Makefile
#
# For now it documents the env + step sequence; fill in the TODO blocks
# when the kernel author picks it up (≤ 1 wk focused work per the effort
# breakdown in docs/wiki/Roadmap.md).

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Paths you almost certainly need to override for your box
# ---------------------------------------------------------------------------
: "${MLIR_AIE_ROOT:=/home/bcloud/repos/mlir-aie}"
: "${IRONENV:=/home/bcloud/.venvs/ironenv}"
: "${PEANO_ROOT:=/opt/peano}"
: "${XILINX_XRT:=/usr}"      # Arch convention — see project_npu_unlocked.md

# Default decode-shape params. Override on the command line if authoring a
# new variant:   env M=1024 K=1024 ./build_aie.sh
: "${M:=512}"
: "${K:=512}"
: "${N:=16}"                  # decode N=1 padded to 16; tile constraint
: "${m:=64}"
: "${k:=64}"
: "${n:=16}"
: "${devicename:=npu2}"       # Strix Halo AIE2P; IRON device_utils.py terminology

# Where we drop build artifacts. Matches npu-kernels/bitnet/README.md.
: "${BUILD_DIR:=$(dirname "$0")/../build/bitnet_gemv}"

# ---------------------------------------------------------------------------
# 1. Activate toolchain
# ---------------------------------------------------------------------------
if [[ ! -f "${IRONENV}/bin/activate" ]]; then
    echo "error: ironenv not found at ${IRONENV}; see docs/wiki/NPU-AIE2P.md" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"

if [[ ! -f "${MLIR_AIE_ROOT}/utils/env_setup.sh" ]]; then
    echo "error: mlir-aie env_setup.sh missing at ${MLIR_AIE_ROOT}" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${MLIR_AIE_ROOT}/utils/env_setup.sh" "${MLIR_AIE_ROOT}/install"

export XILINX_XRT

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# ---------------------------------------------------------------------------
# 2. Compile the three per-stage AIE objects
# ---------------------------------------------------------------------------
# TODO(kernel-author): replace with real Peano invocations once the
# kernel bodies light up. The target triple is `aie2p-none-unknown-elf`;
# `aie_api/aie.hpp` is on the include path via env_setup.sh; the flags
# ${KERNEL_CC} and ${KERNEL_CFLAGS} come out of env_setup.sh too.
#
# Reference commands (NOT YET FUNCTIONAL — bodies are stubs):
#
#   ${KERNEL_CC} ${KERNEL_CFLAGS} \
#       -DDIM_M=${m} -DDIM_K=${k} \
#       -c ../../../../npu-kernels/bitnet/unpack_ternary_2bit_to_int8.cc \
#       -o unpack_${m}x${k}.o
#
#   ${KERNEL_CC} ${KERNEL_CFLAGS} -Di8_i32_ONLY \
#       -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} \
#       -c ${MLIR_AIE_ROOT}/aie_kernels/aie2p/mm.cc \
#       -o mm_${m}x${k}x${n}.o
#
#   ${KERNEL_CC} ${KERNEL_CFLAGS} \
#       -DDIM_M=${m} -DDIM_N=${n} \
#       -c ../../../../npu-kernels/bitnet/scale_i32_fp16.cc \
#       -o scale_${m}x${n}.o
echo "[scaffold] would compile unpack_${m}x${k}.o, mm_${m}x${k}x${n}.o, scale_${m}x${n}.o" >&2

# Alternate single-file fused path (older design kept as parity oracle):
#
#   ${PEANO_ROOT}/bin/clang --target=aie2p-none-unknown-elf -O2 \
#       -c ../kernels/bitnet_gemv_aie2p.cc -o bitnet_gemv_aie2p.o
#
echo "[scaffold] would compile bitnet_gemv_aie2p.o (fused-path parity oracle)" >&2

# ---------------------------------------------------------------------------
# 3. Emit MLIR + run aiecc to produce the xclbin
# ---------------------------------------------------------------------------
# TODO(kernel-author): drive the IRON emitter at
# ../../../npu-kernels/bitnet/bitnet_gemv.py; it writes
# aie_${M}x${K}x${N}_${m}x${k}x${n}.mlir here. Then aiecc.py:
#
#   python -m aie.iron.compile \
#       --emitter ../../../npu-kernels/bitnet/bitnet_gemv.py \
#       --device ${devicename} \
#       --M ${M} --K ${K} --N ${N} \
#       --m ${m} --k ${k} --n ${n}
#
#   aiecc.py aie_${M}x${K}x${N}_${m}x${k}x${n}.mlir \
#       --aie-generate-xclbin \
#       --xclbin-name final_${M}x${K}x${N}_${m}x${k}x${n}.xclbin \
#       --kernels unpack_${m}x${k}.o,mm_${m}x${k}x${n}.o,scale_${m}x${n}.o
echo "[scaffold] would emit aie_${M}x${K}x${N}_${m}x${k}x${n}.mlir" >&2
echo "[scaffold] would run aiecc to produce final_${M}x${K}x${N}_${m}x${k}x${n}.xclbin" >&2

# ---------------------------------------------------------------------------
# 4. Smoke the artifact via pyxrt
# ---------------------------------------------------------------------------
# Same proven loader as axpy (see project_npu_unlocked.md — 160/160 pass).
# The crates/1bit-aie Rust side loads via libxrt at runtime; pyxrt is only
# a bring-up-side sanity harness.
echo "[scaffold] would run pyxrt smoke: load xclbin, dispatch one tile, \
diff vs CPU reference from rocm-cpp/aie/halo_ternary_mm.cpp" >&2

echo "build_aie.sh: scaffold complete (no artifacts produced)."
