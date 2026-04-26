# ternary_gemv_i8_i32 build (Peano + MLIR-AIE -> .xclbin + insts)
#
# Targets: build/ternary_gemv_kernel.o      (Peano-compiled AIE core kernel)
#          build/ternary_gemv_design.mlir   (lowered MLIR-AIE design)
#          build/final_ternary_gemv_${M}x${K}.xclbin
#          build/insts_ternary_gemv_${M}x${K}.txt
#
# Required env / variables:
#   MLIR_AIE_INSTALL_DIR  -- path to mlir_aie wheel install (provides aiecc.py)
#   PEANO_INSTALL_DIR     -- path to llvm-aie wheel install (provides clang++)
#   M, K                  -- problem dims (M rows, K-vector input). Default 64,64.
#   devicename            -- npu (Phoenix) or npu2 (Strix/Strix Halo). Default npu2.
#
# Source it via the venv-aware wrapper at ../scripts/build_ternary_xclbin.sh
# which sources mlir-aie/utils/env_setup.sh and forwards M/K.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

srcdir := $(realpath $(dir $(firstword $(MAKEFILE_LIST))))
build_dir := $(srcdir)/../build

M ?= 64
K ?= 64
devicename ?= npu2

K_PACK := $(shell echo $$(( $(K) / 4 )) )

target_suffix := $(M)x$(K)
mlir_target := $(build_dir)/ternary_gemv_$(target_suffix).mlir
xclbin_target := $(build_dir)/final_ternary_gemv_$(target_suffix).xclbin
insts_target := $(build_dir)/insts_ternary_gemv_$(target_suffix).txt
kernel_obj := $(build_dir)/ternary_gemv_kernel.o

# Peano flags for AIE2/AIE2P. Mirrors PEANOWRAP{2,2P}_FLAGS verbatim from
# mlir-aie/programming_examples/makefile-common — just MLIR_AIE_DIR substituted
# for the wheel install at MLIR_AIE_INSTALL_DIR.
WARNING_FLAGS := -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw
ifeq ($(devicename),npu2)
PEANO_TRIPLE := aie2p-none-unknown-elf
else
PEANO_TRIPLE := aie2-none-unknown-elf
endif
PEANOWRAP_FLAGS := -O2 -std=c++20 --target=$(PEANO_TRIPLE) \
    $(WARNING_FLAGS) -DNDEBUG -I $(MLIR_AIE_INSTALL_DIR)/include

KERNEL_DEFINES := -DDIM_M=$(M) -DDIM_K_PACK=$(K_PACK)

KERNEL_CC := $(PEANO_INSTALL_DIR)/bin/clang++

aiecc_peano_flags := --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR)

.PHONY: all clean
all: $(xclbin_target)

$(build_dir):
	mkdir -p $@

$(kernel_obj): $(srcdir)/ternary_gemv_i8_i32.cc | $(build_dir)
	cd $(build_dir) && \
	$(KERNEL_CC) $(PEANOWRAP_FLAGS) $(KERNEL_DEFINES) -c $< -o $(notdir $@)

$(mlir_target): $(srcdir)/ternary_gemv_i8_i32.py | $(build_dir)
	python3 $< --dev $(devicename) -M $(M) -K $(K) > $@

$(xclbin_target): $(mlir_target) $(kernel_obj)
	cd $(build_dir) && \
	aiecc.py --aie-generate-xclbin --no-compile-host \
	    --xclbin-name=$(notdir $(xclbin_target)) \
	    --aie-generate-npu-insts \
	    --npu-insts-name=$(notdir $(insts_target)) \
	    $(aiecc_peano_flags) \
	    $(notdir $(mlir_target))

clean:
	rm -rf $(build_dir)/ternary_gemv_*.mlir \
	       $(build_dir)/final_ternary_gemv_*.xclbin \
	       $(build_dir)/insts_ternary_gemv_*.txt \
	       $(build_dir)/ternary_gemv_kernel.o
