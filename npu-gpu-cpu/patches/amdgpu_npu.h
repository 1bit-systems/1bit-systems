/* SPDX-License-Identifier: GPL-2.0 OR MIT */
#ifndef __AMDGPU_NPU_H__
#define __AMDGPU_NPU_H__

#include "amdgpu.h"

/* NPU IP block version descriptor — registered in early_init */
extern const struct amdgpu_ip_block_version npu_ip_block;

/* Per-device NPU state (defined in amdgpu_npu.c) */
struct amdgpu_npu;

#endif /* __AMDGPU_NPU_H__ */
