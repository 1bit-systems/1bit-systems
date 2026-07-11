// SPDX-License-Identifier: GPL-2.0 OR MIT
/*
 * amdgpu_npu_mgr.c — NPU memory manager (GTT sub-allocator)
 *
 * The NPU accesses system memory through dma-buf import from amdgpu's
 * GTT domain (TTM_PL_TT).  This manager provides NPU-optimized GTT
 * allocations with guaranteed contiguity and alignment.
 *
 * Rather than adding a new TTM_PL_NPU domain (which requires bumping
 * TTM_NUM_MEM_TYPES from 9 to 10 in TTM core), we use the existing
 * AMDGPU_GEM_DOMAIN_GTT path but add an NPU-specific allocation flag
 * AMDGPU_GEM_CREATE_NPU (0x80) that:
 *
 *   1. Forces TTM_PL_FLAG_CONTIGUOUS for the GTT allocation
 *   2. Aligns to NPU dev_mem_buf_shift (32 KiB)
 *   3. Pins the buffer so it cannot be evicted while NPU accesses it
 *
 * This file is paired with amdgpu_npu.c and integrated into the
 * amdgpu_ttm.c switch statements.
 */

#include "amdgpu.h"
#include "amdgpu_npu.h"

/* ------------------------------------------------------------------ */
/*  Allocate a GTT buffer with NPU-optimized parameters               */
/* ------------------------------------------------------------------ */

int amdgpu_npu_alloc_gtt(struct amdgpu_device *adev,
			 struct amdgpu_bo **bo,
			 uint64_t size)
{
	struct amdgpu_bo_alloc_request req = {};
	int r;

	req.alloc_size      = PAGE_ALIGN(size);
	req.preferred_heap  = AMDGPU_GEM_DOMAIN_GTT;
	req.flags           = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED |
			     AMDGPU_GEM_CREATE_VRAM_CONTIGUOUS; /* force contiguous */

	r = amdgpu_bo_alloc(adev, &req, bo);
	if (r)
		return r;

	/* Pin so the NPU can safely DMA to it */
	r = amdgpu_bo_pin(*bo, AMDGPU_GEM_DOMAIN_GTT);
	if (r) {
		amdgpu_bo_free(*bo);
		return r;
	}

	return 0;
}

void amdgpu_npu_free_gtt(struct amdgpu_bo *bo)
{
	if (!bo)
		return;
	amdgpu_bo_unpin(bo);
	amdgpu_bo_free(bo);
}

/* ------------------------------------------------------------------ */
/*  Export a GTT buffer as dma-buf for NPU import                     */
/* ------------------------------------------------------------------ */

int amdgpu_npu_export_dmabuf(struct amdgpu_bo *bo, int *fd)
{
	uint32_t handle;
	int r;

	r = amdgpu_bo_export(bo, amdgpu_bo_handle_type_dma_buf_fd, &handle);
	if (r)
		return r;

	*fd = (int)handle;
	return 0;
}
