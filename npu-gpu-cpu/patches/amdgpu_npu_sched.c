// SPDX-License-Identifier: GPL-2.0 OR MIT
/*
 * amdgpu_npu_sched.c — NPU ring scheduler
 *
 * Implements the NPU command submission ring, following the amdgpu_ring
 * pattern.  Commands are submitted to the NPU firmware via the SRAM-based
 * mailbox doorbell.
 *
 * Ring lifecycle (amdgpu_ring):
 *   amdgpu_ring_init()      — allocate ring buffer
 *   amdgpu_ring_alloc()     — reserve space for N commands
 *   amdgpu_ring_emit()      — write NPU AQL packets
 *   amdgpu_ring_commit()    — ring doorbell, notify firmware
 *   amdgpu_ring_fini()      — free ring buffer
 *
 * The NPU firmware reads commands from the ring using its SRAM mailbox
 * protocol (same as amdxdna's aie2_message.c).
 */

#include "amdgpu.h"
#include "amdgpu_npu.h"

/* NPU AQL packet opcodes (mirrors aie2_msg_priv.h) */
#define NPU_OP_CREATE_CONTEXT       0x2
#define NPU_OP_DESTROY_CONTEXT      0x3
#define NPU_OP_EXECUTE_BUFFER       0xC
#define NPU_OP_SYNC_BO              0x7
#define NPU_OP_QUERY_STATUS         0xD

/* Mailbox register offsets (from npu4_regs.c) */
#define NPU_MBOX_X2I_TAIL           0x0
#define NPU_MBOX_X2I_HEAD           0x4
#define NPU_MBOX_I2X_TAIL           0x1000
#define NPU_MBOX_I2X_HEAD           0x1004
#define NPU_DOORBELL_OFFSET         0x2000

/* ------------------------------------------------------------------ */
/*  NPU ring operations                                               */
/* ------------------------------------------------------------------ */

struct amdgpu_npu_ring {
	struct amdgpu_ring	base;
	void __iomem		*mbox_base;	  /* NPU SRAM mailbox */
	void __iomem		*doorbell;	  /* doorbell register */
	uint32_t		tail;		  /* write pointer */
};

/*
 * Initialize the NPU ring buffer in SRAM.
 * Called from amdgpu_npu_sw_init().
 */
int amdgpu_npu_ring_init(struct amdgpu_device *adev,
			 struct amdgpu_npu_ring *nring,
			 void __iomem *mbox_base)
{
	int r;

	nring->mbox_base = mbox_base;
	nring->doorbell  = mbox_base + NPU_DOORBELL_OFFSET;
	nring->tail      = 0;

	/* Initialize as an amdgpu_ring (for integration with
	 * amdgpu's scheduler infrastructure) */
	r = amdgpu_ring_init(adev, &nring->base, 1024, NULL, 0,
			     AMDGPU_RING_PRIO_DEFAULT, NULL);
	if (r)
		return r;

	/* Ring buffer allocated — firmware will DMA from it */
	dev_dbg(adev->dev, "NPU ring at mbox %p\n", mbox_base);
	return 0;
}

/*
 * Write a 64-bit command to the NPU ring and advance tail.
 */
void amdgpu_npu_ring_emit_qword(struct amdgpu_npu_ring *nring,
				uint64_t qword)
{
	void __iomem *slot = nring->mbox_base + nring->tail;

	writeq(qword, slot);
	nring->tail += 8;

	/* Ring buffer size: 4 KiB (512 qwords) */
	nring->tail &= 0xFFF;
}

/*
 * Submit an NPU AQL packet via the mailbox protocol.
 * This sends the packet to the NPU firmware and rings the doorbell.
 *
 * Mirrors aie2_cmd_submit() in amdxdna.
 */
int amdgpu_npu_submit_packet(struct amdgpu_npu_ring *nring,
			     uint32_t opcode,
			     uint64_t arg0,
			     uint64_t arg1)
{
	/* Write command header */
	amdgpu_npu_ring_emit_qword(nring, (uint64_t)opcode |
				   (1ULL << 32)); /* total_size = 1 */

	/* Write arguments */
	amdgpu_npu_ring_emit_qword(nring, arg0);
	amdgpu_npu_ring_emit_qword(nring, arg1);

	/* Update NPU's tail pointer register (doorbell) */
	writel(nring->tail, nring->doorbell);

	return 0;
}

/*
 * Wait for NPU command completion.
 * Polls the I2X head pointer until it matches our tail.
 */
int amdgpu_npu_ring_wait(struct amdgpu_npu_ring *nring, uint32_t timeout_ms)
{
	unsigned long timeout = jiffies + msecs_to_jiffies(timeout_ms);
	uint32_t head;

	do {
		head = readl(nring->mbox_base + NPU_MBOX_I2X_HEAD);
		if (head == nring->tail)
			return 0; /* completed */
		cpu_relax();
	} while (time_before(jiffies, timeout));

	return -ETIMEDOUT;
}

/*
 * Teardown the NPU ring.
 */
void amdgpu_npu_ring_fini(struct amdgpu_npu_ring *nring)
{
	amdgpu_ring_fini(&nring->base);
}
