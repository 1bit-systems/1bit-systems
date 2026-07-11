#ifndef NPU_AQL_H
#define NPU_AQL_H

/* NPU AQL (Accelerated Queueing Layer) packet definitions.
 *
 * These match the NPU firmware's command format, reverse-engineered
 * from amdxdna's aie2_msg_priv.h message opcodes.
 */

#include <cstdint>

/* NPU firmware opcodes (from MSG_OP_* in amdxdna) */
enum npu_opcode : uint32_t {
    NPU_OP_NOP                = 0x0,
    NPU_OP_CREATE_CONTEXT     = 0x2,
    NPU_OP_DESTROY_CONTEXT    = 0x3,
    NPU_OP_GET_TELEMETRY      = 0x4,
    NPU_OP_SYNC_BO            = 0x7,
    NPU_OP_EXECUTE_BUFFER     = 0xC,
    NPU_OP_QUERY_STATUS       = 0xD,
    NPU_OP_EXEC_DPU           = 0x10,
    NPU_OP_CONFIG_CU          = 0x11,
    NPU_OP_CHAIN_EXEC_BUFFER  = 0x12,
    NPU_OP_CHAIN_EXEC_DPU     = 0x13,
    NPU_OP_CHAIN_EXEC_NPU     = 0x18,
};

/* NPU AQL packet header (16 bytes) */
struct __attribute__((packed)) npu_aql_header {
    uint32_t total_size : 11;   /* total packet size in dwords */
    uint32_t reserved   : 5;
    uint32_t protocol   : 8;    /* protocol version */
    uint32_t _pad       : 8;
    uint32_t id;                /* message ID */
    uint32_t opcode;            /* npu_opcode */
};

/* NPU AQL submit packet (for EXECUTE_BUFFER) */
struct __attribute__((packed)) npu_aql_exec {
    npu_aql_header header;
    uint64_t       cmd_addr;    /* GPU virtual address of command buffer */
    uint64_t       cmd_size;    /* size of command buffer */
    uint64_t       completion_signal_addr; /* write completion signal here */
};

/* Create context packet */
struct __attribute__((packed)) npu_aql_create_ctx {
    npu_aql_header header;
    uint32_t       pasid;       /* process PASID for IOMMU SVA */
    uint32_t       num_cols;    /* number of AIE columns */
    uint32_t       priority;
    uint32_t       _pad;
};

/* Status response from NPU */
struct __attribute__((packed)) npu_aql_status {
    uint32_t status_code;
    uint32_t _pad[3];
};

#endif /* NPU_AQL_H */
