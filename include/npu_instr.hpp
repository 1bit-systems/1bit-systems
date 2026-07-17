#pragma once
// npu_instr.hpp -- NPU instruction structures for NPU app.
// Shared between npu_engine, npu-infer/ tools, and the test suite.
#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cassert>

typedef uint32_t u32;
typedef int32_t  i32;

enum Opcode : u32 {
    OP_NOP        = 0,
    OP_WRITE      = 1,
    OP_READ       = 2,
    OP_GEMM       = 3,
    OP_SYNC       = 4,
    OP_DMA        = 5,
    OP_ATTENTION  = 6,
    OP_CONV       = 7,
    OP_POOL       = 8,
    OP_LAYERNORM  = 9,
    OP_SOFTMAX    = 10,
    OP_RMSNORM    = 11,
    OP_ROPE       = 12,
    OP_SILU       = 13,
    OP_ADD        = 14,
    OP_MUL        = 15,
    OP_SQRT       = 16,
    OP_DIV        = 17,
    OP_EXP        = 18,
    OP_LOG        = 19,
    OP_NEG        = 20,
    OP_ABS        = 21,
    OP_MAX        = 22,
    OP_MIN        = 23,
    OP_WHERE      = 24,
    OP_REDUCE_SUM = 25,
    OP_REDUCE_MAX = 26,
    OP_REDUCE_MIN = 27,
    OP_REDUCE_MEAN= 28,
    OP_CAST       = 29,
    OP_QUANTIZE   = 30,
    OP_DEQUANTIZE = 31,
    OP_CUSTOM     = 32,
    OP_END        = 0xFFFFFFFF,
};

struct GEMMDesc {
    u32 opcode;
    u32 pad0;
    u32 M, N, K;
    u32 lda, ldb, ldc;
    u32 A_offset;
    u32 B_offset;
    u32 C_offset;
    u32 relu;
    u32 act_scale;
    u32 wgt_scale;
    u32 reserved[4];
};

struct AttnDesc {
    u32 opcode;
    u32 pad0;
    u32 B, T, S, D;
    u32 n_heads_q, n_heads_kv;
    u32 Q_offset, K_offset, V_offset, O_offset;
    u32 softmax_scale;
    u32 reserved[8];
};

inline std::vector<u32> serialize_gemm(const GEMMDesc& d) {
    std::vector<u32> v(16, 0);
    v[0]  = d.opcode;
    v[2]  = d.M; v[3] = d.N; v[4] = d.K;
    v[5]  = d.lda; v[6] = d.ldb; v[7] = d.ldc;
    v[8]  = d.A_offset;
    v[9]  = d.B_offset;
    v[10] = d.C_offset;
    v[11] = d.relu;
    v[12] = d.act_scale;
    v[13] = d.wgt_scale;
    return v;
}

inline GEMMDesc deserialize_gemm(const u32* v) {
    GEMMDesc d = {};
    d.opcode    = v[0];
    d.M         = v[2]; d.N = v[3]; d.K = v[4];
    d.lda       = v[5]; d.ldb = v[6]; d.ldc = v[7];
    d.A_offset  = v[8];
    d.B_offset  = v[9];
    d.C_offset  = v[10];
    d.relu      = v[11];
    d.act_scale = v[12];
    d.wgt_scale = v[13];
    return d;
}

inline void print_gemm(const GEMMDesc& d) {
    fprintf(stderr, "GEMM: M=%u N=%u K=%u lda=%u ldb=%u ldc=%u\n", d.M, d.N, d.K, d.lda, d.ldb, d.ldc);
    fprintf(stderr, "  A_off=%u B_off=%u C_off=%u relu=%u act_scale=%u wgt_scale=%u\n",
            d.A_offset, d.B_offset, d.C_offset, d.relu, d.act_scale, d.wgt_scale);
}

inline bool validate_gemm(const GEMMDesc& d) {
    if (d.opcode != 3) return false;
    if (d.M == 0 || d.N == 0 || d.K == 0) return false;
    if (d.lda < d.K || d.ldb < d.N || d.ldc < d.N) return false;
    return true;
}

inline std::vector<u32> serialize_attn(const AttnDesc& d) {
    std::vector<u32> v(16, 0);
    v[0] = d.opcode;
    v[2] = d.B; v[3] = d.T; v[4] = d.S; v[5] = d.D;
    v[6] = d.n_heads_q; v[7] = d.n_heads_kv;
    v[8] = d.Q_offset;
    v[9] = d.K_offset;
    v[10] = d.V_offset;
    v[11] = d.O_offset;
    v[12] = d.softmax_scale;
    return v;
}

inline AttnDesc deserialize_attn(const u32* v) {
    AttnDesc d = {};
    d.opcode        = v[0];
    d.B             = v[2]; d.T = v[3]; d.S = v[4]; d.D = v[5];
    d.n_heads_q     = v[6]; d.n_heads_kv = v[7];
    d.Q_offset      = v[8];
    d.K_offset      = v[9];
    d.V_offset      = v[10];
    d.O_offset      = v[11];
    d.softmax_scale = v[12];
    return d;
}
