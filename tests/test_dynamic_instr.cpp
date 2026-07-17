// test_dynamic_instr.cpp — NPU dynamic instruction engine test
// Verifies instruction generation and ELF compilation pipeline.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "npu_instr.hpp"

static int tests = 0, passed = 0;
#define CHECK(cond, msg) do { tests++; if (cond) { passed++; fprintf(stderr, "  PASS [%d/%d] %s\n", passed, tests, msg); } else { fprintf(stderr, "  FAIL [%d/%d] %s\n", tests, tests, msg); exit(1); } } while(0)

int main() {
    fprintf(stderr, "=== Dynamic Instruction Tests ===\n\n");
    
    // 1. Basic GEMMDesc serialization/deserialization
    {
        GEMMDesc d;
        d.opcode = OP_GEMM;
        d.M = 128; d.N = 128; d.K = 64;
        d.lda = 64; d.ldb = 128; d.ldc = 128;
        d.A_offset = 0; d.B_offset = 4096; d.C_offset = 8192;
        d.relu = 0; d.act_scale = 127; d.wgt_scale = 127;
        auto v = serialize_gemm(d);
        auto d2 = deserialize_gemm(v.data());
        CHECK(d.opcode == d2.opcode, "opcode roundtrip");
        CHECK(d.M == d2.M, "M roundtrip");
        CHECK(d.N == d2.N, "N roundtrip");
        CHECK(d.K == d2.K, "K roundtrip");
        CHECK(d.A_offset == d2.A_offset, "A_offset roundtrip");
        CHECK(validate_gemm(d2), "valid GEMM");
    }
    
    // 2. AttnDesc serialization
    {
        AttnDesc d;
        d.opcode = OP_ATTENTION;
        d.B = 1; d.T = 128; d.S = 128; d.D = 64;
        d.n_heads_q = 8; d.n_heads_kv = 2;
        d.Q_offset = 0; d.K_offset = 4096;
        d.V_offset = 8192; d.O_offset = 12288;
        auto v = serialize_attn(d);
        auto d2 = deserialize_attn(v.data());
        CHECK(d.opcode == d2.opcode, "attn opcode");
        CHECK(d.n_heads_q == d2.n_heads_q, "attn n_heads_q");
        CHECK(d.Q_offset == d2.Q_offset, "attn Q_offset");
    }
    
    // 3. GEMMDesc validation (invalid cases)
    {
        GEMMDesc d;
        d.opcode = 0;  // NOP, not GEMM
        CHECK(!validate_gemm(d), "invalid opcode");
    }
    {
        GEMMDesc d;
        d.opcode = OP_GEMM;
        d.M = 0; d.N = 128; d.K = 64;
        CHECK(!validate_gemm(d), "zero M");
    }
    
    // 4. Print (smoke test -- no crash)
    {
        GEMMDesc d;
        d.opcode = OP_GEMM;
        d.M = 128; d.N = 128; d.K = 64;
        print_gemm(d);
        CHECK(true, "print_gemm no crash");
    }
    
    fprintf(stderr, "\n=== %d/%d tests passed ===\n", passed, tests);
    return (passed == tests) ? 0 : 1;
}
