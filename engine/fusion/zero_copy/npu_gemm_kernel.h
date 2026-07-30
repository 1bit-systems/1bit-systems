// npu_gemm_kernel.h — in-process NPU GEMM invocation (int8, single layer op).
//
// Extracted from test_pipeline_real.cpp's original NpuGemmCtx, then rewritten
// to match engine/npu/src/bench_gemm.cpp's invocation convention — the one
// documented in engine/npu/README.md as "verified correct on hardware" for
// these exact xclbins (final_i8_{QKV,O,GU,D}_*.xclbin).
//
// The original NpuGemmCtx used xrt::experimental's aiebu_assembler + xrt::elf
// + xrt::module + xrt::ext::kernel to bake the instruction stream into an ELF,
// then called k->operator()(3, 0, 0, A, B, C) — passing 0,0 where an
// instruction buffer + size would otherwise go. That path runs without
// crashing but produced wrong output, which turned out to be a SEPARATE bug
// from the API choice (see below) — both were fixed together.
//
// THE ACTUAL BUG (found by rebuilding an xclbin from source and reading its
// own generated MLIR): the GEMM kernel's output tensor is int32
// (`matmul_i8_i32`, `aie.runtime_sequence(..., memref<MD*NDxi32>)`), not
// int16. The original NpuGemmCtx/bench_gemm.cpp code allocated the output
// buffer as `MD*ND*2` bytes and read it as `int16_t*` — half the required
// size, read at half the correct element width. That produces exactly the
// symptom observed: a buffer that's structurally too small, reinterpreted at
// the wrong stride, giving a plausible-looking-but-wrong alternating pattern
// (every other 16-bit slot ends up reading the high/low half of an adjacent
// int32). Confirmed by rebuilding final_i8_{GU,D}_qwen3_0_6b.xclbin from
// scratch via generators/n1_core_i8_v23.py + aiecc (see engine/npu/README.md)
// — the freshly-built xclbin's own MLIR source declares the output memref as
// `xi32`, and switching this class from int16 to int32 output fixed
// test_npu_ffn_real_weights (cosine similarity ~0 -> matches CPU reference).
//
// Loads a pre-built xclbin + instruction-transaction blob (e.g.
// engine/npu/xclbins/final_i8_{GU,D}_*.xclbin + insts_i8_{GU,D}_*.txt) and
// runs int8-quantized GEMM: C[M,N] (int32 accumulator) = quant(A[M,K]) @
// quant(B[K,N]), with a single dynamic per-call scale for A and a scale
// fixed at packB() time for B.
//
// Weight layout: B must be [K,N] row-major (input-major, i.e. y = x @ W where
// W is [in_features, out_features]) — the transpose of GGUF/PyTorch's
// nn.Linear [out_features, in_features] convention. Callers loading GGUF/.1bp
// weights must transpose before calling packB().
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <memory>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

namespace fusion {

class NpuGemmKernel {
public:
    int MD, KD, ND;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::unique_ptr<xrt::bo> bI, bA, bB, bC;
    int8_t* Am = nullptr; int32_t* Cm = nullptr;
    bool ok = false;

    bool init(xrt::device& d, const char* xp, const char* ip, int md, int kd, int nd) {
        MD = md; KD = kd; ND = nd;
        FILE* f = fopen(ip, "rb"); if (!f) return false;
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        ins.resize(sz / 4); size_t rd = fread(ins.data(), 4, ins.size(), f); fclose(f);
        if (rd != ins.size()) return false;
        try {
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        } catch (...) { return false; }

        bI = std::make_unique<xrt::bo>(d, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, k->group_id(1));
        memcpy(bI->map(), ins.data(), ins.size() * 4);
        bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        bA = std::make_unique<xrt::bo>(d, (size_t)MD * KD, XRT_BO_FLAGS_HOST_ONLY, k->group_id(3));
        bB = std::make_unique<xrt::bo>(d, (size_t)KD * ND, XRT_BO_FLAGS_HOST_ONLY, k->group_id(4));
        bC = std::make_unique<xrt::bo>(d, (size_t)MD * ND * 4, XRT_BO_FLAGS_HOST_ONLY, k->group_id(5));
        memset(bA->map(), 0, (size_t)MD * KD);
        memset(bC->map(), 0, (size_t)MD * ND * 4);
        Am = (int8_t*)bA->map(); Cm = (int32_t*)bC->map(); ok = true; return true;
    }

    // `w` must be [K,N] row-major (see file header re: transpose from GGUF layout).
    void packB(const float* w, int K, int N, float& sout) {
        float amax = 0;
        for (int i = 0; i < K * N; i++) { float a = fabsf(w[i]); if (std::isfinite(a) && a > amax) amax = a; }
        sout = (amax < 1e-12f) ? 1.0f : amax / 127.0f;
        float is = 127.0f / (amax < 1e-12f ? 1.0f : amax);
        auto* Bm = (int8_t*)bB->map();
        for (int i = 0; i < K * N; i++) {
            float v = w[i]; if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * is); if (q > 127) q = 127; else if (q < -127) q = -127;
            Bm[i] = (int8_t)q;
        }
        bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void go(const float* A, int am, int ak, float as_, float Bs, float* C, int an) {
        float ais = 1.0f / as_;
        memset(Am, 0, (size_t)MD * KD);
        for (int mi = 0; mi < am; mi++) for (int ki = 0; ki < ak; ki++) {
            float v = A[mi * ak + ki]; if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * ais); if (q > 127) q = 127; else if (q < -127) q = -127;
            Am[mi * KD + ki] = (int8_t)q;
        }
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // Re-sync instruction buffer — XDNA AIE hardware consumes it on each run
        bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r = (*k)((unsigned)3, *bI, (unsigned)ins.size(), *bA, *bB, *bC);
        r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs = as_ * Bs;
        for (int m = 0; m < am; m++) for (int n = 0; n < an; n++) {
            float val = (float)Cm[m * ND + n] * cs;
            C[m * an + n] = std::isfinite(val) ? val : 0.0f;
        }
    }
};

} // namespace fusion
