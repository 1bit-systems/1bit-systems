/** npu_engine_hybrid_flm.h — Hybrid FLM NPU GEMM context.
 *
 * Uses FastFlowLM's mm.xclbin (501KB fused AIE kernel) with:
 * 1. Runtime instruction generation via gemm_generate_sequence_i8_split()
 * 2. 1 contiguous weight BO per GEMM type (NOT per-layer BOs)
 * 3. Pre-allocated all-layer weights with per-layer DDR offsets
 *
 * Architecture:
 *   mm.xclbin provides a single fused AIE GEMM kernel (6×8 tile array)
 *   that handles ALL shapes (M, K, N) via RTP registers.
 *   Kernel args: (opcode, ctrl, reserved, bA, bW, bC)
 *     arg_idx=0 → bA (activations, per-inference)
 *     arg_idx=1 → bW (weights, all layers contiguous)
 *     arg_idx=3 → bC (output)
 *
 * Weight BO layout (per GEMM type):
 *   [Layer 0: K×N bytes] [Layer 1: K×N bytes] ... [Layer NC-1: K×N bytes]
 *
 * Instruction generation per layer:
 *   gemm_generate_sequence_i8_split(seq, M, K, N,
 *       a_ddr_offset=0,              // A at offset 0 in bA
 *       b_base_offset=layer*K*N,     // B at per-layer offset in bW
 *       ...);
 *
 * vs the current open-source engine:
 *   ❌ Per-op xclbins (QKV.xclbin, O.xclbin, GU.xclbin, D.xclbin)
 *      → 235-317KB each, suboptimal AIE placement
 *   ❌ Per-layer weight BOs → DMA thrashing (NC ioctls per GEMM)
 *   ❌ Pre-compiled instruction files → inflexible
 *
 *   ✅ Single 501KB mm.xclbin (FLM's optimized fused kernel)
 *   ✅ 1 weight BO per GEMM type → 4 total (QKV, O, GU, D)
 *   ✅ Runtime instructions via gemm_generate_sequence_i8_split()
 *   ✅ Per-layer DDR offsets within contiguous weight BO
 */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

// ── For npu_sequence class (needed by init() to generate instructions) ──
#include "npu_utils/npu_instr_utils.hpp"

void gemm_generate_sequence_i8_split(
    class npu_sequence* seq, uint32_t M, uint32_t K, uint32_t N,
    uint32_t a_ddr_offset, uint32_t b_base_offset,
    bool add_bias, int activation,
    uint32_t bias_offset, uint32_t output_offset);

/// HybridFlmCtx — Single contiguous weight BO + runtime instructions.
///
/// Replaces I8Ctx's per-layer BOs with one BO containing all layers'
/// weights. Instructions are generated at init time for each layer
/// with correct per-layer DDR offsets.
///
/// Template parameters (GEMM-specific):
///   M = batch size (XM=128 for all models)
///   K = inner dimension (H, NH*HD, IM, etc.)
///   N = output dimension (qkv_total, H, etc.)
///   NL = number of layers (NC)
///
struct HybridFlmCtx {
    int MD;          // M (output rows / batch)
    int KD;          // K (inner dimension)
    int ND;          // N (output cols)
    int NL;          // Number of layers

    // XRT objects
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;    // regular kernel (per-call instr BO)

    // BOs: activation, weight (1 per GEMM type), output
    std::unique_ptr<xrt::bo> bA;       // activations: M × K bytes (i8)
    std::unique_ptr<xrt::bo> bW;       // weights: NL × K × N bytes (i8), single BO
    std::unique_ptr<xrt::bo> bC;       // output: M × N × 2 bytes (i16)

    // Per-layer instruction BOs (each contains the instructions for one layer)
    // Using unique_ptr array since instructions differ per layer (different b_base_offset)
    std::vector<std::unique_ptr<xrt::bo>> layer_instr_bo;

    // Per-layer instruction data (kept for potential re-sync)
    std::vector<std::vector<uint32_t>> layer_instr_data;

    // Group scales for dequantization
    std::vector<std::vector<float>> group_scales;

    // Mapped pointers
    int8_t*  Am;   // mapped from bA
    int16_t* Cm;   // mapped from bC

    bool initialized;

    HybridFlmCtx()
        : MD(0), KD(0), ND(0), NL(0)
        , Am(nullptr), Cm(nullptr)
        , initialized(false)
    {}

    ~HybridFlmCtx() {
        // Mapped pointers are owned by their respective BOs.
        // BOs are owned by unique_ptrs → destroyed automatically.
    }

    bool isReady() const {
        return initialized && k && bA && bW && bC &&
               !layer_instr_bo.empty();
    }

    /// Initialize with FLM's mm.xclbin.
    ///
    /// Generates NL instruction sequences, one per layer, each with
    /// b_base_offset = layer × K × N. Creates 1 weight BO with all
    /// NL layer weights contiguous.
    ///
    /// @param dev      XRT device
    /// @param xp       Path to FLM's mm.xclbin
    /// @param mdim     M (output rows, typically XM=128)
    /// @param kdim     K (inner dimension)
    /// @param ndim     N (output cols)
    /// @param nlayers  Number of transformer layers
    bool init(xrt::device& dev, const char* xp,
              int mdim, int kdim, int ndim, int nlayers) {
        MD = mdim;
        KD = kdim;
        ND = ndim;
        NL = nlayers;

        if (MD <= 0 || KD <= 0 || ND <= 0 || NL <= 0) {
            fprintf(stderr, "  HybridFlmCtx: invalid dims M=%d K=%d N=%d NL=%d\n",
                    MD, KD, ND, NL);
            return false;
        }

        // Load xclbin
        try {
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            dev.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(dev, xc->get_uuid());
            // Use regular kernel (not extended) so we can pass instructions per-call
            k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        } catch (std::exception& ex) {
            fprintf(stderr, "  HybridFlmCtx: xclbin/kernel init failed: %s\n", ex.what());
            return false;
        }

        // Get kernel group IDs for BO allocation
        int grp_a  = k->group_id(3);  // arg_idx=0 (A activations)
        int grp_w  = k->group_id(4);  // arg_idx=1 (W weights)
        int grp_c  = k->group_id(5);  // arg_idx=2 (C output)
        int grp_ins = k->group_id(1); // arg_idx=1 (instructions)

        // Allocate BOs
        size_t a_bytes = (size_t)MD * KD;           // int8 activations
        size_t w_bytes = (size_t)NL * KD * ND;      // int8 weights, all layers
        size_t c_bytes = (size_t)MD * ND * 2;       // int16 output

        bA = std::make_unique<xrt::bo>(dev, a_bytes, XRT_BO_FLAGS_HOST_ONLY, grp_a);
        bW = std::make_unique<xrt::bo>(dev, w_bytes, XRT_BO_FLAGS_HOST_ONLY, grp_w);
        bC = std::make_unique<xrt::bo>(dev, c_bytes, XRT_BO_FLAGS_HOST_ONLY, grp_c);

        Am = (int8_t*)bA->map();
        Cm = (int16_t*)bC->map();

        // Zero out weight BO (safety)
        memset(bW->map(), 0, w_bytes);

        // Generate per-layer instruction sequences
        layer_instr_bo.resize(NL);
        layer_instr_data.resize(NL);
        group_scales.resize(NL);

        for (int l = 0; l < NL; l++) {
            npu_sequence seq(device_npu2);
            uint32_t b_base_offset = (uint32_t)l * KD * ND;

            gemm_generate_sequence_i8_split(
                &seq, (uint32_t)MD, (uint32_t)KD, (uint32_t)ND,
                0,                      // a_ddr_offset = 0 (start of bA)
                b_base_offset,          // b_base_offset = layer * K * N
                false,                  // no bias
                0,                      // no activation
                0,                      // no bias offset
                0                       // output via kernel arg, not DDR offset
            );

            auto [dp, sz] = seq.dump();
            layer_instr_data[l].assign(dp, dp + sz / sizeof(uint32_t));

            // Create instruction BO and sync to device
            layer_instr_bo[l] = std::make_unique<xrt::bo>(
                dev, layer_instr_data[l].size() * sizeof(uint32_t),
                XCL_BO_FLAGS_CACHEABLE, grp_ins);
            memcpy(layer_instr_bo[l]->map(), layer_instr_data[l].data(),
                   layer_instr_data[l].size() * sizeof(uint32_t));
            layer_instr_bo[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        initialized = true;
        return true;
    }

    /// Pack weight data for layer l into the contiguous weight BO.
    ///
    /// @param l     Layer index
    /// @param w     Float weight data (row-major, K×N)
    /// @param K     Actual K dimension from dequant
    /// @param N     Actual N dimension from dequant
    /// @param sout  Output: per-group scale
    void packB(int l, const float* w, int K, int N, float& sout) {
        if (l < 0 || l >= NL) return;
        int num_groups = (K + 31) / 32;
        group_scales[l].resize(num_groups);

        // Offset in the contiguous weight BO
        size_t layer_off = (size_t)l * KD * ND;
        auto* Bm = (int8_t*)bW->map() + layer_off;

        for (int g = 0; g < num_groups; g++) {
            int g_start = g * 32;
            int g_size = std::min(32, K - g_start);
            float g_amax = 0;
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < g_size; i++) {
                    float a = fabsf(w[(g_start + i) * N + j]);
                    if (std::isfinite(a) && a > g_amax) g_amax = a;
                }
            }
            if (g_amax < 1e-12f) g_amax = 1.0f;
            group_scales[l][g] = g_amax / 127.0f;
            float g_is = 127.0f / g_amax;
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < g_size; i++) {
                    float v = w[(g_start + i) * N + j];
                    if (!std::isfinite(v)) v = 0;
                    int x = (int)roundf(v * g_is);
                    if (x > 127) x = 127; else if (x < -127) x = -127;
                    Bm[(g_start + i) * N + j] = (int8_t)x;
                }
            }
        }
    }

    /// Sync all layers' weights to device.
    /// Call once after all packB() calls complete.
    void sync_weights() {
        bW->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    /// Dynamic activation scale.
    static inline float dynamic_ascale(const float* x, int n) {
        float amax = 0;
        for (int i = 0; i < n; i++) {
            float a = fabsf(x[i]);
            if (std::isfinite(a) && a > amax) amax = a;
        }
        return (amax < 1e-12f) ? 1.0f : (amax / 127.0f);
    }

    /// Quantize activations into bA (async, no sync).
    inline int8_t* quantize_async(const float* A, int am, int ak, float ascale) {
        float ais = 1.0f / ascale;
        memset(Am, 0, (size_t)am * KD);
        for (int m = 0; m < am; m++) {
            for (int k = 0; k < ak; k++) {
                float v = A[m * ak + k];
                if (!std::isfinite(v)) v = 0;
                int q = (int)roundf(v * ais);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                Am[m * KD + k] = (int8_t)q;
            }
        }
        return Am;
    }

    /// Sync A BO to device.
    inline void sync_A() {
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    /// Compatibility overload (I8Ctx takes unused layer arg).
    inline void sync_A(int /*unused*/) {
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    /// Launch kernel for layer l.
    /// Uses the pre-generated instruction BO for layer l.
    /// Kernel signature: (mode, instr_bo, num_instrs, bA, bW, bC)
    inline xrt::run launch(int l) {
        return (*k)((unsigned)3,
                    *layer_instr_bo[l],
                    (unsigned)(layer_instr_data[l].size()),
                    *bA, *bW, *bC);
    }

    /// Sync A + Launch.
    inline xrt::run sync_and_launch(int l) {
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return (*k)((unsigned)3,
                    *layer_instr_bo[l],
                    (unsigned)(layer_instr_data[l].size()),
                    *bA, *bW, *bC);
    }

    /// Wait for kernel completion.
    inline void wait_kernel(xrt::run& r) {
        r.wait();
    }

    /// Dequantize output: sync C back, compute effective scale, convert to f32.
    inline void dequantize(xrt::run& r, float* C, int am, int an,
                           float ascale, float Bscale, int layer = -1) {
        r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (layer >= 0 && (size_t)layer < group_scales.size() &&
            !group_scales[layer].empty()) {
            float ssum = 0;
            for (float s : group_scales[layer]) ssum += s;
            Bscale = ssum / group_scales[layer].size();
        }
        float cs = ascale * Bscale;
        for (int m = 0; m < am; m++) {
            for (int n = 0; n < an; n++) {
                float val = (float)Cm[m * ND + n] * cs;
                C[m * an + n] = std::isfinite(val) ? val : 0.0f;
            }
        }
    }

    /// Async launch: quantize + sync + launch, return run handle.
    inline xrt::run launch_async(int l, const float* A, int am, int ak, float ascale) {
        quantize_async(A, am, ak, ascale);
        return sync_and_launch(l);
    }

    /// Finish async: wait + dequantize.
    inline void finish_async(xrt::run& r, float* C, int am, int an,
                             float ascale, float Bscale, int layer = -1) {
        r.wait();
        dequantize(r, C, am, an, ascale, Bscale, layer);
    }

    /// Synchronous go(): quantize, launch, wait, dequantize.
    inline bool go(int l, const float* A, int am, int ak,
                   float ascale, float Bscale, float* C, int an) {
        quantize_async(A, am, ak, ascale);
        auto r = sync_and_launch(l);
        r.wait();
        dequantize(r, C, am, an, ascale, Bscale, l);
        return true;
    }

    /// Sync back and dequant (after wait_kernel).
    inline void sync_back_and_dequant(float* C, int am, int an,
                                      float ascale, float Bscale, int layer = -1) {
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (layer >= 0 && (size_t)layer < group_scales.size() &&
            !group_scales[layer].empty()) {
            float ssum = 0;
            for (float s : group_scales[layer]) ssum += s;
            Bscale = ssum / group_scales[layer].size();
        }
        float cs = ascale * Bscale;
        for (int m = 0; m < am; m++) {
            for (int n = 0; n < an; n++) {
                float val = (float)Cm[m * ND + n] * cs;
                C[m * an + n] = std::isfinite(val) ? val : 0.0f;
            }
        }
    }
};
