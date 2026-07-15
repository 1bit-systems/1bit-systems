// test_vulkan_gemv.cpp — correctness + throughput proof for the portable
// Vulkan TQ2_0_g128 / Q1_0_g128 Bonsai GEMV kernels
// (kernels/vulkan/dmmv_tq2_bonsai.comp, dmmv_q1_bonsai.comp), run via the
// minimal src/vulkan_rt.h runtime.
//
// Deliberately standalone: does NOT touch bitnet_decode.cpp's model
// loading or the bonsai.h C API. Those use raw HIP device pointers, which
// don't map onto Vulkan buffers without a real design pass (VkBuffer isn't
// a raw pointer) -- rushing that integration now would repeat the scope
// overrun from earlier work on this feature. This tool proves the kernel
// math and Vulkan plumbing work correctly and fast; wiring into the real
// decode path is a deliberate follow-on step.
//
// Usage: ./test_vulkan_gemv

#include "../src/vulkan_rt.h"
#include "q1_tq2_vk_ref.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>

struct PushConstants {
    uint32_t M;
    uint32_t K;
    uint32_t a_offset;
    uint32_t x_offset;
    uint32_t y_offset;
    uint32_t acc_mode;
};

static const uint32_t kRowsPerWg = 2; // matches local_size_x=64, 2 rows/workgroup in both shaders

// ── TQ2 synthetic data ──────────────────────────────────────────────────────

static uint16_t f32ToF16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 31) & 1u;
    int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | 0x7C00u);
    return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | (mant >> 13));
}

static void fillTq2Weights(std::vector<uint8_t>& dst, uint32_t m, uint32_t k, uint32_t salt) {
    uint32_t num_blocks = k / vkref::kTq2BlockWeights;
    size_t row_bytes = (size_t)num_blocks * vkref::kTq2BlockBytes;
    dst.resize((size_t)m * row_bytes);
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t block = 0; block < num_blocks; block++) {
            size_t off = (size_t)row * row_bytes + (size_t)block * vkref::kTq2BlockBytes;
            float scale = 0.03125f * (float)(1 + (row + block) % 8);
            uint16_t d16 = f32ToF16(scale);
            memcpy(&dst[off], &d16, 2);
            for (uint32_t byte_idx = 0; byte_idx < 32; byte_idx++) {
                uint8_t packed = 0;
                for (uint32_t j = 0; j < 4; j++) {
                    uint32_t code = (byte_idx + j + block + row + salt) % 4; // exercises all 4 codes incl. 3
                    packed |= (uint8_t)(code << (j * 2));
                }
                dst[off + 2 + byte_idx] = packed;
            }
        }
    }
}

static void fillQ1Weights(std::vector<uint8_t>& dst, uint32_t m, uint32_t k, uint32_t salt) {
    uint32_t num_blocks = k / vkref::kQ1BlockWeights;
    size_t row_bytes = (size_t)num_blocks * vkref::kQ1BlockBytes;
    dst.resize((size_t)m * row_bytes);
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t block = 0; block < num_blocks; block++) {
            size_t off = (size_t)row * row_bytes + (size_t)block * vkref::kQ1BlockBytes;
            float scale = 0.03125f * (float)(1 + (row + block) % 8);
            uint16_t d16 = f32ToF16(scale);
            memcpy(&dst[off], &d16, 2);
            for (uint32_t byte_idx = 0; byte_idx < 16; byte_idx++) {
                dst[off + 2 + byte_idx] = (uint8_t)((byte_idx + block + row + salt) % 256);
            }
        }
    }
}

static void fillActivations(std::vector<uint16_t>& dst, uint32_t k, uint32_t salt) {
    dst.resize(k);
    for (uint32_t i = 0; i < k; i++) {
        float lane = (float)(((i + salt) % 11) + 1);
        dst[i] = f32ToF16(lane * 0.0625f);
    }
}

// ── Correctness ──────────────────────────────────────────────────────────

enum class Format { Tq2, Q1 };

static bool runCorrectness(vkrt::VkCtx& ctx, const char* spvPath, Format fmt) {
    const uint32_t m = 16, k = 256;

    std::vector<uint8_t> weights;
    std::vector<uint16_t> act;
    std::vector<float> y_ref(m);

    if (fmt == Format::Tq2) {
        fillTq2Weights(weights, m, k, 42);
        fillActivations(act, k, 7);
        vkref::tq2Gemv(weights.data(), act.data(), y_ref.data(), m, k);
    } else {
        fillQ1Weights(weights, m, k, 42);
        fillActivations(act, k, 7);
        vkref::q1Gemv(weights.data(), act.data(), y_ref.data(), m, k);
    }

    vkrt::Pipeline pipeline;
    pipeline.create(ctx, spvPath, 3, sizeof(PushConstants));

    vkrt::GpuBuffer bufW, bufX, bufY;
    bufW.create(ctx.dev, ctx.memProps, weights.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufX.create(ctx.dev, ctx.memProps, act.size() * sizeof(uint16_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufY.create(ctx.dev, ctx.memProps, m * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufW.upload(weights.data());
    bufX.upload(act.data());
    std::vector<float> y_zero(m, 0.0f);
    bufY.upload(y_zero.data());

    vkrt::GpuBuffer* bufs[3] = {&bufW, &bufX, &bufY};
    VkDescriptorSet ds = vkrt::createDescriptorSet(ctx, pipeline, bufs, 3);

    PushConstants pc{m, k, 0, 0, 0, 0};
    uint32_t wg_x = (m + kRowsPerWg - 1) / kRowsPerWg;
    vkrt::dispatchOnce(ctx, pipeline, ds, wg_x, 1, 1, &pc);

    std::vector<float> y_gpu(m);
    bufY.download(y_gpu.data());

    float max_abs_err = 0.0f;
    int fail_count = 0;
    for (uint32_t i = 0; i < m; i++) {
        float err = fabsf(y_gpu[i] - y_ref[i]);
        float tol = fmaxf(1e-3f, 1e-3f * fabsf(y_ref[i]));
        if (err > tol) fail_count++;
        max_abs_err = fmaxf(max_abs_err, err);
    }

    const char* name = (fmt == Format::Tq2) ? "TQ2" : "Q1";
    if (fail_count > 0) {
        printf("  %s correctness FAILED: %d/%u rows out of tolerance, max_abs_err=%.6f\n", name, fail_count, m, max_abs_err);
    } else {
        printf("  %s correctness PASSED: M=%u K=%u, max_abs_err=%.6f\n", name, m, k, max_abs_err);
    }

    vkFreeDescriptorSets(ctx.dev, ctx.dpool, 1, &ds);
    pipeline.destroy(ctx.dev);
    bufW.destroy();
    bufX.destroy();
    bufY.destroy();

    return fail_count == 0;
}

// ── Throughput ───────────────────────────────────────────────────────────

static void runThroughput(vkrt::VkCtx& ctx, const char* spvPath, Format fmt) {
    const uint32_t m = 6912, k = 6912;
    const uint32_t warmup = 25, iterations = 200;

    std::vector<uint8_t> weights;
    std::vector<uint16_t> act;
    uint32_t block_bytes = (fmt == Format::Tq2) ? vkref::kTq2BlockBytes : vkref::kQ1BlockBytes;
    uint32_t block_weights = (fmt == Format::Tq2) ? vkref::kTq2BlockWeights : vkref::kQ1BlockWeights;

    if (fmt == Format::Tq2) fillTq2Weights(weights, m, k, 1);
    else fillQ1Weights(weights, m, k, 1);
    fillActivations(act, k, 1);

    vkrt::Pipeline pipeline;
    pipeline.create(ctx, spvPath, 3, sizeof(PushConstants));

    vkrt::GpuBuffer bufW, bufX, bufY;
    bufW.create(ctx.dev, ctx.memProps, weights.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufX.create(ctx.dev, ctx.memProps, act.size() * sizeof(uint16_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufY.create(ctx.dev, ctx.memProps, m * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bufW.upload(weights.data());
    bufX.upload(act.data());

    vkrt::GpuBuffer* bufs[3] = {&bufW, &bufX, &bufY};
    VkDescriptorSet ds = vkrt::createDescriptorSet(ctx, pipeline, bufs, 3);

    PushConstants pc{m, k, 0, 0, 0, 0};
    uint32_t wg_x = (m + kRowsPerWg - 1) / kRowsPerWg;

    vkrt::dispatchRepeatedTimed(ctx, pipeline, ds, wg_x, 1, 1, &pc, warmup);
    double ms_total = vkrt::dispatchRepeatedTimed(ctx, pipeline, ds, wg_x, 1, 1, &pc, iterations);
    double ms_per_iter = ms_total / iterations;

    double bytes_per_iter = (double)((size_t)m * (k / block_weights) * block_bytes) +
                             (double)k * 2.0 + (double)m * 4.0;
    double gbps = bytes_per_iter / (ms_per_iter / 1000.0) / 1e9;

    const char* name = (fmt == Format::Tq2) ? "TQ2" : "Q1";
    printf("  %s M=%u K=%u: %.3f ms/iter | %.1f GB/s (warmup=%u, iterations=%u)\n",
           name, m, k, ms_per_iter, gbps, warmup, iterations);

    vkFreeDescriptorSets(ctx.dev, ctx.dpool, 1, &ds);
    pipeline.destroy(ctx.dev);
    bufW.destroy();
    bufX.destroy();
    bufY.destroy();
}

int main() {
    vkrt::VkCtx ctx;
    ctx.init();
    printf("GPU: %s\n", ctx.deviceName);

    std::string shader_dir = VK_SHADER_DIR;
    std::string tq2_spv = shader_dir + "/dmmv_tq2_bonsai.spv";
    std::string q1_spv = shader_dir + "/dmmv_q1_bonsai.spv";

    printf("\n== Correctness ==\n");
    bool ok = true;
    ok &= runCorrectness(ctx, tq2_spv.c_str(), Format::Tq2);
    ok &= runCorrectness(ctx, q1_spv.c_str(), Format::Q1);

    printf("\n== Throughput (steady-state, 6912x6912) ==\n");
    runThroughput(ctx, tq2_spv.c_str(), Format::Tq2);
    runThroughput(ctx, q1_spv.c_str(), Format::Q1);

    ctx.destroy();
    return ok ? 0 : 1;
}
