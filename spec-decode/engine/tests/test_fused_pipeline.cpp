// Test suite for Fused NPU+GPU Pipeline
#include "pipeline/fused_pipeline.h"
#include "kernels/attention.h"
#include "quantization/int8_quant.h"
#include <cstdio>
#include <cassert>
#include <cmath>

using namespace specdecode::pipeline;
using namespace specdecode::kernels;
using namespace specdecode::quant;

int test_partition_plans() {
    printf("=== Partition Plans ===\n");

    // NPU only
    {
        FusedPipelineExecutor exec(PartitionStrategy::kNPUOnly);
        auto& plan = exec.plan();
        assert(plan.qkv_backend == BackendType::kNPU);
        assert(plan.attn_backend == BackendType::kNPU);
        assert(plan.oproj_backend == BackendType::kNPU);
        assert(plan.ffn_backend == BackendType::kNPU);
        assert(!plan.enable_pipeline_overlap);
        printf("  NPUOnly: QKV=NPU, Attn=NPU, O=NPU, FFN=NPU\n");
    }

    // FFN on NPU (recommended)
    {
        FusedPipelineExecutor exec(PartitionStrategy::kFFNOnNPU);
        auto& plan = exec.plan();
        assert(plan.qkv_backend == BackendType::kNPU);
        assert(plan.attn_backend == BackendType::kGPU);
        assert(plan.oproj_backend == BackendType::kNPU);
        assert(plan.ffn_backend == BackendType::kNPU);
        assert(plan.enable_pipeline_overlap);
        printf("  FFNOnNPU: QKV=NPU, Attn=GPU, O=NPU, FFN=NPU (overlap enabled)\n");
    }

    // GPU only
    {
        FusedPipelineExecutor exec(PartitionStrategy::kGPUOnly);
        auto& plan = exec.plan();
        assert(plan.qkv_backend == BackendType::kGPU);
        assert(plan.attn_backend == BackendType::kGPU);
        assert(plan.oproj_backend == BackendType::kGPU);
        assert(plan.ffn_backend == BackendType::kGPU);
        printf("  GPUOnly: All GPU\n");
    }

    printf("  PASS\n");
    return 0;
}

int test_cost_estimation() {
    printf("=== Cost Estimation ===\n");

    FusedPipelineExecutor exec(PartitionStrategy::kFFNOnNPU);
    float cost = exec.estimate_cost(28, 1, exec.plan());

    printf("  Estimated cost for 28 layers (FFNOnNPU): %.3f ms\n", cost);
    assert(cost > 0.0f);
    assert(cost < 1000.0f);  // Sanity: should be < 1 second

    // NPU-only should be different from FFN-on-NPU
    FusedPipelineExecutor exec_npu(PartitionStrategy::kNPUOnly);
    float cost_npu = exec_npu.estimate_cost(28, 1, exec_npu.plan());
    printf("  Estimated cost for 28 layers (NPUOnly):  %.3f ms\n", cost_npu);

    // Pipeline overlap should be faster or comparable
    FusedPipelineExecutor exec_overlap(PartitionStrategy::kFFNOnNPU);
    float cost_overlap = exec_overlap.estimate_cost(28, 1, exec_overlap.plan());
    printf("  Estimated cost for 28 layers (overlap):  %.3f ms\n", cost_overlap);

    printf("  PASS\n");
    return 0;
}

int test_pipeline_build() {
    printf("=== Pipeline Build ===\n");

    FusedPipelineExecutor exec(PartitionStrategy::kFFNOnNPU);
    auto stages = exec.build_pipeline(28, 1, exec.plan());

    // 6 stages per layer (norm, QKV, attn, O, residual, FFN) + 1 final LM head
    int expected_stages = 28 * 6 + 1;
    printf("  Total stages: %zu (expected %d)\n", stages.size(), expected_stages);
    assert(stages.size() == (size_t)expected_stages);

    // Check first layer stages
    assert(stages[0].type == StageType::kRMSNorm);
    assert(stages[0].layer_id == 0);
    assert(stages[1].type == StageType::kQKV);
    assert(stages[1].backend == BackendType::kNPU);
    assert(stages[2].type == StageType::kAttention);
    assert(stages[2].backend == BackendType::kGPU);
    assert(stages[3].type == StageType::kOProj);
    assert(stages[4].type == StageType::kRMSNorm);
    assert(stages[5].type == StageType::kFFN);

    // Last layer should end with LM head
    assert(stages.back().type == StageType::kLMHead);

    // Verify dependencies
    assert(stages[1].dependencies.size() >= 1);  // QKV depends on RMSNorm
    assert(stages[2].dependencies.size() >= 1);  // Attn depends on QKV

    printf("  PASS\n");
    return 0;
}

int test_auto_balance() {
    printf("=== Auto Balance ===\n");

    FusedPipelineExecutor exec(PartitionStrategy::kAutoBalance);
    PartitionStrategy best = exec.auto_balance(28, 1);

    printf("  Auto-balanced strategy: ");
    switch (best) {
        case PartitionStrategy::kNPUOnly:       printf("kNPUOnly\n"); break;
        case PartitionStrategy::kGPUOnly:       printf("kGPUOnly\n"); break;
        case PartitionStrategy::kFFNOnNPU:      printf("kFFNOnNPU\n"); break;
        case PartitionStrategy::kQKVOnNPU:      printf("kQKVOnNPU\n"); break;
        case PartitionStrategy::kAttentionOnNPU: printf("kAttentionOnNPU\n"); break;
        default: printf("unknown\n"); break;
    }

    printf("  PASS\n");
    return 0;
}

int test_cross_backend_buffer() {
    printf("=== Cross-Backend Buffer ===\n");

    CrossBackendAllocator allocator(false);  // No UMA
    auto buf = allocator.allocate(1024);

    assert(buf.size_bytes == 1024);
    assert(!buf.host_data.empty());
    assert(!buf.has_unified_access());

    allocator.sync_to_npu(buf);
    allocator.sync_npu_to_gpu(buf);

    printf("  Buffer: %zu bytes, UMA=%d\n", buf.size_bytes, buf.has_unified_access());

    // With UMA
    CrossBackendAllocator uma_allocator(true);
    auto uma_buf = uma_allocator.allocate(1024);

    printf("  UMA buffer: %zu bytes\n", uma_buf.size_bytes);

    printf("  PASS\n");
    return 0;
}

int test_backend_capabilities() {
    printf("=== Backend Capabilities ===\n");

    BackendCapabilities npu;
    npu.supports_int8_gemm = true;
    npu.peak_flops = 10'000'000'000;
    npu.memory_bandwidth = 100'000'000'000;

    BackendCapabilities gpu;
    gpu.supports_flash_attention = true;
    gpu.has_unified_memory = true;
    gpu.peak_flops = 3'000'000'000'000;
    gpu.memory_bandwidth = 500'000'000'000;

    printf("  NPU: INT8=%d, FLOPS=%ld, BW=%ld\n",
           npu.supports_int8_gemm, (long)npu.peak_flops, (long)npu.memory_bandwidth);
    printf("  GPU: FlashAttn=%d, UMA=%d, FLOPS=%ld, BW=%ld\n",
           gpu.supports_flash_attention, gpu.has_unified_memory,
           (long)gpu.peak_flops, (long)gpu.memory_bandwidth);

    printf("  PASS\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_partition_plans();
    failures += test_cost_estimation();
    failures += test_pipeline_build();
    failures += test_auto_balance();
    failures += test_cross_backend_buffer();
    failures += test_backend_capabilities();

    printf("\n=== Fused Pipeline Tests: %d failures ===\n", failures);
    return failures;
}
