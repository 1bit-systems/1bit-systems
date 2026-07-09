// Fused Pipeline Cost Estimator — analyzes optimal partitioning strategy
// for NPU+GPU fusion across different batch sizes and model configurations
#include "pipeline/fused_pipeline.h"
#include <cstdio>
#include <vector>
#include <algorithm>

using namespace specdecode::pipeline;

void print_partition_table(const char* title, PartitionStrategy strategy,
                           float cost_single, float cost_batch4, float cost_batch16) {
    printf("%-20s %12.3f ms %12.3f ms %12.3f ms\n",
           title, cost_single, cost_batch4, cost_batch16);
}

int main() {
    printf("═══ Fused NPU+GPU Pipeline Cost Estimator ═══\n\n");

    const int N_LAYERS = 28;

    struct StrategyConfig {
        PartitionStrategy strategy;
        const char* name;
    };

    StrategyConfig strategies[] = {
        {PartitionStrategy::kNPUOnly,       "NPU Only"},
        {PartitionStrategy::kGPUOnly,       "GPU Only"},
        {PartitionStrategy::kFFNOnNPU,      "FFN on NPU"},
        {PartitionStrategy::kQKVOnNPU,      "QKV on NPU"},
        {PartitionStrategy::kAttentionOnNPU, "Attn on NPU"},
    };

    // ── Single batch analysis ──
    printf("── Single Batch (M=1 decode) ──\n");
    printf("%-20s %12s %12s %12s\n", "Strategy", "M=1", "M=4", "M=16");
    printf("%s\n", std::string(60, '-').c_str());

    for (auto& s : strategies) {
        FusedPipelineExecutor exec(s.strategy);
        float c1 = exec.estimate_cost(N_LAYERS, 1, exec.plan());
        float c4 = exec.estimate_cost(N_LAYERS, 4, exec.plan());
        float c16 = exec.estimate_cost(N_LAYERS, 16, exec.plan());
        print_partition_table(s.name, s.strategy, c1, c4, c16);
    }

    // ── Auto-balance results ──
    printf("\n── Auto-Balance Results ──\n");
    for (int batch : {1, 2, 4, 8, 16}) {
        FusedPipelineExecutor exec(PartitionStrategy::kAutoBalance);
        auto best = exec.auto_balance(N_LAYERS, batch);
        const char* name = "";
        switch (best) {
            case PartitionStrategy::kNPUOnly:       name = "NPU Only"; break;
            case PartitionStrategy::kGPUOnly:       name = "GPU Only"; break;
            case PartitionStrategy::kFFNOnNPU:      name = "FFN on NPU"; break;
            case PartitionStrategy::kQKVOnNPU:      name = "QKV on NPU"; break;
            case PartitionStrategy::kAttentionOnNPU: name = "Attn on NPU"; break;
            default: name = "?";
        }
        float cost = exec.estimate_cost(N_LAYERS, batch, exec.plan());
        printf("  Batch=%d: %s (%.3f ms)\n", batch, name, cost);
    }

    // ── Throughput analysis ──
    printf("\n── Throughput Estimation ──\n");
    printf("%-20s %12s %12s\n", "Strategy", "tok/s (M=1)", "tok/s (M=4)");
    printf("%s\n", std::string(46, '-').c_str());

    for (auto& s : strategies) {
        FusedPipelineExecutor exec(s.strategy);
        float cost1 = exec.estimate_cost(N_LAYERS, 1, exec.plan());
        float cost4 = exec.estimate_cost(N_LAYERS, 4, exec.plan());
        float tps1 = 1000.0f / cost1;
        float tps4 = 4000.0f / cost4;  // 4 tokens per step
        printf("%-20s %12.1f %12.1f\n", s.name, tps1, tps4);
    }

    // ── Sensitivity: NPU GEMM cost ──
    printf("\n── Sensitivity: NPU GEMM Cost Variation ──\n");
    printf("(FFN on NPU, M=1, 28 layers)\n");
    printf("%-20s %12s %12s\n", "NPU GEMM/FFN (ms)", "Total (ms)", "tok/s");
    printf("%s\n", std::string(46, '-').c_str());

    for (float gemm_cost : {0.1f, 0.2f, 0.3f, 0.5f, 0.8f, 1.0f}) {
        PartitionPlan plan;
        plan.qkv_backend = BackendType::kNPU;
        plan.attn_backend = BackendType::kGPU;
        plan.oproj_backend = BackendType::kNPU;
        plan.ffn_backend = BackendType::kNPU;
        plan.npu_gemm_cost_ms = gemm_cost;
        plan.gpu_attn_cost_ms = 0.5f;
        plan.enable_pipeline_overlap = true;

        FusedPipelineExecutor exec(PartitionStrategy::kFFNOnNPU);
        float cost = exec.estimate_cost(N_LAYERS, 1, plan);
        float tps = 1000.0f / cost;
        printf("%-20s %12.3f %12.1f\n", "", cost, tps);
    }

    // ── Sensitivity: GPU attention cost ──
    printf("\n── Sensitivity: GPU Attention Cost Variation ──\n");
    printf("(FFN on NPU, M=1, 28 layers)\n");
    printf("%-20s %12s %12s\n", "GPU Attn (ms)", "Total (ms)", "tok/s");
    printf("%s\n", std::string(46, '-').c_str());

    for (float attn_cost : {0.2f, 0.3f, 0.5f, 0.8f, 1.0f, 2.0f}) {
        PartitionPlan plan;
        plan.qkv_backend = BackendType::kNPU;
        plan.attn_backend = BackendType::kGPU;
        plan.oproj_backend = BackendType::kNPU;
        plan.ffn_backend = BackendType::kNPU;
        plan.npu_gemm_cost_ms = 0.3f;
        plan.gpu_attn_cost_ms = attn_cost;
        plan.enable_pipeline_overlap = true;

        FusedPipelineExecutor exec(PartitionStrategy::kFFNOnNPU);
        float cost = exec.estimate_cost(N_LAYERS, 1, plan);
        float tps = 1000.0f / cost;
        printf("%-20s %12.3f %12.1f\n", "", cost, tps);
    }

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
