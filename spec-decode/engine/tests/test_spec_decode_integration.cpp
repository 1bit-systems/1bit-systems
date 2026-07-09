// Integration test: all 5 Spec-Decode Engine modules working together
// (quantization, kernels, scheduler, model, pipeline)
#include "quantization/int8_quant.h"
#include "kernels/attention.h"
#include "scheduler/context_scheduler.h"
#include "model/model_registry.h"
#include "pipeline/fused_pipeline.h"
#include <cstdio>
#include <cassert>
#include <cmath>
#include <random>
#include <thread>

using namespace specdecode::quant;
using namespace specdecode::kernels;
using namespace specdecode::sched;
using namespace specdecode::model;
using namespace specdecode::pipeline;

int test_int8_with_model() {
    printf("=== INT8 Quantization with Model Config ===\n");

    auto cfg = Qwen3Config::config_0_6B();
    printf("  Model: %s\n", cfg.name.c_str());
    printf("  Hidden=%d, InterDim=%d\n", cfg.hidden_size, cfg.inter_dim);

    // Simulate weight quantization at QKV scale
    Int8QuantConfig qcfg;
    Int8WeightQuant quantizer(qcfg);

    std::vector<float> qkv_weights((size_t)cfg.qkv_proj_dim() * cfg.hidden_size);
    for (size_t i = 0; i < qkv_weights.size(); i++) {
        qkv_weights[i] = (float)(i % 100) / 100.0f - 0.5f;
    }

    auto packed = quantizer.quantize(qkv_weights, cfg.qkv_proj_dim(), cfg.hidden_size);
    printf("  QKV packed: %d x %d = %zu bytes\n", packed.rows, packed.cols, packed.size_bytes());
    printf("  Compression ratio: %.2fx\n",
           (float)qkv_weights.size() * 4.0f / (float)packed.size_bytes());

    assert(packed.rows == cfg.qkv_proj_dim());
    assert(packed.cols == cfg.hidden_size);

    std::vector<float> deq(packed.rows * packed.cols);
    Int8WeightQuant::dequantize(packed, deq);

    float max_err = 0.0f;
    for (size_t i = 0; i < qkv_weights.size(); i++) {
        float err = std::abs(deq[i] - qkv_weights[i]);
        if (err > max_err) max_err = err;
    }
    printf("  Max dequantization error: %.6f\n", max_err);
    assert(max_err < 0.01f);

    printf("  PASS\n");
    return 0;
}

int test_kernels_with_model_dims() {
    printf("=== Kernels with Model Dimensions ===\n");

    auto cfg = Qwen3Config::config_0_6B();

    // RMSNorm with model dimensions
    std::vector<float> x(cfg.hidden_size);
    std::vector<float> weight(cfg.hidden_size);
    std::vector<float> y(cfg.hidden_size);
    for (int i = 0; i < cfg.hidden_size; i++) {
        x[i] = (float)std::sin(i * 0.1f);
        weight[i] = 1.0f;
    }
    rms_norm(x, weight, y);

    double ss = 0.0;
    for (auto v : y) ss += (double)v * v;
    float mean_sq = (float)(ss / cfg.hidden_size);
    printf("  RMSNorm mean^2: %.6f (expected ~1.0)\n", mean_sq);
    assert(std::abs(mean_sq - 1.0f) < 0.1f);

    // Flash attention with model GQA
    AttentionConfig attn_cfg;
    attn_cfg.num_heads = cfg.num_heads;
    attn_cfg.num_kv_heads = cfg.num_kv_heads;
    attn_cfg.head_dim = cfg.head_dim;
    attn_cfg.gqa_ratio = cfg.gqa_ratio;
    attn_cfg.attn_scale = 1.0f / std::sqrt((float)cfg.head_dim);

    FlashAttention flash_attn(attn_cfg);

    std::vector<float> q((size_t)cfg.num_heads * cfg.head_dim, 1.0f);
    std::vector<float> k_cache((size_t)10 * cfg.num_kv_heads * cfg.head_dim, 0.5f);
    std::vector<float> v_cache((size_t)10 * cfg.num_kv_heads * cfg.head_dim, 0.5f);
    std::vector<float> attn_out((size_t)cfg.num_heads * cfg.head_dim);

    flash_attn.forward(q, k_cache, v_cache, 10, attn_out);

    for (auto v : attn_out) {
        assert(std::isfinite(v));
    }
    printf("  Flash attention (%d heads, %d KV, dim=%d): OK\n",
           attn_cfg.num_heads, attn_cfg.num_kv_heads, attn_cfg.head_dim);

    printf("  PASS\n");
    return 0;
}

int test_scheduler_with_model() {
    printf("=== Scheduler with Model Config ===\n");

    auto cfg = Qwen3Config::config_0_6B();

    KVCacheConfig kv_cfg;
    kv_cfg.num_layers = cfg.num_layers;
    kv_cfg.num_kv_heads = cfg.num_kv_heads;
    kv_cfg.head_dim = cfg.head_dim;
    kv_cfg.total_pages = 256;
    kv_cfg.page_size = 16;

    Scheduler sched(kv_cfg, 4);

    std::vector<std::vector<int32_t>> prompts = {
        {1, 2, 3, 4, 5},
        {10, 20, 30},
        {100, 200, 300, 400},
        {50, 60}
    };

    for (auto& p : prompts) {
        GenerationParams params;
        params.max_new_tokens = 128;
        sched.enqueue(p, params);
    }

    assert(sched.pending_count() == 4);

    int admitted = sched.admit_all();
    printf("  Admitted %d/%zu requests\n", admitted, prompts.size());
    assert(admitted == 4);

    auto prefill = sched.collect_prefill_batch();
    printf("  Prefill batch: %zu requests, %d total tokens\n",
           prefill.requests.size(), prefill.total_tokens);
    assert(prefill.requests.size() == 4);

    auto& pool = sched.kv_pool();
    printf("  KV pages: %d used / %d total\n", pool.used_pages(), pool.total_pages());
    assert(pool.used_pages() > 0);

    printf("  PASS\n");
    return 0;
}

int test_fused_pipeline_end_to_end() {
    printf("=== Fused Pipeline End-to-End ===\n");

    FusedPipelineExecutor exec(PartitionStrategy::kFFNOnNPU);
    int32_t num_layers = 28;
    auto stages = exec.build_pipeline(num_layers, 1, exec.plan());

    float cost = exec.estimate_cost(num_layers, 1, exec.plan());
    printf("  Pipeline: %zu stages, estimated %.2f ms\n", stages.size(), cost);

    float cost_batch4 = exec.estimate_cost(num_layers, 4, exec.plan());
    printf("  Batch=4 estimated: %.2f ms\n", cost_batch4);

    PartitionStrategy best = exec.auto_balance(num_layers, 1);
    printf("  Auto-best strategy for batch=1: %d\n", (int)best);

    PartitionStrategy best_batch4 = exec.auto_balance(num_layers, 4);
    printf("  Auto-best strategy for batch=4: %d\n", (int)best_batch4);

    printf("  PASS\n");
    return 0;
}

int test_all_modules_together() {
    printf("=== All 5 Modules Together ===\n");

    // 1. Load model config
    auto cfg = Qwen3Config::config_0_6B();
    printf("  [1] Model: %s\n", cfg.name.c_str());

    // 2. INT8 quantize weights
    Int8QuantConfig qcfg;
    Int8WeightQuant quantizer(qcfg);
    std::vector<float> weights((size_t)cfg.ffn_proj_dim() * cfg.hidden_size);
    for (size_t i = 0; i < weights.size(); i++)
        weights[i] = (float)(i % 200) / 200.0f - 0.5f;
    auto packed = quantizer.quantize(weights, cfg.ffn_proj_dim(), cfg.hidden_size);
    printf("  [2] INT8: %d x %d = %zu bytes (%.2fx compression)\n",
           packed.rows, packed.cols, packed.size_bytes(),
           (float)weights.size() * 4.0f / (float)packed.size_bytes());

    // 3. Scheduler with model kv config
    KVCacheConfig kv_cfg;
    kv_cfg.num_layers = cfg.num_layers;
    kv_cfg.num_kv_heads = cfg.num_kv_heads;
    kv_cfg.head_dim = cfg.head_dim;
    kv_cfg.total_pages = 128;

    Scheduler sched(kv_cfg, 4);
    GenerationParams gen_params;
    gen_params.max_new_tokens = 256;
    std::vector<int32_t> prompt = {1, 2, 3};
    sched.enqueue(prompt, gen_params);
    sched.admit_next();
    printf("  [3] Scheduler: 1 request admitted, %d KV pages used\n",
           sched.kv_pool().used_pages());

    // 4. Attention kernel with model config
    AttentionConfig attn_cfg;
    attn_cfg.num_heads = cfg.num_heads;
    attn_cfg.num_kv_heads = cfg.num_kv_heads;
    attn_cfg.head_dim = cfg.head_dim;
    attn_cfg.gqa_ratio = cfg.gqa_ratio;

    FlashAttention flash_attn(attn_cfg);
    std::vector<float> q((size_t)cfg.num_heads * cfg.head_dim, 0.5f);
    std::vector<float> k_cache((size_t)64 * cfg.num_kv_heads * cfg.head_dim, 0.3f);
    std::vector<float> v_cache((size_t)64 * cfg.num_kv_heads * cfg.head_dim, 0.3f);
    std::vector<float> out((size_t)cfg.num_heads * cfg.head_dim);
    flash_attn.forward(q, k_cache, v_cache, 64, out);

    bool all_finite = true;
    for (auto v : out) { if (!std::isfinite(v)) { all_finite = false; break; } }
    printf("  [4] Attention: 16H @ 8KV, seq=64, all_finite=%d\n", all_finite);
    assert(all_finite);

    // 5. Pipeline cost with model layers
    FusedPipelineExecutor exec(PartitionStrategy::kFFNOnNPU);
    float cost = exec.estimate_cost(cfg.num_layers, 1, exec.plan());
    float tps = 1000.0f / cost;
    printf("  [5] Pipeline: 28 layers, %.2f ms, ~%.0f tok/s\n", cost, tps);
    assert(cost > 0.0f);

    printf("  ✅ All 5 modules integrated successfully!\n");
    return 0;
}

int main() {
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Spec-Decode Engine v2 Integration Suite  ║\n");
    printf("║  INT8 | Scheduler | Kernels | Model | Pipe ║\n");
    printf("╚═══════════════════════════════════════════╝\n\n");

    int failures = 0;
    failures += test_int8_with_model();
    failures += test_kernels_with_model_dims();
    failures += test_scheduler_with_model();
    failures += test_fused_pipeline_end_to_end();
    failures += test_all_modules_together();

    printf("\n=== Integration Tests: %d failures ===\n", failures);

    if (failures == 0) {
        printf("✅ All integration tests passed!\n");
    }

    return failures;
}
