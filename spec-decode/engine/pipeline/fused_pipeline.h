#pragma once
// Fused NPU+GPU Pipeline — overlap NPU GEMM with GPU attention for maximum throughput.
//
// Architecture:
//   ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
//   │ Layer L:     │     │ Layer L+1:   │     │ Layer L+2:   │
//   │ NPU: QKV    ─┼─ ──▶│ NPU: QKV    ─┼─ ──▶│ ...          │
//   │ GPU: Attn   │     │ GPU: Attn    │     │              │
//   │ NPU: O proj │     │ NPU: O proj  │     │              │
//   │ NPU: FFN    │     │ NPU: FFN     │     │              │
//   └─────────────┘     └──────────────┘     └──────────────┘
//        │                    │
//        └── Data flow ──────┘
//
// Pipeline overlapping:
//   Step 1: NPU QKV for L0 ──────────────────────────┐
//   Step 2: GPU Attn L0 (overlaps with NPU QKV L1)   │
//   Step 3: NPU O_proj L0 + FFN L0 (overlaps with Attn L1)
//
// On Strix Halo (UMA): NPU BOs are directly accessible by GPU via dma-buf.
// No data copying needed between backends.

#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include <span>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstring>

namespace specdecode::pipeline {

// ─── Backend Abstraction ────────────────────────────────────────────────────

enum class BackendType : uint8_t {
    kCPU,
    kNPU,
    kGPU,
    kNPU_GPU_Fused,
};

struct BackendCapabilities {
    bool supports_int8_gemm = false;
    bool supports_fp16_gemm = false;
    bool supports_flash_attention = false;
    bool supports_paged_attention = false;
    bool has_unified_memory = false;  // Strix Halo UMA
    int64_t peak_flops = 0;
    int64_t memory_bandwidth = 0;
    int32_t max_batch_size = 1;
    bool requires_host_staging = true; // If no UMA, needs copy
};

// ─── Pipeline Stage Descriptors ─────────────────────────────────────────────

enum class StageType : uint8_t {
    kQKV,           // QKV projection + RoPE
    kAttention,     // Flash attention
    kOProj,         // Output projection
    kFFN,           // SwiGLU FFN
    kRMSNorm,       // RMS normalization
    kLMHead,        // LM head projection
};

struct StageDependency {
    int32_t stage_idx;
    bool requires_completion;
};

struct PipelineStage {
    StageType type;
    BackendType backend;
    int32_t layer_id;
    int32_t batch_size;

    // Timing (set by profiler)
    float estimated_cost_ms = 0.0f;
    float actual_cost_ms = 0.0f;

    // Dependencies (stages that must complete before this one)
    std::vector<StageDependency> dependencies;

    // Execution function: (input_data, output_data) -> void
    using ExecuteFn = std::function<void(std::span<float>, std::span<float>)>;
    ExecuteFn execute;

    // Resource handles (opaque pointers for each backend)
    void* backend_handle = nullptr;
};

// ─── Partition Strategy ─────────────────────────────────────────────────────

enum class PartitionStrategy : uint8_t {
    kNPUOnly,               // All layers on NPU (fallback)
    kGPUOnly,               // All layers on GPU (for testing)
    kFFNOnNPU,              // QKV + O + FFN on NPU, attention on GPU (recommended)
    kQKVOnNPU,              // Only QKV on NPU, rest on GPU
    kAttentionOnNPU,        // Only attention on NPU, FFN on GPU
    kAutoBalance,           // Profile-guided automatic partitioning
};

struct PartitionPlan {
    BackendType qkv_backend = BackendType::kNPU;
    BackendType attn_backend = BackendType::kGPU;
    BackendType oproj_backend = BackendType::kNPU;
    BackendType ffn_backend = BackendType::kNPU;
    BackendType norm_backend = BackendType::kCPU;

    bool enable_pipeline_overlap = true;
    int32_t pipeline_depth = 2;  // How many layers in flight simultaneously

    // For auto-balance mode
    float npu_gemm_cost_ms = 0.3f;   // Per-layer NPU GEMM cost
    float gpu_attn_cost_ms = 0.5f;   // Per-layer GPU attention cost
};

// ─── Cross-Backend Memory ───────────────────────────────────────────────────

struct CrossBackendBuffer {
    std::vector<float> host_data;      // Always valid (host mirror)
    void* npu_bo = nullptr;            // NPU BO handle (XRT)
    void* gpu_buffer = nullptr;        // GPU buffer handle (Vulkan)
    int32_t dma_buf_fd = -1;           // dma-buf fd for zero-copy (Strix Halo)
    size_t size_bytes = 0;
    bool is_dirty_host = false;
    bool is_dirty_npu = false;
    bool is_dirty_gpu = false;

    bool has_unified_access() const noexcept { return dma_buf_fd >= 0; }
};

class CrossBackendAllocator {
public:
    CrossBackendAllocator(bool has_unified_memory, size_t alignment = 256)
        : has_unified_memory_(has_unified_memory), alignment_(alignment) {}

    // Allocate a buffer accessible by all backends
    CrossBackendBuffer allocate(size_t size_bytes) {
        CrossBackendBuffer buf;
        buf.size_bytes = size_bytes;
        buf.host_data.resize(size_bytes / sizeof(float) + 1);
        // If UMA, we'd set up dma-buf here
        return buf;
    }

    // Synchronize host → NPU (copy or no-op for UMA)
    void sync_to_npu(CrossBackendBuffer& buf) {
        if (has_unified_memory_ && buf.dma_buf_fd >= 0) {
            // UMA: no copy needed, just flush cache
            buf.is_dirty_npu = true;
            buf.is_dirty_host = false;
            return;
        }
        // Non-UMA: copy to NPU BO
        buf.is_dirty_npu = true;
        buf.is_dirty_host = false;
    }

    // Synchronize NPU → GPU (zero-copy if UMA)
    void sync_npu_to_gpu(CrossBackendBuffer& buf) {
        if (has_unified_memory_ && buf.dma_buf_fd >= 0) {
            // Zero-copy: NPU wrote directly to shared memory, GPU reads same pages
            buf.is_dirty_gpu = true;
            return;
        }
        // Stage through host
        buf.is_dirty_gpu = true;
    }

    // Synchronize GPU → NPU (for next layer)
    void sync_gpu_to_npu(CrossBackendBuffer& buf) {
        if (has_unified_memory_ && buf.dma_buf_fd >= 0) {
            buf.is_dirty_npu = true;
            return;
        }
        buf.is_dirty_npu = true;
    }

    bool has_unified_memory() const noexcept { return has_unified_memory_; }

private:
    bool has_unified_memory_;
    size_t alignment_;
};

// ─── NPU+GPU Fused Pipeline Executor ───────────────────────────────────────

class FusedPipelineExecutor {
public:
    FusedPipelineExecutor(
        PartitionStrategy strategy = PartitionStrategy::kFFNOnNPU
    ) : strategy_(strategy) {
        plan_ = make_partition_plan(strategy);
        allocator_ = std::make_unique<CrossBackendAllocator>(
            true  // Strix Halo has UMA — will detect at runtime
        );
    }

    // Build pipeline stages for a single forward pass through all layers
    std::vector<PipelineStage> build_pipeline(
        int32_t num_layers,
        int32_t batch_size,
        const PartitionPlan& plan
    ) {
        std::vector<PipelineStage> stages;

        for (int32_t l = 0; l < num_layers; l++) {
            // Stage 0: RMSNorm (always on CPU/NPU)
            PipelineStage norm_stage;
            norm_stage.type = StageType::kRMSNorm;
            norm_stage.backend = plan.norm_backend;
            norm_stage.layer_id = l;
            norm_stage.batch_size = batch_size;
            norm_stage.estimated_cost_ms = 0.02f;
            if (!stages.empty()) {
                norm_stage.dependencies.push_back({(int32_t)stages.size() - 1, true});
            }
            stages.push_back(std::move(norm_stage));

            // Stage 1: QKV projection
            PipelineStage qkv_stage;
            qkv_stage.type = StageType::kQKV;
            qkv_stage.backend = plan.qkv_backend;
            qkv_stage.layer_id = l;
            qkv_stage.batch_size = batch_size;
            qkv_stage.estimated_cost_ms = plan.npu_gemm_cost_ms;
            qkv_stage.dependencies.push_back({(int32_t)stages.size() - 1, true});
            stages.push_back(std::move(qkv_stage));

            // Stage 2: Flash attention (can overlap with next layer's QKV)
            PipelineStage attn_stage;
            attn_stage.type = StageType::kAttention;
            attn_stage.backend = plan.attn_backend;
            attn_stage.layer_id = l;
            attn_stage.batch_size = batch_size;
            attn_stage.estimated_cost_ms = plan.gpu_attn_cost_ms;

            // Attention depends on QKV of same layer (for K,V) but NOT on
            // previous layer's attention if pipeline overlap is enabled
            attn_stage.dependencies.push_back({(int32_t)stages.size() - 1, true}); // QKV dep

            if (plan.enable_pipeline_overlap && l > 0) {
                // With overlap: attention L depends on QKV L (just done) and
                // ORuns in parallel with QKV L+1
                // No dependency on attention L-1's completion
            } else {
                // Without overlap: serial execution
                // (dependency chain handled by execution order)
            }
            stages.push_back(std::move(attn_stage));

            // Stage 3: Output projection
            PipelineStage oproj_stage;
            oproj_stage.type = StageType::kOProj;
            oproj_stage.backend = plan.oproj_backend;
            oproj_stage.layer_id = l;
            oproj_stage.batch_size = batch_size;
            oproj_stage.estimated_cost_ms = plan.npu_gemm_cost_ms;
            oproj_stage.dependencies.push_back({(int32_t)stages.size() - 1, true}); // Attn dep
            stages.push_back(std::move(oproj_stage));

            // Stage 4: Residual add + RMSNorm
            PipelineStage res_stage;
            res_stage.type = StageType::kRMSNorm;
            res_stage.backend = BackendType::kCPU;
            res_stage.layer_id = l;
            res_stage.batch_size = batch_size;
            res_stage.estimated_cost_ms = 0.01f;
            res_stage.dependencies.push_back({(int32_t)stages.size() - 1, true}); // OProj dep
            stages.push_back(std::move(res_stage));

            // Stage 5: SwiGLU FFN
            PipelineStage ffn_stage;
            ffn_stage.type = StageType::kFFN;
            ffn_stage.backend = plan.ffn_backend;
            ffn_stage.layer_id = l;
            ffn_stage.batch_size = batch_size;
            ffn_stage.estimated_cost_ms = plan.npu_gemm_cost_ms * 1.5f; // GU + D
            ffn_stage.dependencies.push_back({(int32_t)stages.size() - 1, true});
            stages.push_back(std::move(ffn_stage));
        }

        // Final LM head
        PipelineStage lm_stage;
        lm_stage.type = StageType::kLMHead;
        lm_stage.backend = BackendType::kCPU;
        lm_stage.layer_id = num_layers;
        lm_stage.batch_size = batch_size;
        lm_stage.dependencies.push_back({(int32_t)stages.size() - 1, true});
        stages.push_back(std::move(lm_stage));

        return stages;
    }

    // Execute a pipeline with stage overlap.
    // In stage-level execution (w/out actual hardware dispatch), this simulates
    // the dependency-aware ordering. With real backends, overlapping stages
    // run on separate threads.
    void execute_pipeline(
        std::vector<PipelineStage>& stages,
        std::span<float> input,
        std::span<float> output
    ) {
        // Topological sort by dependencies (already in order from build_pipeline)
        // For simulation: execute each stage, respecting dependencies
        std::vector<float> stage_output;

        for (auto& stage : stages) {
            // Check dependencies
            for (auto& dep : stage.dependencies) {
                if (dep.requires_completion) {
                    // In real execution: wait for dependency's completion signal
                }
            }

            // Execute
            if (stage.execute) {
                stage.execute(input, output);
            }

            // Record timing
            auto start = std::chrono::steady_clock::now();
            // ... execute ...
            auto end = std::chrono::steady_clock::now();
            stage.actual_cost_ms = std::chrono::duration<float, std::milli>(end - start).count();
        }
    }

    // Estimate total cost for a given strategy with batch-aware scaling
    float estimate_cost(
        int32_t num_layers,
        int32_t batch_size,
        const PartitionPlan& plan
    ) const {
        float total = 0.0f;

        // Batch-aware cost model:
        //   NPU: BW-bound at M=1, compute-bound at M>=64
        //   GPU: BW-bound at M=1, compute-bound at M>=128
        //   Attention: O(seq_len) BW-bound, WMMA makes it compute-bound at seq>=32
        // Norm: O(1) no matter the batch (element-wise)

        float cpu_norm = 0.02f;

        // NPU GEMM scales sub-linearly with batch:
        //   M=1:  0.3ms (GEMV, BW-bound, 100% of 560GB/s)
        //   M=8:  0.5ms (GEMM starts using more compute)
        //   M=64: 1.2ms (GEMM near compute-bound)
        //   M=128: 2.0ms (GEMM compute-bound at ~40 TFLOPS)
        float batch_factor = 1.0f + (float)(batch_size - 1) * 0.15f;
        float npu_gemm = plan.npu_gemm_cost_ms * batch_factor;

        // GPU attention scales sub-linearly with batch (tiled WMMA)
        //   seq=128 fixed, varying batch:
        //   M=1:  0.5ms (BW-bound)
        //   M=8:  0.6ms
        //   M=64: 1.0ms
        //   M=128: 1.5ms
        float gpu_attn = plan.gpu_attn_cost_ms +
                         (float)(batch_size - 1) * 0.008f;

        // FFN: NPU does Gate+Up+Down, 1.5× QKV cost
        float npu_ffn_cost = npu_gemm * 1.5f;

        float per_layer = 0.0f;
        if (plan.enable_pipeline_overlap) {
            // Overlap: NPU does QKV(L+1) while GPU does Attn(L)
            // Effective cost = max(NPU_cost, GPU_attn) + remaining
            float npu_path = npu_gemm + npu_ffn_cost;  // QKV + FFN
            float gpu_path = gpu_attn;                   // Attn

            // With pipeline: start QKV(L+1) while Attn(L) runs
            // Critical path = max(QKV+FFN, QKV+Attn+...)
            // Approx: max(NPU_per_layer, GPU_attn) + norm overhead
            per_layer = std::max(npu_path, gpu_path) + 2 * cpu_norm;
        } else {
            per_layer = npu_gemm + gpu_attn + npu_gemm + npu_ffn_cost + 2 * cpu_norm;
        }

        total = per_layer * num_layers;

        // LM head: scales with batch, large GEMM
        float lm_cost = 12.0f * (1.0f + (float)(batch_size - 1) * 0.01f);
        if (batch_size >= 8) lm_cost = 12.0f * (1.0f + std::log2((float)batch_size) * 0.1f);
        total += lm_cost;

        return total;
    }

    // Choose the best partition strategy based on estimated cost
    PartitionStrategy auto_balance(int32_t num_layers, int32_t batch_size) {
        float best_cost = std::numeric_limits<float>::max();
        PartitionStrategy best_strategy = PartitionStrategy::kFFNOnNPU;

        auto strategies = {
            PartitionStrategy::kNPUOnly,
            PartitionStrategy::kGPUOnly,
            PartitionStrategy::kFFNOnNPU,
            PartitionStrategy::kQKVOnNPU,
            PartitionStrategy::kAttentionOnNPU,
        };

        for (auto s : strategies) {
            auto plan = make_partition_plan(s);
            float cost = estimate_cost(num_layers, batch_size, plan);
            if (cost < best_cost) {
                best_cost = cost;
                best_strategy = s;
            }
        }

        return best_strategy;
    }

    // Get current partition plan
    const PartitionPlan& plan() const noexcept { return plan_; }

private:
    static PartitionPlan make_partition_plan(PartitionStrategy strategy) {
        PartitionPlan plan;
        switch (strategy) {
            case PartitionStrategy::kNPUOnly:
                plan.qkv_backend = BackendType::kNPU;
                plan.attn_backend = BackendType::kNPU;
                plan.oproj_backend = BackendType::kNPU;
                plan.ffn_backend = BackendType::kNPU;
                plan.enable_pipeline_overlap = false;
                break;
            case PartitionStrategy::kGPUOnly:
                plan.qkv_backend = BackendType::kGPU;
                plan.attn_backend = BackendType::kGPU;
                plan.oproj_backend = BackendType::kGPU;
                plan.ffn_backend = BackendType::kGPU;
                plan.enable_pipeline_overlap = false;
                break;
            case PartitionStrategy::kFFNOnNPU:
                plan.qkv_backend = BackendType::kNPU;
                plan.attn_backend = BackendType::kGPU;
                plan.oproj_backend = BackendType::kNPU;
                plan.ffn_backend = BackendType::kNPU;
                plan.enable_pipeline_overlap = true;
                break;
            case PartitionStrategy::kQKVOnNPU:
                plan.qkv_backend = BackendType::kNPU;
                plan.attn_backend = BackendType::kGPU;
                plan.oproj_backend = BackendType::kGPU;
                plan.ffn_backend = BackendType::kGPU;
                plan.enable_pipeline_overlap = true;
                break;
            case PartitionStrategy::kAttentionOnNPU:
                plan.qkv_backend = BackendType::kGPU;
                plan.attn_backend = BackendType::kNPU;
                plan.oproj_backend = BackendType::kGPU;
                plan.ffn_backend = BackendType::kGPU;
                plan.enable_pipeline_overlap = true;
                break;
            case PartitionStrategy::kAutoBalance:
                // Default to FFNOnNPU, will be resolved
                plan.attn_backend = BackendType::kGPU;
                plan.enable_pipeline_overlap = true;
                break;
        }
        return plan;
    }

    PartitionStrategy strategy_;
    PartitionPlan plan_;
    std::unique_ptr<CrossBackendAllocator> allocator_;
};

// ─── NPU ↔ GPU DMA-BUF Interop ─────────────────────────────────────────────
//
// On Strix Halo (AMD Ryzen AI Max), the NPU and iGPU share unified memory.
// NPU XRT BOs can be exported as dma-buf file descriptors and imported
// into Vulkan/ROCm for zero-copy access.

class NpuGpuInterop {
public:
    // Export an NPU buffer as a dma-buf fd
    // Returns fd on success, -1 on failure
    static int export_npu_buffer(void* xrt_bo_handle) {
        if (!xrt_bo_handle) return -1;

        // In production, calls xrtBOExport(bo) which returns a dma-buf fd
        // On Strix Halo with UMA, this is supported by the AMD XDNA driver.

        // For now, return -1 indicating no dma-buf support detected
        // (fallback to staging copy)
        return -1;
    }

    // Import a dma-buf fd into Vulkan for GPU access
    // Returns Vulkan buffer handle or nullptr
    static void* import_dma_buf_to_vulkan(int dma_buf_fd, size_t size) {
        if (dma_buf_fd < 0) return nullptr;

        // In production:
        // VkImportMemoryFdInfoKHR { sType, fd = dma_buf_fd }
        // vkAllocateMemory with VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
        // vkBindBufferMemory

        return nullptr;
    }

    // Check if UMA (unified memory access) is available
    static bool has_unified_memory() {
        // Check for Strix Halo / AMD APU with unified NPU+GPU memory
        // Read from /sys/class/drm/ or use xrt::device query
        // For now, assume true on supported hardware
        static const bool detected = []() {
            // Try to detect UMA capability
            FILE* f = fopen("/proc/cpuinfo", "r");
            if (!f) return false;
            char buf[4096];
            bool found = false;
            while (fgets(buf, sizeof(buf), f)) {
                if (strstr(buf, "AMD Ryzen AI") || strstr(buf, "Strix")) {
                    found = true;
                    break;
                }
            }
            fclose(f);
            return found;
        }();
        return detected;
    }
};

} // namespace specdecode::pipeline
