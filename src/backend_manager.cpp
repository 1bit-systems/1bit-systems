// backend_manager.cpp — Backend Manager implementation
// Windows ML-style backend orchestrator with auto-detection, selection, failover.

#include "backend_manager.h"
#include "backend_plugin.h"
#include "backend_detect.h"
#include "backend.h"
#include "model_router.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <thread>
#include <chrono>
#ifndef _WIN32
#include <unistd.h>
#else
// Windows: _S_IFMT/_S_IFREG for S_ISREG
#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif
#endif
#include <sys/stat.h>

// ── Backend priority by tier ──
static int tier_priority(BackendTier t) {
    switch (t) {
        case BackendTier::T1_ACCELERATOR: return 300;
        case BackendTier::T2_GPU:         return 200;
        case BackendTier::T3_CPU:         return 100;
        default: return 0;
    }
}

// ── Constructor ──
BackendManager::BackendManager() : monitor_() {}

BackendManager::~BackendManager() {
    destroy();
}

// ── Discover: probe hardware, enumerate backends ──
void BackendManager::discover() {
    std::lock_guard<std::mutex> lock(mtx_);
    backends_.clear();

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║   Backend Manager — Hardware Discovery   ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    // ── Probe each backend in priority order ──

    // 1. NPU (XRT) — lowest power, always preferred for inference
    {
        BackendInfo info;
        info.id = "npu_xrt";
        info.type = BackendType::NPU_XRT;
        info.tier = BackendTier::T1_ACCELERATOR;
        info.description = "AMD XDNA NPU via native worker engine";
        info.priority = tier_priority(info.tier) + 50;
        info.available = has_npu();
        info.functional = false;  // needs init to confirm
        // Uses backend_npu.cpp (worker subprocess protocol with
        // npu_engine_universal). This is the same verified path used
        // for all NPU inference — GEMM via pre-compiled xclbins,
        // attention via pre-compiled KV instructions, CPU fallback
        // for RoPE/norm/residual. Zero FLM dependency.
        info.auto_selectable = true;
        info.score = 0;
        info.total_inferences = 0;
        info.failed_inferences = 0;
        info.cumulative_ms = 0;
        info.instance = nullptr;
        info.plugin_handle = nullptr;
        printf("  %-25s %s\n", "NPU XDNA (XRT)", info.available ? "✅ detected" : "❌ not available");
        backends_.push_back(info);
    }

    // 1c. ZINC GPU — general GGUF, multi-architecture/multi-quant (see model_router.h;
    // this is the default GPU path for non-Zaya, non-qwen3 GGUF models).
    {
        BackendInfo info;
        info.id = "zinc_gpu";
        info.type = BackendType::ZINC_GPU;
        info.tier = BackendTier::T2_GPU;
        info.description = "ZINC GPU (Vulkan, multi-arch)";
        info.priority = tier_priority(info.tier) + 40;
#ifdef ZINC_DISABLED
        info.available = false;
#else
        info.available = has_vulkan() || has_hip_gpu();
#endif
        info.functional = false;
        info.score = 0;
        info.total_inferences = 0;
        info.failed_inferences = 0;
        info.cumulative_ms = 0;
        info.instance = nullptr;
        info.plugin_handle = nullptr;
        printf("  %-25s %s\n", "ZINC GPU (Vulkan)", info.available ? "✅ detected" : "❌ not available");
        backends_.push_back(info);
    }

    // 2. HIP GPU (ROCm) — highest throughput
    {
        BackendInfo info;
        info.id = "hip_gpu";
        info.type = BackendType::HIP_GPU;
        info.tier = BackendTier::T2_GPU;
        info.description = "AMD ROCm GPU via HIP";
        info.priority = tier_priority(info.tier) + 50;
        info.available = has_hip_gpu();
        info.functional = false;
        info.score = 0;
        info.total_inferences = 0;
        info.failed_inferences = 0;
        info.cumulative_ms = 0;
        info.instance = nullptr;
        info.plugin_handle = nullptr;
        printf("  %-25s %s\n", "HIP GPU (ROCm)", info.available ? "✅ detected" : "❌ not available");
        backends_.push_back(info);
    }

    // 2b. Mamba1 GPU — Mamba1 SSM + MoE HIP kernels (Zamba-7B-v1, BlackMamba)
    // Shares HIP availability; created on-demand by architecture.
    {
        BackendInfo info;
        info.id = "mamba1_gpu";
        info.type = BackendType::HIP_GPU;
        info.tier = BackendTier::T2_GPU;
        info.description = "AMD ROCm GPU via Mamba1 HIP kernels";
        info.priority = tier_priority(info.tier) + 49;  // just below general HIP
        info.available = has_hip_gpu();
        info.functional = false;
        info.score = 0;
        info.total_inferences = 0;
        info.failed_inferences = 0;
        info.cumulative_ms = 0;
        info.instance = nullptr;
        info.plugin_handle = nullptr;
        printf("  %-25s %s\n", "Mamba1 GPU (Mamba1 HIP)", info.available ? "✅ detected" : "❌ not available");
        backends_.push_back(info);
    }

    // 2c. Zamba2 GPU — Mamba2 hybrid SSD kernels (Zamba2-1.2B/2.7B/7B)
    // Shares HIP availability; created on-demand by architecture.
    {
        BackendInfo info;
        info.id = "zamba2_gpu";
        info.type = BackendType::HIP_GPU;
        info.tier = BackendTier::T2_GPU;
        info.description = "AMD ROCm GPU via Mamba2 SSD kernels";
        info.priority = tier_priority(info.tier) + 48;  // just below mamba1_gpu
        info.available = has_hip_gpu();
        info.functional = false;
        info.score = 0;
        info.total_inferences = 0;
        info.failed_inferences = 0;
        info.cumulative_ms = 0;
        info.instance = nullptr;
        info.plugin_handle = nullptr;
        printf("  %-25s %s\n", "Zamba2 GPU (Mamba2 HIP)", info.available ? "✅ detected" : "❌ not available");
        backends_.push_back(info);
    }

    // 3. Vulkan GPU — portable fallback (runs on any GPU vendor)
    {
        BackendInfo info;
        info.id = "vulkan_gpu";
        info.type = BackendType::VULKAN;
        info.tier = BackendTier::T2_GPU;
        info.description = "Portable Vulkan GPU";
        info.priority = tier_priority(info.tier) + 30;
        info.available = has_vulkan();
        info.functional = false;
        info.score = 0;
        info.total_inferences = 0;
        info.failed_inferences = 0;
        info.cumulative_ms = 0;
        info.instance = nullptr;
        info.plugin_handle = nullptr;
        printf("  %-25s %s\n", "Vulkan GPU", info.available ? "✅ detected" : "❌ not available");
        backends_.push_back(info);
    }

    // 4. CPU AVX-512
    {
        BackendInfo info;
        info.id = "cpu_avx512";
        info.type = BackendType::CPU_AVX512;
        info.tier = BackendTier::T3_CPU;
        info.description = "CPU with AVX-512";
        info.priority = tier_priority(info.tier) + 30;
        info.available = has_avx512();
        info.functional = false;
        info.score = 0;
        info.total_inferences = 0;
        info.failed_inferences = 0;
        info.cumulative_ms = 0;
        info.instance = nullptr;
        info.plugin_handle = nullptr;
        printf("  %-25s %s\n", "CPU AVX-512", info.available ? "✅ detected" : "❌ not available");
        backends_.push_back(info);
    }

    // 5. CPU scalar — always available (the safety net)
    {
        BackendInfo info;
        info.id = "cpu_scalar";
        info.type = BackendType::CPU_SCALAR;
        info.tier = BackendTier::T3_CPU;
        info.description = "CPU (portable scalar fallback)";
        info.priority = tier_priority(info.tier) + 10;
        info.available = true;  // always
        info.functional = false;
        info.score = 0;
        info.total_inferences = 0;
        info.failed_inferences = 0;
        info.cumulative_ms = 0;
        info.instance = nullptr;
        info.plugin_handle = nullptr;
        printf("  %-25s %s\n", "CPU (scalar)", "✅ always available");
        backends_.push_back(info);
    }

    // 6. CPU generic — general-purpose GGUF backend (Llama/Mistral/Qwen2/Gemma/Phi),
    // unlike cpu_avx512/cpu_scalar (CPUBackend) which only accept the hardcoded
    // Zaya1-8B dims. Always available; the router (model_router.h) is what actually
    // prefers this for non-Zaya models — see select_backend_route().
    {
        BackendInfo info;
        info.id = "cpu_generic";
        info.type = BackendType::GENERIC;
        info.tier = BackendTier::T3_CPU;
        info.description = "Generic CPU (GGUF)";
        info.priority = tier_priority(info.tier) + 20;
        info.available = true;  // always
        info.functional = false;
        info.score = 0;
        info.total_inferences = 0;
        info.failed_inferences = 0;
        info.cumulative_ms = 0;
        info.instance = nullptr;
        info.plugin_handle = nullptr;
        printf("  %-25s %s\n", "CPU Generic (GGUF)", "✅ always available");
        backends_.push_back(info);
    }

    // Rack 'em
    rank_backends();
    active_idx_ = 0;

    printf("\n  %zu backend(s) discovered.\n", backends_.size());
    printf("  Primary: %s\n\n", backends_.empty() ? "none" : backends_[0].id.c_str());
}

// ── Init: create selected backend, load weights ──
bool BackendManager::init(const ModelConfig& cfg, const std::string& weights_dir) {
    std::lock_guard<std::mutex> lock(mtx_);
    cfg_ = cfg;
    weights_dir_ = weights_dir;

    if (backends_.empty()) {
        fprintf(stderr, "BackendManager: no backends discovered. Run discover() first.\n");
        return false;
    }

    // Validate model_path exists and is a regular file before passing to backends
    // (prevents arch-specific backends from crashing when given a directory instead of a file)
    if (!cfg.model_path.empty()) {
        struct stat st;
        if (stat(cfg.model_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
            fprintf(stderr, "BackendManager: model_path '%s' is not a regular file — clearing\n", cfg.model_path.c_str());
            cfg_.model_path.clear();
        }
    }

    std::vector<size_t> order(backends_.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    return init_in_order(cfg, weights_dir, order);
}

bool BackendManager::init(const ModelConfig& cfg, const std::string& weights_dir,
                           const std::vector<std::string>& preferred_ids) {
    std::lock_guard<std::mutex> lock(mtx_);
    cfg_ = cfg;
    weights_dir_ = weights_dir;

    if (backends_.empty()) {
        fprintf(stderr, "BackendManager: no backends discovered. Run discover() first.\n");
        return false;
    }

    // Validate model_path exists and is a regular file before passing to backends
    if (!cfg.model_path.empty()) {
        struct stat st;
        if (stat(cfg.model_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
            fprintf(stderr, "BackendManager: model_path '%s' is not a regular file — clearing\n", cfg.model_path.c_str());
            cfg_.model_path.clear();
        }
    }

    // Preferred ids first (in the order given), then everything else in the
    // usual priority order. Unknown preferred ids are silently skipped.
    std::vector<size_t> order;
    std::vector<bool> used(backends_.size(), false);
    for (const auto& id : preferred_ids) {
        for (size_t i = 0; i < backends_.size(); i++) {
            if (!used[i] && backends_[i].id == id) {
                order.push_back(i);
                used[i] = true;
                break;
            }
        }
    }
    for (size_t i = 0; i < backends_.size(); i++) if (!used[i]) order.push_back(i);

    return init_in_order(cfg, weights_dir, order);
}

bool BackendManager::init_in_order(const ModelConfig& cfg, const std::string& weights_dir,
                                    const std::vector<size_t>& order) {
    // Try each backend in the given order until one initializes.
    for (size_t idx : order) {
        auto& info = backends_[idx];
        if (!info.available || !info.auto_selectable) continue;

        printf("BackendManager: trying %s (%s)...\n", info.id.c_str(), info.description.c_str());
        // Try to create via dlsym (GPU/NPU backends live in librocm_cpp.so or standalone)
        // CPU backend is linked directly
        auto* raw = create_instance_rt(info);
        if (!raw) {
            printf("  → creation failed\n");
            continue;
        }
        info.instance = std::shared_ptr<Backend>(raw);

        if (info.instance->init(cfg, weights_dir)) {
            if (!info.instance->can_infer()) {
                // Detected and initialized, but cannot actually run inference
                // (e.g. the NPU stub). Report as available but not selectable (fixes #82).
                printf("  → ⚠️  detected, but not inference-capable (can_infer()==false) — not selectable\n");
                info.functional = false;
                destroy_instance(info);
                continue;
            }
            info.functional = true;
            info.instance->reset();
            initialized_ = true;
            // Set active_idx_ to this backend so generate() works immediately
            // without requiring the caller to manually call select_backend() (#fix #17).
            active_idx_ = idx;
            printf("  → ✅ initialized successfully\n");

            // Create monitor entry
            auto* pm = monitor_.for_backend(info.id);
            if (pm) pm->healthy = true;

            // Initialize cross-layer prefetch pilot
            // NOTE: Disabled pending investigation of heap corruption (issue #932).
            // The worker thread calling preload_layer() may race with httplib
            // completion handlers. Re-enable when the root cause is found.
            // if (raw) {
            //     raw->set_pilot(&pilot_);
            //     pilot_.init(cfg.num_layers, info.type,
            //         [raw](int layer, PilotBackend pb) -> bool {
            //             return raw->preload_layer(layer);
            //         });
            //     pilot_.start_worker();
            //     pilot_active_ = true;
            //     printf("  → PILOT prefetch active (%d layers)\n", cfg.num_layers);
            // }

            return true;
        }

        // Init failed — destroy and move on
        fprintf(stderr, "  → ❌ init failed\n");
        destroy_instance(info);
    }

    fprintf(stderr, "BackendManager: no backends could initialize!\n");
    return false;
}

// ── Select active backend by strategy ──
bool BackendManager::select_best() {
    std::lock_guard<std::mutex> lock(mtx_);

    switch (strategy_) {
        case SelectionStrategy::FASTEST: {
            // Pick the backend with the best (lowest) benchmark score
            size_t best_idx = backends_.size();
            float best_score = 1e30f;
            for (size_t i = 0; i < backends_.size(); i++) {
                auto& b = backends_[i];
                if (!b.available || !b.functional || !b.auto_selectable) continue;
                if (b.score > 0 && b.score < best_score) {
                    best_score = b.score;
                    best_idx = i;
                }
            }
            // If no backend has a score, fall back to first available+functional
            if (best_idx == backends_.size()) {
                for (size_t i = 0; i < backends_.size(); i++) {
                    if (backends_[i].available && backends_[i].functional && backends_[i].auto_selectable) {
                        best_idx = i;
                        break;
                    }
                }
            }
            if (best_idx < backends_.size()) {
                active_idx_ = best_idx;
                printf("BackendManager: selected %s (%.1f ms/tok)\n",
                       backends_[best_idx].id.c_str(), backends_[best_idx].score);
                return true;
            }
            return false;
        }

        case SelectionStrategy::LOWEST_POWER: {
            // Pick the highest-priority available+functional backend (NPU > GPU > CPU)
            for (size_t i = 0; i < backends_.size(); i++) {
                if (backends_[i].available && backends_[i].functional && backends_[i].auto_selectable) {
                    active_idx_ = i;
                    return true;
                }
            }
            return false;
        }

        case SelectionStrategy::ROUND_ROBIN: {
            // Cycle to the next available+functional backend
            size_t start = (active_idx_ + 1) % backends_.size();
            for (size_t i = 0; i < backends_.size(); i++) {
                size_t idx = (start + i) % backends_.size();
                if (backends_[idx].available && backends_[idx].functional && backends_[idx].auto_selectable) {
                    active_idx_ = idx;
                    return true;
                }
            }
            return false;
        }

        case SelectionStrategy::MANUAL:
        default:
            // Don't auto-change — user controls it via select_backend()
            if (active_idx_ < backends_.size() &&
                backends_[active_idx_].available && backends_[active_idx_].functional)
                return true;
            // Fallback to first available
            for (size_t i = 0; i < backends_.size(); i++) {
                if (backends_[i].available && backends_[i].functional) {
                    active_idx_ = i;
                    return true;
                }
            }
            return false;
    }
}

bool BackendManager::select_backend(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (size_t i = 0; i < backends_.size(); i++) {
        if (backends_[i].id == id && backends_[i].available && backends_[i].functional) {
            active_idx_ = i;
            strategy_ = SelectionStrategy::MANUAL;
            printf("BackendManager: manually selected %s\n", id.c_str());
            return true;
        }
    }
    fprintf(stderr, "BackendManager: backend '%s' not found or not functional\n", id.c_str());
    return false;
}

bool BackendManager::select_backend(BackendType type) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (size_t i = 0; i < backends_.size(); i++) {
        if (backends_[i].type == type && backends_[i].available && backends_[i].functional) {
            active_idx_ = i;
            strategy_ = SelectionStrategy::MANUAL;
            printf("BackendManager: manually selected %s (%s)\n",
                   backends_[i].id.c_str(), backend_name(type));
            return true;
        }
    }
    fprintf(stderr, "BackendManager: backend type %d not found\n", (int)type);
    return false;
}

Backend* BackendManager::active_backend() {
    if (active_idx_ >= backends_.size()) return nullptr;
    return backends_[active_idx_].instance.get();
}

const BackendInfo* BackendManager::active_info() const {
    if (active_idx_ >= backends_.size()) return nullptr;
    return &backends_[active_idx_];
}

// ── Inference with failover ──
int BackendManager::generate(int token_id) {
    if (!initialized_ || backends_.empty()) return -1;

    // Phase 1: snapshot under lock (shared_ptr keeps Backend alive even if
    // destroy() runs on another thread while we release the lock in Phase 2).
    // Capture active_idx_ so Phase 3 stats update goes to the backend that
    // actually ran the inference, not whatever select_backend() may have
    // switched to in the meantime (issue #357).
    std::shared_ptr<Backend> snap;
    std::shared_ptr<std::mutex> compute_mtx;
    size_t snap_idx = 0;
    bool need_failover = false;
    size_t prev_idx = 0;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (active_idx_ >= backends_.size()) return -1;
        snap_idx = active_idx_;
        auto& info = backends_[active_idx_];
        if (info.functional && info.instance) {
            snap = info.instance;  // shared_ptr copy — keeps Backend alive
            compute_mtx = info.compute_mtx;
        } else {
            need_failover = true;
            prev_idx = active_idx_;
        }
    }

    // Phase 2: inference WITHOUT mtx_ — snap keeps the Backend alive.
    // compute_mtx IS held here: it serializes against health_check()'s
    // reset() and benchmark_all()'s benchmark()/init() on this same
    // instance, which mtx_ alone can't do since it's released for this call.
    if (snap) {
        std::lock_guard<std::mutex> compute_lock(*compute_mtx);
        auto t0 = std::chrono::high_resolution_clock::now();
        int result = -1;
        // A backend that throws (e.g. a missing Vulkan shader, a HIP fault)
        // must NOT take the whole server down — treat it as a failed
        // inference and let the failover path below pick another backend.
        try {
            result = snap->generate(token_id);
        } catch (const std::exception& e) {
            fprintf(stderr, "BackendManager: %s threw during generate() (%s) — failing over\n",
                    snap_idx < backends_.size() ? backends_[snap_idx].id.c_str() : "?", e.what());
            result = -1;
        } catch (...) {
            fprintf(stderr, "BackendManager: backend threw an unknown exception during generate() — failing over\n");
            result = -1;
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        if (result >= 0) {
            // Fast path — re-acquire lock briefly for stats.
            // Use snap_idx (captured in Phase 1) instead of active_idx_
            // to avoid attributing stats to a backend that was switched
            // in by another thread (issue #357).
            std::lock_guard<std::mutex> lock(mtx_);
            if (snap_idx < backends_.size()) {
                auto* info = &backends_[snap_idx];
                info->total_inferences++;
                info->cumulative_ms += ms;
                monitor_.record(info->id, ms, true);
                // Update running score as exponential moving average
                // so re_evaluate() can use live performance data
                float ema = (info->score > 0)
                    ? 0.9f * info->score + 0.1f * ms
                    : ms;
                info->score = ema;
            }
            return result;
        }

        // Failed — re-acquire lock for stats + failover.
        // Use snap_idx (captured in Phase 1) — not active_idx_ — for the
        // same thread-safety reason (issue #357).
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (snap_idx < backends_.size()) {
                auto* info = &backends_[snap_idx];
                info->failed_inferences++;
                info->functional = false;
                monitor_.record(info->id, ms, false);
                monitor_.record_failure(info->id, "generate() returned -1");
            }
            need_failover = true;
            prev_idx = snap_idx;
        }
    }

    // Phase 3: failover under lock
    if (need_failover) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (failover()) {
            auto* info = &backends_[active_idx_];
            monitor_.record_fallback(backends_[prev_idx < backends_.size() ? prev_idx : 0].id, info->id);
            printf("BackendManager: failed over to %s\n", info->id.c_str());
            auto t0 = std::chrono::high_resolution_clock::now();
            int result = -1;
            try {
                result = info->instance->generate(token_id);
            } catch (const std::exception& e) {
                fprintf(stderr, "BackendManager: failover backend %s also threw (%s)\n", info->id.c_str(), e.what());
            } catch (...) {
                fprintf(stderr, "BackendManager: failover backend %s threw unknown exception\n", info->id.c_str());
            }
            float ms = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            if (result >= 0) {
                info->total_inferences++;
                info->cumulative_ms += ms;
                monitor_.record(info->id, ms, true);
                return result;
            }
            info->failed_inferences++;
            info->functional = false;
            monitor_.record(info->id, ms, false);
        }
    }

    fprintf(stderr, "BackendManager: ALL BACKENDS FAILED\n");
    return -1;
}

bool BackendManager::forward(int token_id, float* hidden_out) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto* b = active_backend();
    if (!b || !initialized_) return false;

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = b->forward(token_id, hidden_out);
    float ms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    if (active_idx_ < backends_.size()) {
        auto* info = &backends_[active_idx_];
        info->total_inferences++;
        info->cumulative_ms += ms;
        monitor_.record(info->id, ms, ok);
        if (!ok) {
            info->failed_inferences++;
            monitor_.record_failure(info->id, "forward() returned false");
        }
    }
    return ok;
}

bool BackendManager::lm_head(const float* hidden, float* logits, int* argmax) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto* b = active_backend();
    if (!b || !initialized_) return false;
    return b->lm_head(hidden, logits, argmax);
}

bool BackendManager::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    auto* b = active_backend();
    if (!b) return false;
    bool ok = b->reset();
    if (ok && active_idx_ < backends_.size()) {
        backends_[active_idx_].functional = true;
    }
    // Reset pilot for new sequence
    pilot_.reset();
    return ok;
}

// ── Failover ──
void BackendManager::set_fallback_policy(FallbackPolicy p) {
    fallback_policy_ = p;
}

FallbackPolicy BackendManager::fallback_policy() const {
    return fallback_policy_;
}

bool BackendManager::failover() {
    if (fallback_policy_ == FallbackPolicy::NONE) return false;

    // Mark current as failed
    if (active_idx_ < backends_.size()) {
        backends_[active_idx_].functional = false;
    }

    // Try each remaining backend in priority order
    for (size_t i = 0; i < backends_.size(); i++) {
        size_t idx = (active_idx_ + 1 + i) % backends_.size();
        if (idx == active_idx_) continue;
        auto& info = backends_[idx];
        if (!info.available) continue;

        // Create and init on-demand
        if (!info.instance) {
            auto* raw = create_instance_rt(info);
            if (!raw) continue;
            info.instance = std::shared_ptr<Backend>(raw);
            if (!info.instance->init(cfg_, weights_dir_)) {
                destroy_instance(info);
                continue;
            }
            if (!info.instance->can_infer()) continue;  // stub backend, not selectable (#82)
            info.functional = true;  // newly created + initialized → selectable (fixes #78)
        }

        if (info.instance && info.functional) {
            active_idx_ = idx;
            fallback_idx_ = (idx + 1) % backends_.size();
            info.instance->reset();
            return true;
        }
    }

    return false;
}

// ── Health ──
bool BackendManager::health_check() {
    std::lock_guard<std::mutex> lock(mtx_);
    auto* b = active_backend();
    if (!b) return false;
    if (!b->can_infer()) {  // stub backend is never healthy (fixes #82)
        if (active_idx_ < backends_.size()) backends_[active_idx_].functional = false;
        return false;
    }

    if (active_idx_ < backends_.size()) {
        auto& info = backends_[active_idx_];
        // Simple probe: try reset. Hold compute_mtx — reset() mutates the
        // same instance state (e.g. HIPBackend's ZayaState/pos) that a
        // concurrent generate() call may be using via Phase 2's lock-free path.
        bool ok;
        {
            std::lock_guard<std::mutex> compute_lock(*info.compute_mtx);
            ok = b->reset();
        }
        info.functional = ok;
        auto* pm = monitor_.for_backend(info.id);
        if (pm) pm->healthy = ok;
        return ok;
    }
    return false;
}

void BackendManager::monitor() {
    // health_check acquires its own lock; failover needs us to hold the lock.
    if (!health_check()) {
        std::lock_guard<std::mutex> lock(mtx_);
        fprintf(stderr, "BackendManager: health check failed, failing over...\n");
        failover();
    }
}

// ── Benchmarking ──
void BackendManager::benchmark_all(int tokens) {
    std::lock_guard<std::mutex> lock(mtx_);
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║   Backend Manager — Benchmark Suite      ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    // Only speculatively init()/benchmark() a backend if the router
    // considers it compatible with the currently loaded model's architecture
    // (same table BackendManager::init() used to pick the active backend),
    // or it already has a live instance (proved compatible by successfully
    // init'ing already). The generic CPU tier is architecture-agnostic by
    // design, so it's always eligible regardless of what the router lists.
    // Without this, benchmark_all() would freely try e.g. the Zaya HIP
    // kernels or the Zamba2 kernels against a Mamba1 model's weights —
    // neither backend's init()/benchmark() is hardened against that, and
    // both crashed with a real SIGSEGV (not a catchable C++ exception) when
    // this was reproduced under gdb: an OOB vector read in zaya_destroy()
    // and a segfault in mamba2_cpu_forward(), both on the agent-watchdog
    // thread, both taking the whole process down with them.
    auto route = select_backend_route(cfg_);
    auto architecture_compatible = [&](const BackendInfo& info) {
        if (info.tier == BackendTier::T3_CPU) return true;
        if (info.instance) return true;
        return std::find(route.backend_ids_in_order.begin(),
                          route.backend_ids_in_order.end(),
                          info.id) != route.backend_ids_in_order.end();
    };

    for (auto& info : backends_) {
        if (!info.available) continue;

        if (!architecture_compatible(info)) {
            printf("  %s... ⏭️  (skipped — not compatible with loaded model architecture)\n", info.id.c_str());
            continue;
        }

        // Skip CPU_SCALAR if CPU_AVX512 already benchmarked
        // (they share the same CPUBackend code, only the above is meaningful)
        if (info.type == BackendType::CPU_SCALAR &&
            std::any_of(backends_.begin(), backends_.end(), [](auto& b) {
                return b.type == BackendType::CPU_AVX512 && b.score > 0;
            })) {
            info.score = 9999;
            continue;
        }

        printf("  %s... ", info.id.c_str());
        fflush(stdout);

        // Hold compute_mtx for init()/benchmark() below — if info.instance
        // is already live (e.g. this is the currently-active backend), it
        // may be in concurrent use via generate()'s lock-free Phase 2; this
        // serializes against that instead of racing on shared instance state.
        std::lock_guard<std::mutex> compute_lock(*info.compute_mtx);

        // Create instance if needed. init()/benchmark() may THROW (missing
        // Vulkan shader, driver fault, OOM) — a broken backend must be skipped,
        // never allowed to std::terminate the whole server.
        if (!info.instance) {
            auto* raw = create_instance_rt(info);
            if (!raw) {
                printf("❌ (creation failed)\n");
                continue;
            }
            info.instance = std::shared_ptr<Backend>(raw);
            bool init_ok = false;
            try {
                init_ok = info.instance->init(cfg_, weights_dir_);
            } catch (const std::exception& e) {
                printf("❌ (init threw: %s)\n", e.what());
            } catch (...) {
                printf("❌ (init threw unknown exception)\n");
            }
            if (!init_ok) {
                destroy_instance(info);
                continue;
            }
        }

        float ms;
        try {
            ms = info.instance->benchmark(tokens);
        } catch (const std::exception& e) {
            printf("❌ (benchmark threw: %s — skipping backend)\n", e.what());
            info.available = false; info.functional = false;
            destroy_instance(info);
            continue;
        } catch (...) {
            printf("❌ (benchmark threw — skipping backend)\n");
            info.available = false; info.functional = false;
            destroy_instance(info);
            continue;
        }
        info.score = ms;
        info.functional = true;  // benchmarked and ready to use
        printf("%.1f ms/tok\n", ms);
    }

    rank_backends();
    printf("\n  Rankings:\n");
    for (size_t i = 0; i < backends_.size(); i++) {
        if (backends_[i].available && backends_[i].score > 0) {
            printf("    %zu. %s — %.1f ms/tok\n",
                   i + 1, backends_[i].id.c_str(), backends_[i].score);
        }
    }
    printf("\n");
}

void BackendManager::set_score(const std::string& id, float ms) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& info : backends_) {
        if (info.id == id) {
            info.score = ms;
            info.functional = true;
            return;
        }
    }
}

// ── Re-evaluate: check if a better backend is available per strategy ──
bool BackendManager::re_evaluate() {
    std::lock_guard<std::mutex> lock(mtx_);
    size_t old_idx = active_idx_;
    std::string old_id = (old_idx < backends_.size())
        ? backends_[old_idx].id : "none";

    // Keep the list sorted per current strategy, then re-select.
    // Use unlocked helper: rank_backends sorts backends_ in-place.
    // We hold the lock, so this is safe.
    // Temporarily save/restore active_idx_ because the sort may move it.
    rank_backends();

    // Find the best backend per strategy
    size_t best_idx = backends_.size();

    switch (strategy_) {
        case SelectionStrategy::FASTEST: {
            float best_score = 1e30f;
            for (size_t i = 0; i < backends_.size(); i++) {
                auto& b = backends_[i];
                if (!b.available || !b.functional) continue;
                if (b.score > 0 && b.score < best_score) {
                    best_score = b.score;
                    best_idx = i;
                }
            }
            if (best_idx == backends_.size()) {
                // No scored backends — pick first available+functional
                for (size_t i = 0; i < backends_.size(); i++) {
                    if (backends_[i].available && backends_[i].functional) {
                        best_idx = i;
                        break;
                    }
                }
            }
            break;
        }

        case SelectionStrategy::LOWEST_POWER: {
            for (size_t i = 0; i < backends_.size(); i++) {
                if (backends_[i].available && backends_[i].functional) {
                    best_idx = i;
                    break;
                }
            }
            break;
        }

        case SelectionStrategy::ROUND_ROBIN: {
            // Find the current backend's position in the sorted list, then
            // pick the next available+functional one.
            size_t current_pos = backends_.size();
            for (size_t i = 0; i < backends_.size(); i++) {
                if (backends_[i].id == old_id) {
                    current_pos = i;
                    break;
                }
            }
            for (size_t i = 1; i <= backends_.size(); i++) {
                size_t idx = (current_pos + i) % backends_.size();
                if (backends_[idx].available && backends_[idx].functional) {
                    best_idx = idx;
                    break;
                }
            }
            if (best_idx == backends_.size()) {
                for (size_t i = 0; i < backends_.size(); i++) {
                    if (backends_[i].available && backends_[i].functional) {
                        best_idx = i;
                        break;
                    }
                }
            }
            break;
        }

        case SelectionStrategy::MANUAL:
        default:
            // Don't auto-change
            if (old_idx < backends_.size() &&
                backends_[old_idx].available && backends_[old_idx].functional)
                best_idx = old_idx;
            break;
    }

    if (best_idx < backends_.size()) {
        active_idx_ = best_idx;
        if (best_idx != old_idx || backends_[best_idx].id != old_id) {
            printf("BackendManager: re-evaluated → switched %s → %s (%.1f ms/tok)\n",
                   old_id.c_str(),
                   backends_[best_idx].id.c_str(),
                   backends_[best_idx].score);
            return true;
        }
    }

    return false;
}

const BackendInfo* BackendManager::best_for_tier(BackendTier tier) const {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& info : backends_) {
        if (info.tier == tier && info.available && info.functional)
            return &info;
    }
    return nullptr;
}

// ── Plugins ──
bool BackendManager::load_plugin(const std::string& so_path) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::string error;
    auto* loader = BackendPluginLoader::load(so_path, &error);
    if (!loader) {
        fprintf(stderr, "BackendManager: plugin load failed: %s\n", error.c_str());
        return false;
    }

    BackendPlugin plugin;
    plugin.path = so_path;
    plugin.id = loader->id();
    plugin.version = loader->version();
    plugin.loaded = true;
    plugins_.push_back(plugin);

    // Instantiate and add to backends list
    Backend* instance = loader->instantiate();
    if (!instance) {
        fprintf(stderr, "BackendManager: plugin %s instantiation failed\n", plugin.id.c_str());
        delete loader;
        return false;
    }

    BackendInfo info;
    info.id = plugin.id;
    info.type = loader->type();
    info.tier = (info.type == BackendType::NPU_XRT) ? BackendTier::T1_ACCELERATOR
               : (info.type == BackendType::HIP_GPU || info.type == BackendType::VULKAN)
                 ? BackendTier::T2_GPU : BackendTier::T3_CPU;
    info.description = loader->description();
    info.priority = tier_priority(info.tier);
    info.available = true;
    info.functional = false;
    info.score = 0;
    info.instance = std::shared_ptr<Backend>(instance);
    info.plugin_handle = (void*)loader;
    backends_.push_back(info);

    printf("BackendManager: loaded plugin %s v%s (%s)\n",
           plugin.id.c_str(), plugin.version.c_str(), so_path.c_str());
    rank_backends();
    return true;
}

int BackendManager::load_plugins(const std::string& directory) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::string> errors;
    auto loaders = BackendPluginLoader::scan_directory(directory, &errors);
    for (auto e : errors)
        fprintf(stderr, "BackendManager: %s\n", e.c_str());

    int loaded = 0;
    for (auto* loader : loaders) {
        BackendPlugin plugin;
        plugin.path = loader->description(); // crude: can't get path back
        plugin.id = loader->id();
        plugin.version = loader->version();
        plugin.loaded = true;
        plugins_.push_back(plugin);

        Backend* instance = loader->instantiate();
        if (!instance) { delete loader; continue; }

        BackendInfo info;
        info.id = plugin.id;
        info.type = loader->type();
        info.tier = (loader->type() == BackendType::NPU_XRT) ? BackendTier::T1_ACCELERATOR : BackendTier::T2_GPU;
        info.description = loader->description();
        info.priority = tier_priority(info.tier);
        info.available = true;
        info.functional = true; // presume functional
        info.instance = std::shared_ptr<Backend>(instance);
        info.plugin_handle = (void*)loader;
        backends_.push_back(info);
        loaded++;
    }

    if (loaded > 0) {
        printf("BackendManager: loaded %d plugin(s)\n", loaded);
        rank_backends();
    }
    return loaded;
}

// ── Destroy ──
void BackendManager::destroy() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& info : backends_) {
        destroy_instance(info);
        // If loaded via plugin, free the plugin loader
        if (info.plugin_handle) {
            auto* loader = (BackendPluginLoader*)info.plugin_handle;
            delete loader;
        }
    }
    backends_.clear();
    plugins_.clear();
    initialized_ = false;
}

// ── Strategy ──
void BackendManager::set_strategy(SelectionStrategy s) {
    strategy_ = s;
    switch (s) {
        case SelectionStrategy::FASTEST:
            printf("BackendManager: strategy → FASTEST (best benchmark score)\n");
            re_evaluate();
            break;
        case SelectionStrategy::LOWEST_POWER:
            printf("BackendManager: strategy → LOWEST POWER (NPU > GPU > CPU)\n");
            re_evaluate();
            break;
        case SelectionStrategy::MANUAL:
            printf("BackendManager: strategy → MANUAL (user-selected)\n");
            break;
        case SelectionStrategy::ROUND_ROBIN:
            printf("BackendManager: strategy → ROUND ROBIN\n");
            re_evaluate();
            break;
    }
}

SelectionStrategy BackendManager::strategy() const {
    return strategy_;
}

// ── Report ──
std::string BackendManager::report() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::string r;
    r += "╔══════════════════════════════════════════╗\n";
    r += "║   Backend Manager — Full Report          ║\n";
    r += "╚══════════════════════════════════════════╝\n\n";

    r += "Strategy: ";
    switch (strategy_) {
        case SelectionStrategy::FASTEST:     r += "Fastest"; break;
        case SelectionStrategy::LOWEST_POWER: r += "Lowest Power"; break;
        case SelectionStrategy::MANUAL:      r += "Manual"; break;
        case SelectionStrategy::ROUND_ROBIN: r += "Round Robin"; break;
    }
    r += "\nFallback: ";
    switch (fallback_policy_) {
        case FallbackPolicy::NONE:       r += "None (fail-fast)"; break;
        case FallbackPolicy::SEQUENTIAL: r += "Sequential"; break;
        case FallbackPolicy::BEST_EFFORT:r += "Best Effort"; break;
    }
    r += "\n\nActive backend: ";
    if (active_idx_ < backends_.size())
        r += backends_[active_idx_].id + " (" + backends_[active_idx_].description + ")\n";
    else
        r += "none\n";

    r += "\nDiscovered backends:\n";
    for (size_t i = 0; i < backends_.size(); i++) {
        auto& b = backends_[i];
        char line[256];
        snprintf(line, sizeof(line), "  %s [%s] %s%s — %.1f ms/tok, %lu infer%s\n",
                 b.id.c_str(),
                 b.available ? (b.functional ? "✓ " : "⚠ ") : "✗ ",
                 b.description.c_str(),
                 (i == active_idx_) ? " ← ACTIVE" : "",
                 b.score,
                 (unsigned long)b.total_inferences,
                 b.cumulative_ms > 0 ? (", " + std::to_string((long long)b.cumulative_ms) + " ms total").c_str() : "");
        r += line;
    }

    r += "\n" + monitor_.full_report();
    return r;
}

// ── Internal helpers (runtime loading via dlsym) ──
// GPU/NPU backends are loaded from the rocm_cpp shared library at runtime.
// Windows: no dynamic backend loading — all backends are compiled directly.
#ifndef _WIN32
#include <dlfcn.h>
#include <unordered_map>

// Cache dlopen handles so repeated backend (re-)creation doesn't grow the library
// refcount forever (fixes #90). The handle is intentionally never dlclose'd: a
// Backend's vtable lives in this library, so it must stay resident for the backend's
// lifetime — closing it would be a use-after-free.
static void* cached_dlopen(const char* lib) {
    static std::unordered_map<std::string, void*> cache;
    auto it = cache.find(lib);
    if (it != cache.end()) return it->second;
    void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (h) cache.emplace(lib, h);
    return h;
}
#endif

static Backend* try_load_backend(const char* lib, const char* sym) {
    void* h = cached_dlopen(lib);
    if (!h) return nullptr;
    auto* fn = (Backend* (*)())dlsym(h, sym);
    if (!fn) return nullptr;   // library stays cached; never dlclose
    Backend* b = fn();
    if (!b) return nullptr;
    return b;
}

Backend* BackendManager::create_instance_rt(const BackendInfo& info) {
    Backend* b = nullptr;
    switch (info.type) {
        case BackendType::HIP_GPU:
            // Mamba1 backend (Zamba-7B-v1, BlackMamba) — uses specialized HIP kernels
            if (info.id == "mamba1_gpu") {
#ifdef ROCM_CPP_STATIC_HIP
                b = create_mamba1_backend();
                if (b) return b;
#endif
                b = try_load_backend("librocm_cpp.so", "create_mamba1_backend");
                if (!b) b = try_load_backend("libmamba1_backend.so", "create_mamba1_backend");
                if (!b) { void* self = dlopen(NULL, RTLD_NOW|RTLD_LOCAL);
                    if (self) { auto* fn = (Backend*(*)())dlsym(self, "create_mamba1_backend");
                        if (fn) b = fn(); } }
                return b;
            }
            // Zamba2 backend (Zamba2-1.2B/2.7B/7B) — Mamba2 hybrid SSD kernels
            if (info.id == "zamba2_gpu") {
#ifdef ROCM_CPP_STATIC_HIP
                b = create_zamba2_backend();
                if (b) return b;
#endif
                b = try_load_backend("librocm_cpp.so", "create_zamba2_backend");
                if (!b) b = try_load_backend("libzamba2_backend.so", "create_zamba2_backend");
                if (!b) { void* self = dlopen(NULL, RTLD_NOW|RTLD_LOCAL);
                    if (self) { auto* fn = (Backend*(*)())dlsym(self, "create_zamba2_backend");
                        if (fn) b = fn(); } }
                return b;
            }
            // General HIP backend — loaded from shared library (keeps
            // backend_manager HIP-free, so pure-C++ consumers like
            // backend_demo can link without HIP symbols). If static linking
            // is desired, compile with -DROCM_CPP_STATIC_HIP and link
            // src/backend_hip.cpp directly into the target.
#ifdef ROCM_CPP_STATIC_HIP
            b = create_hip_backend();
            if (b) return b;
#endif
            b = try_load_backend("librocm_cpp.so", "create_hip_backend");
            if (!b) b = try_load_backend("libhip_backend.so", "create_hip_backend");
            if (!b) { void* self = dlopen(NULL, RTLD_NOW|RTLD_LOCAL);
                if (self) { auto* fn = (Backend*(*)())dlsym(self, "create_hip_backend");
                    if (fn) b = fn(); } }
            return b;
        case BackendType::VULKAN:
            b = try_load_backend("librocm_cpp.so", "create_vulkan_backend");
            if (!b) b = try_load_backend("libvulkan_backend.so", "create_vulkan_backend");
            if (!b) { void* self = dlopen(NULL, RTLD_NOW|RTLD_LOCAL);
                if (self) { auto* fn = (Backend*(*)())dlsym(self, "create_vulkan_backend");
                    if (fn) b = fn(); } }
            return b;
        case BackendType::NPU_XRT:
#ifdef ROCM_CPP_STATIC_NPU
            b = create_npu_backend();
            if (b) return b;
#endif
            b = try_load_backend("librocm_cpp.so", "create_npu_backend");
            if (!b) b = try_load_backend("libnpu_backend.so", "create_npu_backend");
            if (!b) { void* self = dlopen(NULL, RTLD_NOW|RTLD_LOCAL);
                if (self) { auto* fn = (Backend*(*)())dlsym(self, "create_npu_backend");
                    if (fn) b = fn(); } }
            return b;
        case BackendType::CPU_AVX512:
        case BackendType::CPU_SCALAR:
            return create_cpu_backend();
        case BackendType::GENERIC:
            return create_generic_backend();
        case BackendType::ZINC_GPU:
#ifdef ZINC_DISABLED
            return nullptr;
#else
            return create_zinc_backend();
#endif
        default:
            return nullptr;
    }
}

void BackendManager::destroy_instance(BackendInfo& info) {
    // shared_ptr reset destroys the Backend; virtual destructor chain ensures cleanup.
    info.instance.reset();
}

void BackendManager::rank_backends() {
    // Sort by strategy: FASTEST uses score first, LOWEST_POWER uses tier priority first.
    std::sort(backends_.begin(), backends_.end(),
        [this](const BackendInfo& a, const BackendInfo& b) {
            // Available always beats unavailable
            if (a.available != b.available) return a.available > b.available;
            // Functional always beats non-functional
            if (a.functional != b.functional) return a.functional > b.functional;

            if (strategy_ == SelectionStrategy::FASTEST) {
                // Primary sort: benchmark score (lower ms/tok = better)
                // Secondary sort: priority as tiebreaker
                bool a_scored = (a.score > 0);
                bool b_scored = (b.score > 0);
                if (a_scored != b_scored) return a_scored > b_scored;
                if (a_scored && b_scored && a.score != b.score)
                    return a.score < b.score;  // lower ms = faster
                return a.priority > b.priority;
            }

            // LOWEST_POWER, MANUAL, ROUND_ROBIN: priority first, score as tiebreaker
            if (a.priority != b.priority) return a.priority > b.priority;
            if (a.score > 0 && b.score > 0) return a.score < b.score;
            return a.score > b.score;
        });
}

// ── Global singleton ──
BackendManager& backend_manager() {
    static BackendManager instance;
    return instance;
}
