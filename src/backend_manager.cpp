// backend_manager.cpp — Backend Manager implementation
// Windows ML-style backend orchestrator with auto-detection, selection, failover.

#include "backend_manager.h"
#include "backend_plugin.h"
#include "backend_detect.h"
#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <thread>
#include <chrono>

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
        info.description = "AMD XDNA NPU via XRT";
        info.priority = tier_priority(info.tier) + 50;
        info.available = has_npu();
        info.functional = false;  // needs init to confirm
        info.score = 0;
        info.total_inferences = 0;
        info.failed_inferences = 0;
        info.cumulative_ms = 0;
        info.instance = nullptr;
        info.plugin_handle = nullptr;
        printf("  %-25s %s\n", "NPU XDNA (XRT)", info.available ? "✅ detected" : "❌ not available");
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

    // Try each backend in priority order until one initializes
    for (auto& info : backends_) {
        if (!info.available) continue;

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
            printf("  → ✅ initialized successfully\n");
            initialized_ = true;

            // Create monitor entry
            auto* pm = monitor_.for_backend(info.id);
            if (pm) pm->healthy = true;

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
                if (!b.available || !b.functional) continue;
                if (b.score > 0 && b.score < best_score) {
                    best_score = b.score;
                    best_idx = i;
                }
            }
            // If no backend has a score, fall back to first available+functional
            if (best_idx == backends_.size()) {
                for (size_t i = 0; i < backends_.size(); i++) {
                    if (backends_[i].available && backends_[i].functional) {
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
                if (backends_[i].available && backends_[i].functional) {
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
                if (backends_[idx].available && backends_[idx].functional) {
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
    std::shared_ptr<Backend> snap;
    bool need_failover = false;
    size_t prev_idx = 0;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (active_idx_ >= backends_.size()) return -1;
        auto& info = backends_[active_idx_];
        if (info.functional && info.instance) {
            snap = info.instance;  // shared_ptr copy — keeps Backend alive
        } else {
            need_failover = true;
            prev_idx = active_idx_;
        }
    }

    // Phase 2: inference WITHOUT the lock — snap keeps the Backend alive.
    if (snap) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int result = snap->generate(token_id);
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        if (result >= 0) {
            // Fast path — re-acquire lock briefly for stats
            std::lock_guard<std::mutex> lock(mtx_);
            if (active_idx_ < backends_.size()) {
                auto* info = &backends_[active_idx_];
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

        // Failed — re-acquire lock for stats + failover
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (active_idx_ < backends_.size()) {
                auto* info = &backends_[active_idx_];
                info->failed_inferences++;
                info->functional = false;
                monitor_.record(info->id, ms, false);
                monitor_.record_failure(info->id, "generate() returned -1");
            }
            need_failover = true;
            prev_idx = active_idx_;
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
            int result = info->instance->generate(token_id);
            float ms = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            if (result >= 0) {
                info->total_inferences++;
                info->cumulative_ms += ms;
                monitor_.record(info->id, ms, true);
                return result;
            }
            info->failed_inferences++;
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
        // Simple probe: try reset
        bool ok = b->reset();
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

    for (auto& info : backends_) {
        if (!info.available) continue;

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

        // Create instance if needed
        if (!info.instance) {
            auto* raw = create_instance_rt(info);
            if (!raw) {
                printf("❌ (creation failed)\n");
                continue;
            }
            info.instance = std::shared_ptr<Backend>(raw);
            if (!info.instance->init(cfg_, weights_dir_)) {
                printf("❌ (init failed)\n");
                destroy_instance(info);
                continue;
            }
        }

        float ms = info.instance->benchmark(tokens);
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
        info.tier = BackendTier::T2_GPU; // default for plugins
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
// CPU backend is linked in directly (pure C++, no deps).
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
            // Load HIP backend from shared library (keeps backend_manager HIP-free,
            // so pure-C++ consumers like backend_demo can link without HIP symbols).
            // If static linking is desired, compile with -DROCM_CPP_STATIC_HIP
            // and link src/backend_hip.cpp directly into the target.
#ifdef ROCM_CPP_STATIC_HIP
            b = create_hip_backend();
            if (b) return b;
#endif
            b = try_load_backend("librocm_cpp.so", "create_hip_backend");
            if (!b) b = try_load_backend("libhip_backend.so", "create_hip_backend");
            // Last resort: lookup in the main executable itself (statically linked
            // via unified_server with -rdynamic). The `self` handle is NOT cached
            // because repeated dlopen(NULL) just bumps the refcount — safe.
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
            b = try_load_backend("librocm_cpp.so", "create_npu_backend");
            if (!b) b = try_load_backend("libnpu_backend.so", "create_npu_backend");
            if (!b) { void* self = dlopen(NULL, RTLD_NOW|RTLD_LOCAL);
                if (self) { auto* fn = (Backend*(*)())dlsym(self, "create_npu_backend");
                    if (fn) b = fn(); } }
            return b;
        case BackendType::CPU_AVX512:
        case BackendType::CPU_SCALAR:
            return create_cpu_backend();
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
