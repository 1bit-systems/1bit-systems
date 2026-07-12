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
        info.instance = create_instance_rt(info);
        if (!info.instance) {
            printf("  → creation failed\n");
            continue;
        }

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

// ── Select active backend ──
bool BackendManager::select_best() {
    for (size_t i = 0; i < backends_.size(); i++) {
        if (backends_[i].available && backends_[i].functional) {
            active_idx_ = i;
            return true;
        }
    }
    return false;
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
    return backends_[active_idx_].instance;
}

const BackendInfo* BackendManager::active_info() const {
    if (active_idx_ >= backends_.size()) return nullptr;
    return &backends_[active_idx_];
}

// ── Inference with failover ──
int BackendManager::generate(int token_id) {
    if (!initialized_ || backends_.empty()) return -1;

    std::lock_guard<std::mutex> lock(mtx_);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto* info = &backends_[active_idx_];

    // Fast path
    if (info->functional && info->instance) {
        int result = info->instance->generate(token_id);
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        if (result >= 0) {
            info->total_inferences++;
            info->cumulative_ms += ms;
            monitor_.record(info->id, ms, true);
            return result;
        }

        // Failed — record and fall over
        info->failed_inferences++;
        info->functional = false;
        monitor_.record(info->id, ms, false);
        monitor_.record_failure(info->id, "generate() returned -1");
        fprintf(stderr, "BackendManager: %s failed on generate(), failing over...\n",
                info->id.c_str());
    }

    // Failover
    size_t prev_idx = active_idx_;  // snapshot before failover() reassigns active_idx_ (fixes #89)
    if (failover()) {
        info = &backends_[active_idx_];
        monitor_.record_fallback(backends_[prev_idx].id, info->id);
        printf("BackendManager: failed over to %s\n", info->id.c_str());
        t0 = std::chrono::high_resolution_clock::now();
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

    fprintf(stderr, "BackendManager: ALL BACKENDS FAILED\n");
    return -1;
}

bool BackendManager::forward(int token_id, float* hidden_out) {
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
    auto* b = active_backend();
    if (!b || !initialized_) return false;
    return b->lm_head(hidden, logits, argmax);
}

bool BackendManager::reset() {
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
            info.instance = create_instance_rt(info);
            if (!info.instance) continue;
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
    if (!health_check()) {
        fprintf(stderr, "BackendManager: health check failed, failing over...\n");
        failover();
    }
}

// ── Benchmarking ──
void BackendManager::benchmark_all(int tokens) {
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
        bool needs_cleanup = false;
        if (!info.instance) {
            info.instance = create_instance_rt(info);
            if (!info.instance) {
                printf("❌ (creation failed)\n");
                continue;
            }
            if (!info.instance->init(cfg_, weights_dir_)) {
                printf("❌ (init failed)\n");
                destroy_instance(info);
                continue;
            }
            needs_cleanup = true;
        }

        float ms = info.instance->benchmark(tokens);
        info.score = ms;
        printf("%.1f ms/tok\n", ms);

        if (needs_cleanup && info.instance) {
            delete info.instance;
            info.instance = nullptr;
            info.functional = false;  // not selectable until re-instantiated (fixes #93)
        }
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

const BackendInfo* BackendManager::best_for_tier(BackendTier tier) const {
    for (auto& info : backends_) {
        if (info.tier == tier && info.available && info.functional)
            return &info;
    }
    return nullptr;
}

// ── Plugins ──
bool BackendManager::load_plugin(const std::string& so_path) {
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
    info.instance = instance;
    info.plugin_handle = (void*)loader;
    backends_.push_back(info);

    printf("BackendManager: loaded plugin %s v%s (%s)\n",
           plugin.id.c_str(), plugin.version.c_str(), so_path.c_str());
    rank_backends();
    return true;
}

int BackendManager::load_plugins(const std::string& directory) {
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
        info.instance = instance;
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
            break;
        case SelectionStrategy::LOWEST_POWER:
            printf("BackendManager: strategy → LOWEST POWER (NPU > GPU > CPU)\n");
            break;
        case SelectionStrategy::MANUAL:
            printf("BackendManager: strategy → MANUAL (user-selected)\n");
            break;
        case SelectionStrategy::ROUND_ROBIN:
            printf("BackendManager: strategy → ROUND ROBIN\n");
            break;
    }
}

SelectionStrategy BackendManager::strategy() const {
    return strategy_;
}

// ── Report ──
std::string BackendManager::report() const {
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
    switch (info.type) {
        case BackendType::HIP_GPU:
            return try_load_backend("librocm_cpp.so", "create_hip_backend");
        case BackendType::VULKAN:
            return try_load_backend("librocm_cpp.so", "create_vulkan_backend");
        case BackendType::NPU_XRT:
            return try_load_backend("librocm_cpp.so", "create_npu_backend");
        case BackendType::CPU_AVX512:
        case BackendType::CPU_SCALAR:
            return create_cpu_backend();
        default:
            return nullptr;
    }
}

void BackendManager::destroy_instance(BackendInfo& info) {
    if (info.instance) {
        // Delete triggers the destructor which calls destroy() — no need to call it twice.
        // The virtual destructor chain ensures proper cleanup.
        delete info.instance;
        info.instance = nullptr;
    }
}

void BackendManager::rank_backends() {
    // Sort by priority descending (highest first), then by score ascending
    std::sort(backends_.begin(), backends_.end(),
        [](const BackendInfo& a, const BackendInfo& b) {
            if (a.available != b.available) return a.available > b.available;
            if (a.functional != b.functional) return a.functional > b.functional;
            if (a.priority != b.priority) return a.priority > b.priority;
            if (a.score > 0 && b.score > 0) return a.score < b.score; // lower ms = better
            return a.score > b.score; // untested goes last
        });
}

// ── Global singleton ──
BackendManager& backend_manager() {
    static BackendManager instance;
    return instance;
}
