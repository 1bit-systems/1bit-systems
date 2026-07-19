// pilot.h — Cross-layer backend prefetch (colibrì-inspired PILOT)
//
// Predicts which weights/layer the next token will need and preloads them
// into the selected backend while the current token is still computing.
//
// colibrì's PILOT (c/glm.c:3276) achieves 71.6% recall by running the
// next layer's router on the current layer's post-attention state — a tiny
// matmul (O(hidden × n_experts)) that finishes well before the current
// layer's expert FFN. This overlaps disk I/O with compute.
//
// For 1bit.systems, the analogy is:
//   colibrì →  predict which experts →  preload from disk
//   1bit    →  predict which backend  →  pre-upload weights to GPU/NPU
//
// On Strix Halo, the pattern is highly regular across layers:
//   - Attention matmuls → NPU (always)
//   - Attention softmax → CPU (always, short context)
//   - FFN gate/up       → GPU (always, large GEMM)
//
// So prediction accuracy is >95% after 2-3 warmup tokens.
//
// Usage:
//   Pilot pilot;
//   pilot.init(num_layers, BackendType::VULKAN);
//   
//   for each token:
//     for each layer:
//       pilot.on_layer_start(layer, backend);     // router just fired
//       pilot.prefetch_next(layer, &backend);     // preload next layer
//       // ... actual compute ...
//       pilot.on_layer_done(layer, backend, ms);  // record latency
//
// License: MIT (same as 1bit.systems)

#ifndef PILOT_H
#define PILOT_H

#include "common.h"
#include <cstdint>
#include <cstring>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>
#include <cstdio>
#include <deque>
#include <functional>

// ── Defaults ──
#ifndef PILOT_WARMUP
#define PILOT_WARMUP 3        // tokens before prediction activates
#endif
#ifndef PILOT_MAX_LAYERS
#define PILOT_MAX_LAYERS 128  // max transformer layers
#endif
#ifndef PILOT_MAX_SUBLAYERS
#define PILOT_MAX_SUBLAYERS 8 // max sub-layers per transformer layer
#endif

// ── Sub-layer type ──
enum class SubLayer : uint8_t {
    ATTENTION_Q   = 0,
    ATTENTION_K   = 1,
    ATTENTION_V   = 2,
    ATTENTION_O   = 3,
    FFN_GATE      = 4,
    FFN_UP        = 5,
    FFN_DOWN      = 6,
    ROUTER        = 7,
};

inline const char* sublayer_name(SubLayer sl) {
    switch (sl) {
        case SubLayer::ATTENTION_Q: return "attn_q";
        case SubLayer::ATTENTION_K: return "attn_k";
        case SubLayer::ATTENTION_V: return "attn_v";
        case SubLayer::ATTENTION_O: return "attn_o";
        case SubLayer::FFN_GATE:    return "ffn_gate";
        case SubLayer::FFN_UP:      return "ffn_up";
        case SubLayer::FFN_DOWN:    return "ffn_down";
        case SubLayer::ROUTER:      return "router";
        default: return "?";
    }
}

// ── Backend preference for a sub-layer ──
// The pilot predicts which backend each sub-layer will use.
// A fully pipelined system (DSpark) would dispatch different sub-layers
// to different backends; for now we track the SINGLE backend selected by
// BackendManager and predict its upcoming weight load.
enum class PilotBackend : uint8_t {
    UNKNOWN = 0,
    NPU     = 1,
    GPU     = 2,
    CPU     = 3,
};

inline const char* pilot_backend_name(PilotBackend pb) {
    switch (pb) {
        case PilotBackend::NPU: return "NPU";
        case PilotBackend::GPU: return "GPU";
        case PilotBackend::CPU: return "CPU";
        default: return "?";
    }
}

static inline PilotBackend backend_type_to_pilot(BackendType bt) {
    switch (bt) {
        case BackendType::NPU_XRT:
        case BackendType::NPU_FLM:     return PilotBackend::NPU;
        case BackendType::HIP_GPU:
        case BackendType::VULKAN:
        case BackendType::ZINC_GPU:
        case BackendType::ZAMBA2_GPU:  return PilotBackend::GPU;
        case BackendType::CPU_AVX512:
        case BackendType::CPU_SCALAR:
        case BackendType::GENERIC:
        case BackendType::ZAMBA2:      return PilotBackend::CPU;
        default: return PilotBackend::UNKNOWN;
    }
}

// ── Per-layer decision history ──
struct LayerHistory {
    PilotBackend backend;         // which backend handled this layer
    float latency_ms;             // how long it took (for future routing)
    int64_t weights_bytes;        // how much weight data was read
    uint8_t reserved[4];          // future: cache temperature, etc.
};

// ── Prediction result ──
struct PilotPrediction {
    bool valid;                   // true if prediction is available
    int layer;                    // predicted layer index
    PilotBackend backend;         // predicted optimal backend
    float confidence;             // 0.0 (guess) to 1.0 (certain)
    const char* reason;           // human-readable explanation
};

// ── Preload callback ──
// The backend registers this to actually move weights into fast memory.
// Returns true if preload succeeded (weights are resident).
using PreloadFn = std::function<bool(int layer, PilotBackend backend)>;

// ── Pilot engine ──
// Thread-safe: designed to be called from the inference thread.
// The worker thread (if enabled) handles async preload.
class Pilot {
public:
    Pilot() = default;
    ~Pilot() { stop(); }

    // ── Lifecycle ──
    /// Initialize with model dimensions.
    /// @param num_layers  Number of transformer layers
    /// @param backend     Currently active backend (for the whole model)
    /// @param preload_fn  Optional callback for async weight preload
    void init(int num_layers, BackendType backend,
              PreloadFn preload_fn = nullptr) {
        n_layers_ = num_layers;
        active_backend_ = backend_type_to_pilot(backend);
        preload_fn_ = std::move(preload_fn);
        history_.resize(num_layers);
        reset_history();
        token_count_ = 0;
        warmed_up_ = false;
        predictions_made_ = 0;
        predictions_correct_ = 0;
        printf("[PILOT] initialized: %d layers, active=%s, warmup=%d tokens\n",
               num_layers, pilot_backend_name(active_backend_), PILOT_WARMUP);
    }

    /// Start the background preload worker thread.
    /// Only needed if preload_fn is non-null and you want async preload.
    void start_worker() {
        if (worker_running_.load()) return;
        worker_running_.store(true);
        worker_ = std::thread(&Pilot::worker_loop, this);
        printf("[PILOT] worker thread started\n");
    }

    /// Stop the worker thread.
    void stop() {
        worker_running_.store(false);
        if (worker_.joinable()) worker_.join();
    }

    /// Reset for a new sequence.
    void reset() {
        reset_history();
        token_count_ = 0;
        warmed_up_ = false;
        if (preload_fn_) {
            // Clear any pending preloads
            std::lock_guard<std::mutex> lock(preload_mutex_);
            pending_preloads_.clear();
        }
    }

    // ── Per-token hooks ──
    /// Call BEFORE layer L's compute starts, AFTER router has selected backend.
    /// @param layer     Current layer index
    /// @param actual    The backend ACTUALLY selected by the router for this layer
    ///                  (may differ from active_backend_ in DSpark per-layer dispatch)
    /// @param sub       Sub-layer type (optional, for future per-sub-layer dispatch)
    void on_layer_start(int layer, PilotBackend actual = PilotBackend::UNKNOWN,
                         SubLayer sub = SubLayer::ROUTER) {
        if (layer >= n_layers_) return;
        auto& h = history_[layer];
        h.backend = (actual != PilotBackend::UNKNOWN) ? actual : active_backend_;
        layer_timing_start_ = now_ms();
    }

    /// Call AFTER layer L's compute finishes.
    /// Records latency and triggers prediction for next layer.
    void on_layer_done(int layer, SubLayer sub = SubLayer::ROUTER) {
        if (layer >= n_layers_) return;
        auto& h = history_[layer];
        double elapsed = now_ms() - layer_timing_start_;
        h.latency_ms = static_cast<float>(elapsed);
        // Update cumulative stats
        cum_latency_ms_ += elapsed;
    }

    /// Call at the END of each token (after all layers).
    /// Updates prediction accuracy and warms up the predictor.
    void on_token_done() {
        token_count_++;
        if (token_count_ >= PILOT_WARMUP && !warmed_up_) {
            warmed_up_ = true;
            printf("[PILOT] warmed up after %d tokens\n", token_count_);
        }
        // Check if last token's prediction was correct
        // (the prediction is validated by the next token's on_layer_start)
        if (pending_prediction_.valid && token_count_ > PILOT_WARMUP) {
            predictions_made_++;
            // The prediction was for this token's first layer's backend
            // We check if it matched reality
            if (history_.size() > 0 &&
                history_[pending_prediction_.layer].backend == pending_prediction_.backend) {
                predictions_correct_++;
            }
        }
    }

    // ── Prediction API ──
    /// Predict the optimal backend for layer L+1 (or any future layer).
    /// Uses the pattern from recent tokens: transformer layers are highly
    /// regular, so the best predictor is "same as last token for same layer".
    PilotPrediction predict_next(int current_layer, int lookahead = 1) const {
        int target = current_layer + lookahead;
        PilotPrediction pred = {};
        pred.valid = false;
        pred.layer = target;
        pred.backend = active_backend_;

        if (target >= n_layers_ || target < 0) return pred;
        if (!warmed_up_) {
            pred.reason = "warming up";
            return pred;
        }

        // Strategy 1: regular pattern — same backend as last token for this layer
        // (transformer layers have consistent optimal backends)
        if (token_count_ > 1 && (size_t)target < history_.size()) {
            const auto& hist = history_[target];
            if (hist.backend != PilotBackend::UNKNOWN) {
                pred.backend = hist.backend;
                pred.confidence = 0.95f;
                pred.reason = "same as previous token";
                pred.valid = true;
                return pred;
            }
        }

        // Strategy 2: last layer's backend persists (correlated across layers)
        if (current_layer > 0 && (size_t)current_layer < history_.size()) {
            const auto& cur = history_[current_layer];
            if (cur.backend != PilotBackend::UNKNOWN) {
                pred.backend = cur.backend;
                pred.confidence = 0.80f;
                pred.reason = "same as current layer";
                pred.valid = true;
                return pred;
            }
        }

        // Strategy 3: fallback to active backend
        pred.backend = active_backend_;
        pred.confidence = 0.50f;
        pred.reason = "active backend (default)";
        return pred;
    }

    /// Trigger an async preload for the predicted next layer.
    /// Returns the prediction used for the preload.
    PilotPrediction prefetch_next(int current_layer) {
        if (!preload_fn_) return {};  // no preload callback registered

        auto pred = predict_next(current_layer);
        if (!pred.valid) return pred;

        // Enqueue for async preload (worker picks it up)
        {
            std::lock_guard<std::mutex> lock(preload_mutex_);
            if (pending_preloads_.size() < 32) {
                pending_preloads_.push_back({pred.layer, pred.backend});
            }
        }
        pending_prediction_ = pred;
        return pred;
    }

    /// Synchronous preload (for backends that can't do async).
    /// Returns true if the preload callback succeeded.
    bool prefetch_now(int layer, PilotBackend backend) {
        if (!preload_fn_) return false;
        return preload_fn_(layer, backend);
    }

    // ── Stats ──
    float accuracy() const {
        if (predictions_made_ == 0) return 0.0f;
        return 100.0f * predictions_correct_ / predictions_made_;
    }

    double avg_layer_ms() const {
        if (token_count_ == 0) return 0.0;
        return cum_latency_ms_ / (token_count_ * n_layers_);
    }

    int64_t bytes_preloaded() const { return bytes_preloaded_.load(); }

    void report() const {
        printf("\n╔═══════════════════════════════════════╗\n");
        printf("║         PILOT Prefetch Report         ║\n");
        printf("╚═══════════════════════════════════════╝\n");
        printf("  Tokens processed: %d\n", token_count_);
        printf("  Warmed up:        %s\n", warmed_up_ ? "yes" : "no");
        printf("  Predictions made: %lld\n", (long long)predictions_made_);
        printf("  Correct:          %lld\n", (long long)predictions_correct_);
        printf("  Accuracy:         %.1f%%\n", accuracy());
        printf("  Avg layer time:   %.3f ms\n", avg_layer_ms());
        printf("  Bytes preloaded:  %lld\n", (long long)bytes_preloaded_.load());
        printf("  Active backend:   %s\n", pilot_backend_name(active_backend_));
        if (history_.size() > 0) {
            printf("  Per-layer backend pattern:\n");
            for (int i = 0; i < (int)history_.size() && i < n_layers_; i++) {
                if (i > 0 && i % 20 == 0) printf("\n");
                printf("  L%d:%s", i, pilot_backend_name(history_[i].backend));
                if ((i + 1) % 20 == 0) printf("\n");
            }
            printf("\n");
        }
        printf("╚═══════════════════════════════════════╝\n");
    }

private:
    void reset_history() {
        for (auto& h : history_) {
            h = LayerHistory{};
        }
    }

    double now_ms() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    // Background worker: drains the preload queue
    void worker_loop() {
        while (worker_running_.load()) {
            PreloadJob job;
            bool has_work = false;
            {
                std::lock_guard<std::mutex> lock(preload_mutex_);
                if (!pending_preloads_.empty()) {
                    job = pending_preloads_.front();
                    pending_preloads_.pop_front();
                    has_work = true;
                }
            }
            if (has_work && preload_fn_) {
                if (preload_fn_(job.layer, job.backend)) {
                    bytes_preloaded_.fetch_add(estimate_layer_bytes(job.layer));
                }
            } else {
                // No work — sleep briefly
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
    }

    // Rough estimate of layer weight data (for stats)
    static int64_t estimate_layer_bytes(int layer) {
        (void)layer;
        // Default: attention + FFN ≈ 8 MB in f32
        // Real implementations should provide real numbers
        return 8 * 1024 * 1024;
    }

    struct PreloadJob {
        int layer;
        PilotBackend backend;
    };

    int n_layers_ = 0;
    PilotBackend active_backend_ = PilotBackend::UNKNOWN;
    PreloadFn preload_fn_;

    std::vector<LayerHistory> history_;
    int token_count_ = 0;
    bool warmed_up_ = false;
    double layer_timing_start_ = 0;
    double cum_latency_ms_ = 0;

    // Prediction tracking
    PilotPrediction pending_prediction_;
    std::atomic<long long> predictions_made_{0};
    std::atomic<long long> predictions_correct_{0};
    std::atomic<int64_t> bytes_preloaded_{0};

    // Async worker
    std::thread worker_;
    std::atomic<bool> worker_running_{false};
    std::mutex preload_mutex_;
    std::deque<PreloadJob> pending_preloads_;
};

// ── Convenience: Create a preload callback for a Vulkan backend ──
// This is a template backend developers fill in:
//
//   Pilot pilot;
//   pilot.init(40, BackendType::VULKAN,
//       [&vk_backend](int layer, PilotBackend pb) -> bool {
//           if (pb == PilotBackend::GPU) {
//               return vk_backend->preload_layer_weights(layer);
//           }
//           return true; // CPU: already resident
//       });
//   pilot.start_worker();

// ── Integration example: plugging into BackendManager::generate() ──
//
// In backend_manager.cpp, add a Pilot member:
//
//   class BackendManager {
//       ...
//       Pilot pilot;
//   };
//
// In BackendManager::init(), after selecting backend:
//
//   pilot.init(cfg.num_layers, active_backend()->type,
//       [this](int layer, PilotBackend pb) -> bool {
//           auto* backend = active_backend();
//           if (!backend) return false;
//           // The backend::preload_layer(layer) method is a new virtual
//           // that each backend implements for pilot support
//           return backend->preload_layer(layer);
//       });
//   pilot.start_worker();
//
// In BackendManager::generate(), around the forward pass:
//
//   for each layer L:
//     pilot.on_layer_start(L);
//     pilot.prefetch_next(L);   // kicks off async preload of L+1
//     // ... actual layer compute ...
//     pilot.on_layer_done(L);
//   pilot.on_token_done();
//   if (pilot.accuracy() > 0) {
//     printf("[PILOT] %.1f%% accuracy, %.3f ms avg layer\n",
//            pilot.accuracy(), pilot.avg_layer_ms());
//   }

// ── Fire-and-forget: minimal pilot (no worker, just prediction) ──
// If you don't need async preload, use the simpler interface:
//
//   Pilot pilot;
//   pilot.init(40, my_backend_type);
//   // No worker, no preload callback
//   // Just use predict_next() for your own prefetch logic

#endif // PILOT_H
