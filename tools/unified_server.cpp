// unified_server.cpp — One binary, all backends, auto-failover.
// Replaces tests/zaya_server.cpp by wiring BackendManager into an HTTP server.
// Auto-detects NPU, GPU (HIP/Vulkan), and CPU backends, picks the fastest,
// and transparently failsover on error — zero config, one binary.
//
// Build: cmake --build . --target unified_server -j8
// Run:   ./build/unified_server [--port 8088] [--weights /tmp/zaya_weights]
//
// API (OpenAI-compatible):
//   GET  /v1/health           — Backend status & metrics dashboard
//   GET  /v1/models           — Available model(s)
//   POST /v1/chat/completions — Generate with auto-selected best backend
//   POST /v1/completions      — Legacy completion endpoint
//   POST /v1/backend/select   — Manually select a specific backend
//   GET  /v1/backend/status   — Full backend manager report
//
// Headers (optional):
//   X-Backend: hip_gpu | npu_xrt | vulkan_gpu | cpu_avx512 | cpu_scalar | auto
//   X-Strategy: fastest | lowest_power | manual | round_robin

#include "backend_manager.h"
#include "backend_monitor.h"
#include "backend_plugin.h"
#include "backend.h"
#include "model_discovery.h"
#include "model_router.h"
#include "simple_tokenizer.h"
#include "vl_processor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <signal.h>
#ifdef _WIN32
// Minimal getopt for Windows — MSVC doesn't ship it
#include <io.h>
#include <string.h>
static int optind = 1; static int opterr = 1; static const char* optarg = nullptr;
struct option { const char* name; int has_arg; int* flag; int val; };
enum { no_argument = 0, required_argument = 1, optional_argument = 2 };
static int getopt_long(int argc, char* const argv[], const char* optstring, const struct option* longopts, int* longindex) {
    if (optind >= argc || argv[optind][0] != '-') return -1;
    if (argv[optind][1] == '-' && longopts) {
        // Long option
        for (int i = 0; longopts[i].name; i++) {
            if (strcmp(argv[optind]+2, longopts[i].name) == 0) {
                if (longindex) *longindex = i;
                optind++;
                if (longopts[i].has_arg == required_argument) {
                    optarg = (optind < argc) ? argv[optind++] : nullptr;
                }
                if (longopts[i].flag) { *longopts[i].flag = longopts[i].val; return 0; }
                return longopts[i].val;
            }
        }
        return '?';
    }
    // Short option
    int opt = argv[optind][1];
    const char* p = optstring ? strchr(optstring, opt) : nullptr;
    if (!p) return '?';
    if (p[1] == ':') {
        optarg = (optind + 1 < argc) ? argv[optind+1] : nullptr;
        optind += (optarg ? 2 : 1);
    } else {
        optind++;
    }
    return opt;
}
#else
#include <getopt.h>
#endif
#include <fstream>
#include <fcntl.h>
#ifndef _WIN32
#include <dirent.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#else
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#define getpid _getpid
#define close _close
#define unlink _unlink
// Use wrappers instead of macros to avoid clashing with std::ifstream::read
static inline int read(int fd, void* buf, unsigned int count) { return (int)_read(fd, buf, count); }
#endif

#include "strategy_engine.h"
#include "agent_watchdog.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Globals ──
static std::atomic<bool> keep_running{true};
static std::string g_weights_dir = []() -> std::string {
    const char* env = getenv("ZAYA_WEIGHTS_DIR");
    if (env && env[0]) { std::string s(env); if (s.back()!='/') s+='/'; return s; }
    const char* xdg = getenv("XDG_DATA_HOME");
    if (xdg && xdg[0]) return std::string(xdg) + "/1bit-systems/weights/";
    const char* home = getenv("HOME");
    if (home && home[0]) return std::string(home) + "/.local/share/1bit-systems/weights/";
    return "/tmp/zaya_weights/";
}();
static int g_port = 8088;

// Protect global state accessed from HTTP handler threads (fixes #364)
static std::mutex g_strategy_mutex;   // protects g_strategy_engine + g_watchdog
static std::mutex g_config_mutex;     // protects current_cfg, g_tokenizer, model switching

// Serializes ALL access to the single shared BackendManager compute context —
// the actual decode (mgr.reset/generate), model reload (mgr.init), and
// active-backend switch (mgr.select_backend/set_strategy). A BackendManager
// holds one mutable inference state (KV cache + active-backend pointer); two
// requests decoding concurrently corrupt it (AUDIT_ISSUES.md #2). This is the
// OUTERMOST lock: g_config_mutex / g_strategy_mutex may be acquired while it is
// held, but g_inference_mutex must never be acquired while holding either of
// those, or the two orders can deadlock.
// Generation timeout: abort in-flight generate_completion() if it exceeds
// this wall-clock limit. Prevents a single slow/memory-hungry request from
// holding g_inference_mutex indefinitely and OOM-killing the entire server
// (issue #948). Configurable via CLI --gen-timeout-ms or GEN_TIMEOUT_MS env.
// Default: 600000ms = 10 minutes. 0 = no timeout.
static int g_generation_timeout_ms = []() -> int {
    const char* env = getenv("GEN_TIMEOUT_MS");
    if (env && env[0]) {
        int v = atoi(env);
        if (v >= 0) return v;
    }
    return 600000;  // 10 minutes default
}();

static std::mutex g_inference_mutex;

// ── Strategy engine + agent watchdog (global for HTTP handler access) ──
static StrategyEngine g_strategy_engine;
static AgentWatchdog* g_watchdog = nullptr;

// ── Tokenizer: uses RCPP BPE tokenizer (.htok) when available, falls back
// to SimpleTokenizer (ASCII + UTF-8 byte passthrough) when not.
// The RCPP tokenizer is linked via librocm_cpp and reads .htok binary format.
// SimpleTokenizer itself lives in include/simple_tokenizer.h; g_tokenizer is
// defined once in src/simple_tokenizer.cpp (part of backend_manager) so any
// backend that needs it (e.g. FlmBackend) can link against the same instance.

// ── Signal handler ──
static void handle_sigint(int) {
    keep_running = false;
}

// ── Helpers ──
static std::string tokenizer_path() {
    // Prefer .htok (BPE tokenizer from RCPP). Fall back to tokenizer.json for diagnostics.
    std::string htok = g_weights_dir + "/tokenizer.htok";
    std::ifstream f(htok, std::ios::binary);
    if (f.good()) return htok;
    return g_weights_dir + "/tokenizer.json";
}

static ModelConfig default_model_config() {
    ModelConfig cfg;
    cfg.hidden = 2048;
    cfg.n_heads = 8;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 128;
    cfg.n_layers = 40;
    cfg.n_experts = 16;
    cfg.n_ff = 2048;
    cfg.vocab = 262272;
    cfg.router_hidden = 256;
    cfg.qkv_dim = 1280;
    return cfg;
}

// ── Resolve backend ID from header or query param ──
static std::string resolve_backend_id(const httplib::Request& req) {
    // Check header first, then query param
    if (req.has_header("X-Backend"))
        return req.get_header_value("X-Backend");
    if (req.has_param("backend"))
        return req.get_param_value("backend");
    return "auto";
}

static SelectionStrategy resolve_strategy(const httplib::Request& req) {
    std::string s;
    if (req.has_header("X-Strategy"))
        s = req.get_header_value("X-Strategy");
    else if (req.has_param("strategy"))
        s = req.get_param_value("strategy");

    if (s == "lowest_power") return SelectionStrategy::LOWEST_POWER;
    if (s == "manual")       return SelectionStrategy::MANUAL;
    if (s == "round_robin")  return SelectionStrategy::ROUND_ROBIN;
    return SelectionStrategy::FASTEST;  // default
}

// ── Resolve strategy engine name from header or query param ──
static std::string resolve_strategy_name(const httplib::Request& req) {
    if (req.has_header("X-Router-Strategy"))
        return req.get_header_value("X-Router-Strategy");
    if (req.has_param("strategy"))
        return req.get_param_value("strategy");
    return "";  // use default
}

// ── Build model info JSON ──
static json model_info_json(const BackendInfo* active, const std::string& model_name = "unknown") {
    json j;
    j["id"] = model_name;
    j["object"] = "model";
    j["created"] = time(nullptr);
    j["owned_by"] = "1bit-systems";
    if (active) {
        j["backend"] = active->id;
        j["backend_type"] = backend_name(active->type);
    }
    return j;
}

// ── Build health/metrics JSON ──
static json health_json(BackendManager& mgr) {
    json j;
    j["status"] = "ok";
    j["service"] = "1bit-systems unified inference server";

    auto* active = mgr.active_info();
    if (active) {
        j["active_backend"]["id"] = active->id;
        j["active_backend"]["type"] = backend_name(active->type);
        j["active_backend"]["tier"] = tier_name(active->tier);
        j["active_backend"]["score_ms_per_tok"] = active->score;
        j["active_backend"]["total_inferences"] = active->total_inferences;
        j["active_backend"]["failed_inferences"] = active->failed_inferences;
        j["active_backend"]["functional"] = active->functional;
    }

    json backends = json::array();
    for (auto& b : mgr.backends()) {
        json bj;
        bj["id"] = b.id;
        bj["type"] = backend_name(b.type);
        bj["tier"] = tier_name(b.tier);
        bj["available"] = b.available;
        bj["functional"] = b.functional;
        bj["score_ms_per_tok"] = b.score;
        bj["total_inferences"] = b.total_inferences;
        bj["failed_inferences"] = b.failed_inferences;
        backends.push_back(bj);
    }
    j["backends"] = backends;

    auto* monitor = mgr.monitor_stats();
    j["metrics"]["total_inferences"] = monitor->total_inferences();
    j["metrics"]["total_failures"] = monitor->total_failures();
    j["metrics"]["total_fallbacks"] = monitor->total_fallbacks();

    json per_backend = json::array();
    for (auto* pm : monitor->all_metrics()) {
        json mj;
        mj["backend_id"] = pm->backend_id;
        mj["inferences"] = pm->inferences.load();
        mj["failures"] = pm->failures.load();
        mj["fallbacks"] = pm->fallbacks.load();
        mj["last_ms"] = pm->last_ms.load();
        mj["avg_ms"] = pm->recent_ms.avg();
        mj["tokens_per_second"] = pm->tokens_per_second.load();
        mj["healthy"] = pm->healthy.load();
        per_backend.push_back(mj);
    }
    j["metrics"]["per_backend"] = per_backend;

    return j;
}

// ── Generate completion with per-token strategy routing ──
// Returns { text, tokens, backend_used, ms_per_tok, tok_s }
// When strategy_engine is provided, routes each token through the strategy
// instead of using a fixed backend.
static json generate_completion(BackendManager& mgr,
                                 const std::vector<int>& prompt_tokens,
                                 const std::vector<double>& prompt_logprobs,
                                 int max_tokens,
                                 const std::string& backend_id,
                                 StrategyEngine* strategy_engine = nullptr,
                                 const std::string& user_message = "") {
    json result;

    // Select fixed backend if specified (overrides strategy routing)
    if (backend_id != "auto" && backend_id != "") {
        mgr.select_backend(backend_id);
    }

    // ── Pre-generate: pick initial backend via strategy ──
    std::string active_backend_id;
    if (strategy_engine) {
        // Use strategy to pick initial backend
        // For the first token, we don't have logprobs yet
        TokenContext init_ctx{-1, 0.0, -1.0, 0,
                             prompt_tokens.size(), (size_t)max_tokens,
                             user_message};
        auto decision = strategy_engine->route(init_ctx);
        active_backend_id = decision.backend;
        if (!decision.draft_backend.empty()) {
            active_backend_id = decision.draft_backend;
        }
        mgr.select_backend(active_backend_id);
    } else {
        active_backend_id = backend_id;
    }

    auto* active = mgr.active_info();
    result["backend_used"] = active ? active->id : "none";
    result["strategy"] = strategy_engine ? strategy_engine->name() : "none";

    // Reset backend state for new sequence
    if (!mgr.reset()) {
        result["error"] = "Failed to reset backend";
        result["tokens"] = json::array();
        result["text"] = "";
        return result;
    }

    std::vector<int> output_tokens;
    std::vector<double> output_logprobs;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto timeout_deadline = t0 + std::chrono::milliseconds(g_generation_timeout_ms);
    bool timed_out = false;
    int last_token = prompt_tokens.empty() ? g_tokenizer.bos_id : prompt_tokens.back();

    // Prefill: process prompt tokens
    // Skip BOS token (first token == bos_id) for SSM models where
    // feeding BOS as a regular token corrupts the recurrent state.
    size_t prefill_start = 0;
    if (!prompt_tokens.empty() && prompt_tokens[0] == g_tokenizer.bos_id) {
        prefill_start = 1;
    }
    for (size_t i = prefill_start; i + 1 < prompt_tokens.size(); i++) {
        // Check generation timeout between prefill tokens (issue #948)
        if (g_generation_timeout_ms > 0 &&
            std::chrono::high_resolution_clock::now() >= timeout_deadline) {
            float ms = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            result["error"] = "Generation timed out during prefill after " +
                               std::to_string((int)ms) + "ms";
            result["gen_ms"] = ms;
            result["gen_tokens"] = (int)output_tokens.size();
            result["text"] = "";
            result["timed_out"] = true;
            return result;
        }
        int result_id = mgr.generate(prompt_tokens[i]);
        if (result_id < 0) {
            float ms = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            result["error"] = "Backend failed during prefill at token " + std::to_string(i);
            result["gen_ms"] = ms;
            result["gen_tokens"] = (int)output_tokens.size();
            return result;
        }
    }

    // ── Generate new tokens with per-token strategy routing ──
    // Track which backend each token goes to
    std::vector<std::string> per_token_backend;
    for (int i = 0; i < max_tokens; i++) {
        // ── Check generation timeout (issue #948) ──
        // Timeout is checked per-token so g_inference_mutex is released promptly
        // when a slow request exceeds the wall-clock limit. This prevents a single
        // client from holding the server hostage while the model slowly accumulates
        // memory (e.g. zamba-7b-v1 on CPU grew to 56.5GB over ~11 minutes).
        if (g_generation_timeout_ms > 0 &&
            std::chrono::high_resolution_clock::now() >= timeout_deadline) {
            timed_out = true;
            break;
        }

        // ── Strategically choose backend for this token ──
        if (strategy_engine) {
            // Build logprob/entropy from last generated output
            double last_lp = 0.0;
            double last_entropy = -1.0;
            if (!output_logprobs.empty()) {
                last_lp = output_logprobs.back();
            } else if (!prompt_logprobs.empty()) {
                last_lp = prompt_logprobs.back();
            }

            TokenContext ctx{last_token, last_lp, last_entropy,
                            (size_t)i, prompt_tokens.size(), (size_t)max_tokens,
                            user_message};
            auto decision = strategy_engine->route(ctx);

            // Select the decided backend
            if (mgr.select_backend(decision.backend)) {
                active_backend_id = decision.backend;
            }
            per_token_backend.push_back(decision.backend);
        }

        // ── Generate token ──
        // When strategy engine needs logprobs (cascade/adaptive), use
        // forward()+lm_head() separately for real model confidence.
        // Otherwise, use fast generate() which does both in one call.
        bool need_logprobs = strategy_engine && (
            strategy_engine->name() == std::string("cascade") ||
            strategy_engine->name() == std::string("adaptive")
        );

        int next = -1;
        double token_logprob = 0.0;

        // First token (i==0): compute logprob from the last prompt token's
        // forward pass so cascade has a real confidence signal immediately.
        int vs = 262272, hs = 2048;
        auto* ai = mgr.active_info();
        if (ai && ai->instance) {
            auto& mcfg = ai->instance->cfg;
            if (mcfg.vocab_size > 0) vs = mcfg.vocab_size;
            else if (mcfg.vocab > 0) vs = mcfg.vocab;
            if (mcfg.hidden_size > 0) hs = mcfg.hidden_size;
            else if (mcfg.hidden > 0) hs = mcfg.hidden;
        }
        if (i == 0 && need_logprobs && output_logprobs.empty()) {
            std::vector<float> hidden_buf(hs);
            std::vector<float> logits_buf(vs);
            if (mgr.forward(last_token, hidden_buf.data())) {
                int tmp_id = -1;
                if (mgr.lm_head(hidden_buf.data(), logits_buf.data(), &tmp_id)) {
                    float max_l = -1e30f;
                    for (int v = 0; v < vs; v++)
                        if (logits_buf[v] > max_l) max_l = logits_buf[v];
                    double sum_exp = 0.0;
                    for (int v = 0; v < vs; v++)
                        sum_exp += exp((double)(logits_buf[v] - max_l));
                    if (sum_exp > 0 && tmp_id >= 0 && tmp_id < vs)
                        token_logprob = (double)(logits_buf[tmp_id] - max_l) - log(sum_exp);
                    else
                        token_logprob = -20.0;
                }
                next = tmp_id;
            }
        } else if (need_logprobs) {
            // Slow path: forward + lm_head + softmax for real logprobs
            std::vector<float> hidden_buf(hs);
            std::vector<float> logits_buf(vs);
            if (mgr.forward(last_token, hidden_buf.data())) {
                if (mgr.lm_head(hidden_buf.data(), logits_buf.data(), &next)) {
                    float max_l = -1e30f;
                    for (int v = 0; v < vs; v++)
                        if (logits_buf[v] > max_l) max_l = logits_buf[v];
                    double sum_exp = 0.0;
                    for (int v = 0; v < vs; v++)
                        sum_exp += exp((double)(logits_buf[v] - max_l));
                    if (sum_exp > 0 && next >= 0 && next < vs)
                        token_logprob = (double)(logits_buf[next] - max_l) - log(sum_exp);
                    else
                        token_logprob = -20.0;
                }
            }
        } else {
            // Fast path: generate() does forward+lm_head+argmax in one call
            next = mgr.generate(last_token);
        }

        if (next < 0) {
            // Backend failed — try fallback with generate()
            if (mgr.backends().size() > 1) {
                for (auto& b : mgr.backends()) {
                    if (b.available && b.functional && b.instance) {
                        mgr.select_backend(b.id);
                        active_backend_id = b.id;
                        next = mgr.generate(last_token);
                        if (next >= 0) {
                            // Compute actual logprob for cascade/adaptive strategy
                            std::vector<float> hb(hs);
                            std::vector<float> lb(vs);
                            if (need_logprobs && mgr.forward(next, hb.data())) {
                                int argmax;
                                if (mgr.lm_head(hb.data(), lb.data(), &argmax)) {
                                    float max_l = -1e30f;
                                    for (int v = 0; v < vs; v++) if (lb[v] > max_l) max_l = lb[v];
                                    double sum_exp = 0.0;
                                    for (int v = 0; v < vs; v++) sum_exp += exp((double)(lb[v] - max_l));
                                    if (sum_exp > 0 && next >= 0 && next < vs)
                                        token_logprob = (double)(lb[next] - max_l) - log(sum_exp);
                                }
                            } else {
                                token_logprob = -10.0;  // uncertain
                            }
                            break;
                        }
                    }
                }
            }
            if (next < 0) break;
        }

        output_tokens.push_back(next);
        output_logprobs.push_back(token_logprob);
        last_token = next;

        if (next == g_tokenizer.eos_id) break;
    }

    float ms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    result["tokens"] = output_tokens;
    result["text"] = g_tokenizer.decode(output_tokens);
    result["gen_ms"] = ms;
    result["gen_tokens"] = (int)output_tokens.size();
    result["per_token_backend"] = per_token_backend;
    if (timed_out) {
        result["error"] = "Generation timed out after " +
                           std::to_string((int)ms) + "ms";
        result["timed_out"] = true;
    }
    if (ms > 0) {
        result["tok_s"] = (float)output_tokens.size() / (ms / 1000.0f);
        result["ms_per_tok"] = ms / (float)output_tokens.size();
    } else {
        result["tok_s"] = 0;
        result["ms_per_tok"] = 0;
    }

    return result;
}


// ── Singleton guard ──
// Only one unified_server should ever run at a time: each instance holds a
// full model's worth of RAM, and dev iteration (rebuild + relaunch, often on
// a different --port) otherwise leaves the old one running in the background
// forever. On startup, stop whatever instance is already running before
// taking over. The lock fd is kept open for the process lifetime so the OS
// releases it automatically on exit or crash — no explicit cleanup needed.
//
// Uses XDG_RUNTIME_DIR when available (private per-user) to avoid /tmp races.
// Never kills processes based on a comm-name heuristic (fixes #615).

#ifndef _WIN32
#include <uuid/uuid.h>

static std::string lock_file_path() {
    const char* xdg = getenv("XDG_RUNTIME_DIR");
    if (xdg && xdg[0]) return std::string(xdg) + "/unified_server.lock";
    return "/tmp/unified_server.lock";
}

static void acquire_singleton_lock() {
    std::string lock_path = lock_file_path();
    const char* kLockPath = lock_path.c_str();

    int fd = open(kLockPath, O_CREAT | O_RDWR, 0644);
    if (fd < 0) {
        fprintf(stderr, "Fatal: could not open lock file %s (%s) — cannot guard against concurrent instances. Exiting.\n",
                kLockPath, strerror(errno));
        exit(EXIT_FAILURE);
    }

    // Write a unique token (PID + random uuid) so we can verify ownership
    // without relying on /proc/PID/comm matching.
    uuid_t uuid;
    uuid_generate(uuid);
    char uuid_str[37];
    uuid_unparse(uuid, uuid_str);

    char my_token[128];
    int n = snprintf(my_token, sizeof(my_token), "%d:%s", (int)getpid(), uuid_str);

    // Only try to kill previous instance if we can confirm ownership via our own token.
    // We DO NOT kill processes based on comm-name matching (see #615).
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        // Read the previous owner's token
        char old_buf[128] = {0};
        pread(fd, old_buf, sizeof(old_buf) - 1, 0);

        pid_t old_pid = 0;
        char* colon = strchr(old_buf, ':');
        if (colon) {
            *colon = '\0';
            old_pid = (pid_t)atoi(old_buf);
            *colon = ':';  // restore
        }

        if (old_pid > 0) {
            // Check if the old process still exists by sending signal 0.
            // If it doesn't exist, the lock is stale — remove and retry.
            if (kill(old_pid, 0) != 0) {
                fprintf(stderr, "Warning: stale lock from pid %d — removing\n", (int)old_pid);
                close(fd);
                unlink(kLockPath);
                fd = open(kLockPath, O_CREAT | O_RDWR, 0644);
                if (fd < 0) {
                    fprintf(stderr, "Fatal: could not recreate lock file (%s)\n", strerror(errno));
                    exit(EXIT_FAILURE);
                }
                // Retry flock
                if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
                    fprintf(stderr, "Fatal: could not acquire singleton lock (%s) — another instance running. Exiting.\n",
                            strerror(errno));
                    exit(EXIT_FAILURE);
                }
            } else {
                // Previous instance is still alive. Log and exit.
                fprintf(stderr, "Another instance is already running (pid %d). Exiting.\n", (int)old_pid);
                fprintf(stderr, "  Use a different --port or stop the existing instance first.\n");
                close(fd);
                exit(EXIT_FAILURE);
            }
        } else {
            // Can't parse token — stale or corrupted lock file. Remove and retry.
            fprintf(stderr, "Warning: unparseable lock file — removing stale lock\n");
            close(fd);
            unlink(kLockPath);
            fd = open(kLockPath, O_CREAT | O_RDWR, 0644);
            if (fd < 0) {
                fprintf(stderr, "Fatal: could not recreate lock file (%s)\n", strerror(errno));
                exit(EXIT_FAILURE);
            }
            if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
                fprintf(stderr, "Fatal: could not acquire singleton lock (%s) — another instance running. Exiting.\n",
                        strerror(errno));
                exit(EXIT_FAILURE);
            }
        }
    }

    // Write our unique token
    if (ftruncate(fd, 0) != 0) { /* best-effort */ }
    if (pwrite(fd, my_token, n, 0) != n) { /* best-effort */ }
    // fd intentionally leaked (kept open) for the process lifetime.
}
#else
// Windows: no singleton lock (POSIX-only)
static void acquire_singleton_lock() {}
#endif

// ════════════════════════════════════════════════════════════════════════
//  Main
// ════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);

    // ── Help flag — must be handled FIRST, before any hardware init ──
    // Scan for -h/--help so we never open /dev/accel, acquire locks,
    // or init NPU/GPU just to print usage info.
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [OPTIONS]\n", argv[0]);
            printf("  -p, --port PORT         HTTP port (default: 8088)\n");
            printf("  -w, --weights DIR       Model weights directory\n");
            printf("  -m, --model NAME        Model name to load\n");
            printf("  -q, --quick             Quick mode (skip full init)\n");
            printf("  -c, --cors-origin ORG   CORS origin header value\n");
            printf("  -t, --gen-timeout-ms MS Generation timeout (default: 600000)\n");
            printf("  -h, --help              Show this help and exit\n");
            exit(0);
        }
    }

    // ── Parse CLI args ──
    // Generation timeout: CLI override of g_generation_timeout_ms (issue #948)
    int cli_gen_timeout_ms = -1;
    static struct option long_opts[] = {
        {"port",          required_argument, nullptr, 'p'},
        {"weights",       required_argument, nullptr, 'w'},
        {"model",         required_argument, nullptr, 'm'},
        {"quick",         no_argument,       nullptr, 'q'},
        {"cors-origin",   required_argument, nullptr, 'c'},
        {"gen-timeout-ms", required_argument, nullptr, 't'},
        {"free-npu",      no_argument,       nullptr, 'F'},
        {nullptr, 0, nullptr, 0}
    };

    bool quick_mode = false;
    bool free_npu = false; (void)free_npu;
    std::string g_cors_origin;
    std::string g_model_name;
    int opt;
    while ((opt = getopt_long(argc, argv, "p:w:m:c:q", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'p': g_port = atoi(optarg); break;
            case 'w': g_weights_dir = optarg; break;
            case 'm': g_model_name = optarg; break;
            case 'q': quick_mode = true; break;
            case 'c': g_cors_origin = optarg; break;
            case 't': cli_gen_timeout_ms = atoi(optarg); break;
            case 'F': free_npu = true; break;
        }
    }

    // Apply CLI timeout override after env default
    if (cli_gen_timeout_ms >= 0) {
        g_generation_timeout_ms = cli_gen_timeout_ms;
    }

    acquire_singleton_lock();

    printf("\n");
    printf("╔═══════════════════════════════════════════════╗\n");
    printf("║                                               ║\n");
    printf("║   1bit.systems — Unified Inference Server    ║\n");
    printf("║   One binary, all backends, auto-failover    ║\n");
    printf("║                                               ║\n");
    printf("╚═══════════════════════════════════════════════╝\n");
    printf("\n");
    // ── Apply RLIMIT_AS as an OOM safety net (issue #948) ──
    // Set virtual address space limit: prevent a single slow request from growing
    // unbounded (e.g. zamba-7b-v1 reached 56.5GB over 11 minutes). The limit is
    // set high enough to never interfere with normal operation but low enough that
    // runaway memory allocation triggers ENOMEM (malloc returns NULL or SIGSEGV
    // in overcommit mode) rather than the kernel OOM killer taking the whole
    // process down. 256 GB leaves ample headroom for 122 GB physical RAM + swap.
#ifndef _WIN32
    {
        struct rlimit as_lim;
        as_lim.rlim_cur = 256L * 1024 * 1024 * 1024;  // 256 GB
        as_lim.rlim_max = 256L * 1024 * 1024 * 1024;
        if (setrlimit(RLIMIT_AS, &as_lim) != 0) {
            fprintf(stderr, "  ⚠  Could not set RLIMIT_AS (%s) — OOM safety net disabled\n",
                    strerror(errno));
        } else {
            printf("  ✓  RLIMIT_AS set to 256 GB (OOM safety net)\n");
        }
    }
#endif

    printf("  Weights:    %s\n", g_weights_dir.c_str());
    printf("  Port:       %d\n", g_port);
    printf("  Gen Timeout: %s\n", g_generation_timeout_ms > 0
           ? (std::to_string(g_generation_timeout_ms) + "ms").c_str()
           : "disabled");
    printf("\n");

    // ── Load tokenizer ──
    std::string tok_path = tokenizer_path();
    if (!g_tokenizer.load(tok_path)) {
        printf("  ⚠  Tokenizer not found at %s\n", tok_path.c_str());
        printf("     Using fallback tokenizer (ASCII passthrough)\n");
    } else {
        printf("  ✓  Tokenizer loaded\n");
    }

    // ── Create BackendManager ──
    auto& mgr = backend_manager();

    // Phase 1: Discover hardware
    printf("\n── Hardware Discovery ──\n");
    mgr.discover();

    // Phase 2: Configure
    mgr.set_strategy(SelectionStrategy::FASTEST);
    mgr.set_fallback_policy(FallbackPolicy::SEQUENTIAL);

    // Phase 2.5: Scan for model files
    printf("\n── Model Discovery ──\n");
    static std::vector<ModelConfig> discovered = discover_models(g_weights_dir);

    // Select model: --model flag takes priority, otherwise first discovered
    static ModelConfig current_cfg = default_model_config();
    if (!g_model_name.empty()) {
        for (auto& m : discovered) {
            if (m.model_name == g_model_name) {
                current_cfg = m;
                break;
            }
        }
        if (current_cfg.model_path.empty()) {
            printf("  ** Model '%s' not found -- using first available.\n",
                   g_model_name.c_str());
        }
    }
    if (current_cfg.model_path.empty() && !discovered.empty()) {
        current_cfg = discovered.front();
    }

    for (auto& m : discovered) {
        bool sel = (m.model_name == current_cfg.model_name);
        printf("  %s %s (%s)%s\n",
               sel ? ">" : "v",
               m.model_name.c_str(), m.model_path.c_str(),
               sel ? " [active]" : "");
    }
    if (discovered.empty()) {
        printf("  (no .gguf/.h1b files in %s)\n", g_weights_dir.c_str());
    }

    // Phase 3: Initialize
    printf("\n── Initialize ──\n");
    ModelConfig cfg = current_cfg;
    // Prefer the model's own embedded GGUF vocab over the fixed .htok/ASCII
    // tokenizer loaded above — correct per-model tokenization matters as
    // much as backend routing for arbitrary (non-Zaya) models. Falls back
    // silently (keeps whatever tokenizer was already loaded) if unavailable.
    g_tokenizer.load_from_gguf(cfg.model_path);
    BackendRoute route = select_backend_route(cfg);
    printf("  Router: %s\n", route.reason.c_str());
    bool inited = mgr.init(cfg, g_weights_dir, route.backend_ids_in_order);
    if (inited) {
        // Ensure active_idx_ points to the initialized backend
        // (init() returns true but doesn't update active_idx_)
        for (auto& b : mgr.backends()) {
            if (b.available && b.functional && b.instance) {
                mgr.select_backend(b.id);
                break;
            }
        }
        auto* active = mgr.active_info();
        printf("  ✓  Active backend: %s (%s)\n",
               active ? active->id.c_str() : "?",
               active ? active->description.c_str() : "?");
#ifndef _WIN32
        // Release /dev/accel/accel0 if the NPU backend is not active.
        // The HSA runtime opens this device during GPU backend init (even
        // for non-NPU backends like Mamba1) as a side effect of accelerator
        // enumeration on Strix Halo. When the NPU isn't being used, close
        // any spurious fds so standalone tools (npu_engine_universal) can
        // access the NPU. The GPU backends don't need it for compute.
        // See issue #1029.
        if (!active || active->type != BackendType::NPU_XRT) {
            // Step 1: Close any open /dev/accel/accel* file descriptors.
            // These are opened by the HSA runtime during GPU backend init
            // as a side effect of accelerator enumeration on Strix Halo.
            int n_closed = 0;
            DIR* fddir = opendir("/proc/self/fd");
            if (fddir) {
                struct dirent* entry;
                while ((entry = readdir(fddir)) != nullptr) {
                    if (entry->d_name[0] == '.') continue;
                    char link[256], target[128];
                    snprintf(link, sizeof(link), "/proc/self/fd/%s", entry->d_name);
                    ssize_t n = readlink(link, target, sizeof(target) - 1);
                    if (n > 0) {
                        target[n] = '\0';
                        if (strncmp(target, "/dev/accel/accel", 16) == 0) {
                            int fd = atoi(entry->d_name);
                            if (fd > 2) {
                                close(fd);
                                n_closed++;
                                printf("  ✓  Closed NPU fd %d (%s)\n", fd, target);
                            }
                        }
                    }
                }
                closedir(fddir);
            }

            // Step 2 (--free-npu only): Unmap any /dev/accel/accel* memory
            // mappings. This is risky — forcibly unmapping regions the HSA
            // runtime may still reference can crash the process. Only enable
            // with --free-npu when NPU coexistence is explicitly needed.
            // The HSA runtime mmaps ~64MB from the NPU device for DMA buffers
            // even on GPU-only workloads. Closing the fd (step 1) releases
            // the file reference but the kernel still considers the device
            // "in use" while any process has it mmap'd. Force-unmap those
            // regions so the device is truly free for standalone tools.
            // See issue #1029.
            FILE* maps = fopen("/proc/self/maps", "r");
            if (maps) {
                char line[512];
                while (fgets(line, sizeof(line), maps)) {
                    // Parse: "7c2f74000000-7c2f78000000 rw-s ... /dev/accel/accel0"
                    unsigned long start = 0, end = 0;
                    char perms[8] = {0}, path[256] = {0};
                    if (sscanf(line, "%lx-%lx %7s %*s %*s %*s %255s",
                               &start, &end, perms, path) >= 3) {
                        if (strstr(path, "/dev/accel/accel") == path) {
                            size_t len = end - start;
                            if (munmap((void*)start, len) == 0) {
                                n_closed++;
                                printf("  ✓  Unmapped NPU region 0x%lx-0x%lx (%zu MB) — device %s freed\n",
                                       start, end, len / (1024*1024), path);
                            } else {
                                fprintf(stderr, "  ⚠  munmap of 0x%lx failed: %s\n",
                                        start, strerror(errno));
                            }
                        }
                    }
                }
                fclose(maps);
            }

            if (n_closed > 0) {
                printf("  ✓  NPU device released (%d handles) — free for standalone tools\n", n_closed);
            }
        }
#endif
    } else {
        printf("  ⚠  No backend initialized (weights missing or no hardware)\n");
        printf("     Server starts in discovery-only mode.\n");
    }

    // Phase 4: Benchmark active backend only
    // (benchmarking all backends can crash on backends that don't support
    //  the current model format, e.g. hip_gpu expects Zaya .bin format)
    if (inited) {
        auto* active_info_raw = mgr.active_info();
        if (active_info_raw) {
            int bench_tokens = quick_mode ? 1 : 5;
            printf("\n── Benchmark (%d token%s) ──\n", bench_tokens, bench_tokens == 1 ? "" : "s");
            printf("  %s... ", active_info_raw->id.c_str());
            fflush(stdout);
            float ms = active_info_raw->instance->benchmark(bench_tokens);
            printf("%.1f ms/tok\n", ms);
            // Phase 5's build_performance_table() reads BackendInfo::score, which
            // benchmark_all() would normally set — but Phase 4 deliberately benchmarks
            // only the active backend (see comment above) and bypasses benchmark_all(),
            // so without this the score stays 0 and the performance table prints a
            // bogus "0 tok/s" for the very backend we just measured.
            mgr.set_score(active_info_raw->id, ms);
        }
    }

    // ── Phase 5: Initialize Strategy Engine ──
    printf("\n── Strategy Engine ──\n");
    {
        auto perf_table = build_performance_table(mgr, current_cfg.model_name);
        printf("  Performance table (%zu backends):\n", perf_table.size());
        for (auto& r : perf_table) {
            printf("    %-20s -> %-12s (%.0f tok/s)\n",
                   r.model_pattern.c_str(), r.backend.c_str(), r.speed_tok_s);
        }
        // Default to adaptive strategy (the "true agent")
        g_strategy_engine.init("adaptive", mgr, perf_table);
        printf("  Active strategy: %s\n", g_strategy_engine.name());
        printf("     Per-token routing: set X-Router-Strategy header\n");
        printf("     Change at runtime: POST /v1/strategy/select\n");
        printf("     Status:            GET /v1/router\n");
    }

    // ── Phase 6: Start Agent Watchdog ──
    // NOTE: Disabled pending investigation of heap corruption (issue #932).
    // The watchdog thread races with httplib completion handlers when it
    // calls benchmark_all() while a generate() call is in progress.
    // printf("\n── Agent Watchdog ──\n");
    // {
    //     if (!g_watchdog) {
    //         g_watchdog = new AgentWatchdog(g_strategy_engine, mgr);
    //     }
    //     g_watchdog->start();
    // }

    // Quick mode: re-profile in background after server starts
    if (quick_mode) {
        printf("  ⚡ Quick mode: full benchmark deferred to background\n");
    }

    // ── HTTP Server ──
    httplib::Server svr;

    // Limit request body size to prevent memory exhaustion from oversized payloads
    svr.set_payload_max_length(16 * 1024 * 1024); // 16 MiB

    // ── CORS middleware (only enabled when --cors-origin is set) ──
    if (!g_cors_origin.empty()) {
        svr.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
            if (req.method == "OPTIONS") {
                res.set_header("Access-Control-Allow-Origin", g_cors_origin);
                if (g_cors_origin == "*") {
                    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                    res.set_header("Access-Control-Allow-Headers", "Content-Type, X-Backend, X-Strategy, Authorization");
                }
                res.status = 204;
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }

    auto add_cors = [&](httplib::Response& res) {
        if (!g_cors_origin.empty())
            res.set_header("Access-Control-Allow-Origin", g_cors_origin);
    };

    // ── GET /v1/health — Backend status dashboard ──
    svr.Get("/v1/health", [&](const httplib::Request&, httplib::Response& res) {
        // Lock both mutexes consistently — health reads mgr backends (g_config_mutex)
        // and strategy state (g_strategy_mutex).
        std::lock(g_config_mutex, g_strategy_mutex);
        std::lock_guard<std::mutex> _l1(g_config_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> _l2(g_strategy_mutex, std::adopt_lock);
        json j = health_json(mgr);
        j["version"] = "unified-server-1.0";
        j["model"] = current_cfg.model_name;
        j["weights_dir"] = g_weights_dir;
        j["uptime"] = std::to_string(time(nullptr)) + "s";
        j["generation_timeout_ms"] = g_generation_timeout_ms;
        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    // ── GET /v1/models — List models ──
    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        std::lock(g_config_mutex, g_strategy_mutex);
        std::lock_guard<std::mutex> _l1(g_config_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> _l2(g_strategy_mutex, std::adopt_lock);
        auto* active = mgr.active_info();
        json j;
        j["object"] = "list";
        json models = json::array();
        // Add all discovered models
        for (auto& m : discovered) {
            json info;
            info["id"] = m.model_name;
            info["object"] = "model";
            info["created"] = 0;
            info["owned_by"] = "1bit-systems";
            info["backend"] = "auto";
            info["details"] = {{
                {"hidden", m.hidden}, {"layers", m.n_layers},
                {"heads", m.n_heads}, {"kv_heads", m.n_kv_heads},
                {"vocab", m.vocab}, {"max_seq_len", m.max_seq_len}
            }};
            models.push_back(info);
        }
        // Also add the active backend model if different
        if (active) {
            bool found = false;
            for (auto& m : discovered) {
                if (m.model_name == active->id) { found = true; break; }
            }
            if (!found) models.push_back(model_info_json(active, current_cfg.model_name));
        }
        j["data"] = models;
        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    // ── POST /v1/chat/completions — OpenAI-compatible chat ──
    // Uses the strategy engine for per-token routing when X-Router-Strategy
    // header is set, or falls back to single-backend selection.
    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            json err = {{"error", "Invalid JSON body"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::string backend_id = resolve_backend_id(req);
        SelectionStrategy strategy = resolve_strategy(req);

        // Check for strategy engine routing
        std::string strategy_name = resolve_strategy_name(req);
        bool use_strategy_engine = !strategy_name.empty();

        // Extract messages and build prompt + user message for content routing
        std::string prompt;
        std::string last_user_msg;
        std::vector<VlProcessor> vision_images;  // holds processed images from content parts
        if (body.contains("messages") && body["messages"].is_array()) {
            for (auto& msg : body["messages"]) {
                std::string role = msg.value("role", "user");
                std::string content;
                if (msg["content"].is_string()) {
                    content = msg["content"].get<std::string>();
                } else if (msg["content"].is_array()) {
                    for (auto& part : msg["content"]) {
                        if (part.value("type", "") == "text") {
                            content += part.value("text", "");
                        } else if (part.value("type", "") == "image_url") {
                            // Load + process image for VL models.
                            // This stores processed pixels; the actual ViT
                            // forward pass happens in generate_completion()
                            // when it detects vision_state has data.
                            std::string url;
                            const auto& iu = part["image_url"];
                            if (iu.is_string()) url = iu.get<std::string>();
                            else if (iu.is_object() && iu.contains("url")) url = iu["url"].get<std::string>();
                            if (!url.empty()) {
                                std::vector<unsigned char> raw;
                                if (vl_is_data_url(url)) {
                                    raw = vl_decode_base64_image(url);
                                } else {
                                    raw = vl_download_image(url);
                                }
                                if (!raw.empty()) {
                                    VlProcessor vp;
                                    if (vp.load_from_memory(raw.data(), raw.size(), 224, 224,
                                                             VL_MEAN_QWEN2VL, VL_STD_QWEN2VL)) {
                                        vision_images.push_back(std::move(vp));
                                        content += "[image]"; // placeholder in text
                                        fprintf(stderr, "[vision] loaded image: %dx%d -> %dx%d\n",
                                                vp.orig_width(), vp.orig_height(),
                                                vp.width(), vp.height());
                                    }
                                }
                            }
                        }
                    }
                }
                prompt += role + ": " + content + "\n";
                if (role == "user") last_user_msg = content;
            }
        } else if (body.contains("prompt") && body["prompt"].is_string()) {
            prompt = body["prompt"].get<std::string>();
            last_user_msg = prompt;
        }

        int max_tokens = body.value("max_tokens", 256);
        if (max_tokens < 1) max_tokens = 1;
        if (max_tokens > 32768) max_tokens = 32768;
        std::string req_model = body.value("model", "");

        // ── Serialize all compute against the single shared backend context ──
        // Everything below — mgr.set_strategy, the mgr.init model switch, the
        // vision mgr.generate/forward_embed injection, and generate_completion's
        // mgr.reset/mgr.generate loop — mutates one BackendManager inference
        // state. Concurrent requests here race on the KV cache / active-backend
        // pointer (AUDIT_ISSUES.md #2). Held to the end of the handler; the short
        // g_config_mutex / g_strategy_mutex sections below nest *inside* it.
        // Metadata endpoints (/v1/health, /v1/models) take only config+strategy,
        // so they are NOT blocked by an in-flight decode — preserving the #701
        // goal of keeping health responsive during generation.
        std::lock_guard<std::mutex> infer_lock(g_inference_mutex);

        // ── Phase 1: Model-switch check under locks only (#701 fix) ──
        // Defer slow I/O (load_from_gguf, mgr.init) to outside the config lock.
        mgr.set_strategy(strategy);

        ModelConfig switch_cfg;
        bool need_model_switch = false;
        {
            std::lock(g_config_mutex, g_strategy_mutex);
            std::lock_guard<std::mutex> _l1(g_config_mutex, std::adopt_lock);
            std::lock_guard<std::mutex> _l2(g_strategy_mutex, std::adopt_lock);

            if (!req_model.empty()) {
                for (auto& dm : discovered) {
                    if (dm.model_name == req_model &&
                        (dm.hidden != current_cfg.hidden || dm.n_layers != current_cfg.n_layers)) {
                        printf("[model] switching to %s (%d layers, %d hidden)\n",
                               dm.model_name.c_str(), dm.n_layers, dm.hidden);
                        switch_cfg = dm;
                        current_cfg = dm;
                        need_model_switch = true;
                        break;
                    }
                }
            }
        } // release both mutexes

        // ── Phase 1b: Model-switch I/O outside locks (#701 fix) ──
        if (need_model_switch) {
            g_tokenizer.load_from_gguf(switch_cfg.model_path);
            BackendRoute swrt = select_backend_route(switch_cfg);
            mgr.init(switch_cfg, g_weights_dir, swrt.backend_ids_in_order);
        }

        // Tokenize with logprobs for cascade strategy (#696 fix: config_mutex only)
        std::vector<int> prompt_tokens;
        std::vector<double> prompt_logprobs;
        {
            std::lock_guard<std::mutex> cfg_lock(g_config_mutex);
            if (use_strategy_engine) {
                prompt_tokens = g_tokenizer.encode_with_logprobs(prompt, prompt_logprobs);
            } else {
                prompt_tokens = g_tokenizer.encode(prompt);
            }
            if (prompt_tokens.empty()) {
                prompt_tokens = {g_tokenizer.bos_id};
            }
        }

        // Capture strategy engine pointer under its mutex (#696 fix)
        StrategyEngine* se = nullptr;
        {
            std::lock_guard<std::mutex> strat_lock(g_strategy_mutex);
            se = use_strategy_engine ? &g_strategy_engine : nullptr;
        }
        // ── Inject vision embeddings (if any) before text generation ──
        // Runs forward_embed() for each vision token, splicing the image into
        // the KV cache before the text prompt is processed.
        // This path activates when the message content includes image_url parts.
        // For full ViT forward (mmproj GGUF), use tools/vision_server.cpp.
        if (!vision_images.empty()) {
            auto* active = mgr.active_info();
            int hidden = (active && active->instance) ? active->instance->cfg.hidden : 2048;
            if (hidden <= 0) hidden = 2048;

            // Wrap vision tokens with <|vision_start|> / <|vision_end|>
            // (Qwen2-VL convention: token IDs 151652/151653)
            const int VISION_START = 151652;
            const int VISION_END   = 151653;
            const int VISION_TOKENS_PER_IMG = 64; // 16x16 patches / 4 merger

            mgr.generate(VISION_START);
            std::vector<float> embed_buf(hidden, 0.0f);
            for (auto& vp : vision_images) {
                (void)vp;  // used when real ViT forward is implemented
                for (int t = 0; t < VISION_TOKENS_PER_IMG; t++) {
                    // TODO: replace with real ViT forward from vision_qwen2vl_poc
                    // For now, feed zeros — the KV cache advances but content is
                    // dummy. Full integration requires:
                    //   1. Load mmproj GGUF
                    //   2. Run ViT forward (CPU or GPU)
                    //   3. Use projected embeddings here
                    // See tools/vision_server.cpp for the full pipeline.
                    if (active && active->instance) {
                        active->instance->forward_embed(embed_buf.data());
                    }
                }
            }
            mgr.generate(VISION_END);
            fprintf(stderr, "[vision] injected %zu images (%d tokens each)\n",
                    vision_images.size(), VISION_TOKENS_PER_IMG);
        }

        // Generate with strategy-aware routing (#696 fix: no global lock held)
        json gen_result = generate_completion(mgr, prompt_tokens, prompt_logprobs,
                                               max_tokens, backend_id,
                                               se, last_user_msg);

        // Build OpenAI-compatible response
        json response;
        response["id"] = "cmpl-" + std::to_string(time(nullptr));
        response["object"] = "chat.completion";
        response["created"] = time(nullptr);
        response["model"] = current_cfg.model_name;

        json choice;
        choice["index"] = 0;
        json message;
        message["role"] = "assistant";
        message["content"] = gen_result.value("text", "");
        std::string finish_reason = "stop";
        if (gen_result.value("timed_out", false)) {
            finish_reason = "timeout";
        } else if (gen_result.contains("error")) {
            finish_reason = "error";
        }
        choice["message"] = message;
        choice["finish_reason"] = finish_reason;

        json usage;
        usage["prompt_tokens"] = (int)prompt_tokens.size();
        usage["completion_tokens"] = gen_result.value("gen_tokens", 0);
        usage["total_tokens"] = (int)prompt_tokens.size() + gen_result.value("gen_tokens", 0);

        response["choices"] = json::array({choice});
        response["usage"] = usage;

        // Add routing metadata as headers
        res.set_header("X-Backend-Id", gen_result.value("backend_used", "unknown"));
        res.set_header("X-Strategy", gen_result.value("strategy", "none"));
        char tok_s_buf[32];
        snprintf(tok_s_buf, sizeof(tok_s_buf), "%.1f", gen_result.value("tok_s", 0.0f));
        res.set_header("X-Backend-Tok-s", tok_s_buf);

        // Include per-token backend info when using strategy engine
        if (use_strategy_engine && gen_result.contains("per_token_backend")) {
            response["per_token_backend"] = gen_result["per_token_backend"];
        }

        // error_handler_t::replace: decoded model text isn't guaranteed valid
        // UTF-8 (byte-level BPE tokenizers like blackmamba's GPT-NeoX-20B
        // .htok can decode to a stray byte sequence at a token boundary) --
        // strict (default) dump() throws type_error.316, which previously
        // escaped as an uncaught exception -> empty 500, no app-level log line.
        res.set_content(response.dump(2, ' ', false, json::error_handler_t::replace), "application/json");
        add_cors(res);

        // Log to stdout
        printf("  [%s] %d tokens → %d tokens (%.1f tok/s, backend: %s, strategy: %s)\n",
               backend_id.c_str(),
               (int)prompt_tokens.size(),
               gen_result.value("gen_tokens", 0),
               gen_result.value("tok_s", 0.0f),
               gen_result.value("backend_used", "?").c_str(),
               gen_result.value("strategy", "none").c_str());
    });

    // ── POST /v1/completions — Legacy completion endpoint ──
    svr.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            json err = {{"error", "Invalid JSON body"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::string backend_id = resolve_backend_id(req);

        // Serialize compute against the single shared backend context, exactly
        // as /v1/chat/completions does (AUDIT_ISSUES.md #2). Held across
        // set_strategy + tokenize + generate_completion; the g_config_mutex
        // tokenize section below nests inside it.
        std::lock_guard<std::mutex> infer_lock(g_inference_mutex);

        mgr.set_strategy(resolve_strategy(req));

        // ── Model-switch from body["model"], same as /v1/chat/completions ──
        // Fixes the bug where /v1/completions silently ignored the model field
        // and always used whatever backend was last loaded.
        std::string req_model = body.value("model", "");
        ModelConfig switch_cfg;
        bool need_model_switch = false;
        {
            std::lock(g_config_mutex, g_strategy_mutex);
            std::lock_guard<std::mutex> _l1(g_config_mutex, std::adopt_lock);
            std::lock_guard<std::mutex> _l2(g_strategy_mutex, std::adopt_lock);

            if (!req_model.empty()) {
                for (auto& dm : discovered) {
                    if (dm.model_name == req_model &&
                        (dm.hidden != current_cfg.hidden || dm.n_layers != current_cfg.n_layers)) {
                        printf("[model] /v1/completions switching to %s (%d layers, %d hidden)\n",
                               dm.model_name.c_str(), dm.n_layers, dm.hidden);
                        switch_cfg = dm;
                        current_cfg = dm;
                        need_model_switch = true;
                        break;
                    }
                }
            }
        } // release both mutexes

        // ── Model-switch I/O outside locks (#701 fix) ──
        if (need_model_switch) {
            g_tokenizer.load_from_gguf(switch_cfg.model_path);
            BackendRoute swrt = select_backend_route(switch_cfg);
            mgr.init(switch_cfg, g_weights_dir, swrt.backend_ids_in_order);
        }

        // ── Tokenize under config_mutex only (#696 fix) ──
        std::vector<int> prompt_tokens;
        {
            std::lock_guard<std::mutex> cfg_lock(g_config_mutex);
            if (body.contains("tokens") && body["tokens"].is_array()) {
                for (auto& t : body["tokens"]) {
                    if (t.is_number_integer()) prompt_tokens.push_back(t.get<int>());
                }
            } else if (body.contains("prompt") && body["prompt"].is_string()) {
                try {
                    prompt_tokens = g_tokenizer.encode(body["prompt"].get<std::string>());
                } catch (const std::exception& e) {
                    fprintf(stderr, "[completions] encode error: %s\n", e.what());
                }
            }
            if (prompt_tokens.empty()) {
                prompt_tokens = {g_tokenizer.bos_id};
            }
        }

        int max_tokens = body.value("max_tokens", 256);
        if (max_tokens < 1) max_tokens = 1;
        if (max_tokens > 32768) max_tokens = 32768;

        std::vector<double> empty_logprobs;
        json gen_result;
        try {
            gen_result = generate_completion(mgr, prompt_tokens, empty_logprobs, max_tokens, backend_id);
        } catch (const std::exception& e) {
            fprintf(stderr, "[completions] generate error: %s\n", e.what());
            gen_result = {{"error", std::string("Generation failed: ") + e.what()}};
        } catch (...) {
            fprintf(stderr, "[completions] unknown error\n");
            gen_result = {{"error", "Generation failed: unknown error"}};
        }

        // Check for generation errors before accessing result fields (issue #959)
        if (gen_result.contains("error")) {
            json err_resp = {{"error", gen_result["error"]}};
            res.status = 500;
            res.set_content(err_resp.dump(), "application/json");
            return;
        }

        json response;
        response["tokens"] = gen_result["tokens"];
        response["text"] = gen_result.value("text", "");
        response["gen_ms"] = gen_result.value("gen_ms", 0);
        response["tok_s"] = gen_result.value("tok_s", 0.0f);
        response["backend_used"] = gen_result.value("backend_used", "unknown");

        res.set_header("X-Backend-Id", gen_result.value("backend_used", "unknown"));
        // See error_handler_t::replace note on the /v1/chat/completions handler above.
        res.set_content(response.dump(2, ' ', false, json::error_handler_t::replace), "application/json");
        add_cors(res);
    });

    // ── GET /v1/router — Strategy engine status ──
    svr.Get("/v1/router", [&](const httplib::Request&, httplib::Response& res) {
        json j;
        // Lock strategy mutex for consistent read of strategy state + watchdog (fixes #364)
        std::lock_guard<std::mutex> lock(g_strategy_mutex);
        j["strategy"] = g_strategy_engine.name();
        j["state"] = json::parse(g_strategy_engine.state_json());
        j["watchdog_running"] = g_watchdog ? g_watchdog->running() : false;

        // Add backend metrics
        json backends = json::array();
        for (auto* pm : mgr.monitor_stats()->all_metrics()) {
            json bj;
            bj["id"] = pm->backend_id;
            bj["inferences"] = pm->inferences.load();
            bj["failures"] = pm->failures.load();
            bj["fallbacks"] = pm->fallbacks.load();
            bj["tokens_per_second"] = pm->tokens_per_second.load();
            bj["avg_ms"] = pm->recent_ms.avg();
            bj["p50_ms"] = pm->recent_ms.p50();
            bj["p95_ms"] = pm->recent_ms.p95();
            bj["healthy"] = pm->healthy.load();
            backends.push_back(bj);
        }
        j["backends"] = backends;

        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    // ── POST /v1/strategy/select — Change strategy at runtime ──
    svr.Post("/v1/strategy/select", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            json err = {{"error", "Invalid JSON body"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::string name = body.value("strategy", "");
        if (name.empty()) {
            json err = {{"error", "Missing 'strategy' field"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Build performance table for strategy init
        auto perf_table = build_performance_table(mgr, current_cfg.model_name);
        // Lock strategy mutex while reinitializing the engine (fixes #364)
        std::lock_guard<std::mutex> lock(g_strategy_mutex);
        bool ok = g_strategy_engine.init(name, mgr, perf_table);

        json j;
        j["ok"] = ok;
        j["strategy"] = name;
        j["state"] = json::parse(g_strategy_engine.state_json());
        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    // ── POST /v1/backend/select — Manually select backend ──
    svr.Post("/v1/backend/select", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            json err = {{"error", "Invalid JSON body"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::string backend_id = body.value("backend", "");
        bool ok = false;
        json j;
        {
            // mgr.select_backend() mutates the shared active-backend pointer
            // that generate_completion() also flips mid-decode. Guard it with
            // the SAME g_inference_mutex the decode path holds (not g_config_mutex,
            // which the decode path no longer holds since the #696 change), so a
            // backend switch can't land in the middle of a live generate
            // (AUDIT_ISSUES.md #2).
            std::lock_guard<std::mutex> lock(g_inference_mutex);
            if (!backend_id.empty()) {
                ok = mgr.select_backend(backend_id);
            }
            j["ok"] = ok;
            j["selected"] = backend_id;
            if (ok) {
                auto* active = mgr.active_info();
                j["active_backend"] = active ? active->id : "none";
                j["active_type"] = active ? backend_name(active->type) : "none";
            } else {
                j["error"] = "Backend '" + backend_id + "' not found or not functional";
            }
        }
        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    // ── GET /v1/backend/status — Full backend report ──
    svr.Get("/v1/backend/status", [&](const httplib::Request&, httplib::Response& res) {
        json j;
        // g_strategy_mutex alone (fixes #364) only protects g_strategy_engine
        // + g_watchdog — mgr itself (report/backends/active_info/
        // active_backend, all read below) is the inference handlers'
        // g_config_mutex's responsibility. Reading mgr here under a
        // different mutex than the one that guards it during an in-flight
        // inference call doesn't actually serialize anything (fixes #2).
        // std::lock (not two separate lock_guards) avoids a lock-order
        // deadlock against any other path that might acquire the same two
        // mutexes in the opposite order.
        std::lock(g_config_mutex, g_strategy_mutex);
        std::lock_guard<std::mutex> lock1(g_config_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(g_strategy_mutex, std::adopt_lock);
        j["strategy"] = g_strategy_engine.name();
        j["watchdog_running"] = g_watchdog ? g_watchdog->running() : false;
        j["report"] = mgr.report();

        json backends = json::array();
        for (auto& b : mgr.backends()) {
            json bj;
            bj["id"] = b.id;
            bj["type"] = backend_name(b.type);
            bj["tier"] = tier_name(b.tier);
            bj["available"] = b.available;
            bj["functional"] = b.functional;
            bj["score_ms_per_tok"] = b.score;
            bj["priority"] = b.priority;
            bj["total_inferences"] = b.total_inferences;
            bj["failed_inferences"] = b.failed_inferences;
            bj["cumulative_ms"] = b.cumulative_ms;
            backends.push_back(bj);
        }
        j["backends"] = backends;

        auto* active = mgr.active_info();
        if (active) j["active"] = active->id;
        else j["active"] = "none";

        j["initialized"] = mgr.active_backend() != nullptr;
        j["weights_dir"] = g_weights_dir;

        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    // ---- GET / --- Root health check ----
    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        json j;
        j["service"] = "1bit.systems --- One binary, all backends, intelligent routing";
        j["version"] = "1.0";
        // See /v1/backend/status above: mgr.active_backend() needs
        // g_config_mutex, not just g_strategy_mutex (fixes #2/#364).
        std::lock(g_config_mutex, g_strategy_mutex);
        std::lock_guard<std::mutex> lock1(g_config_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(g_strategy_mutex, std::adopt_lock);
        j["status"] = mgr.active_backend() ? "ready" : "initializing";
        j["strategy"] = g_strategy_engine.name();
        j["endpoints"] = {
            "/v1/health",
            "/v1/models",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/router",
            "/v1/strategy/select",
            "/v1/backend/select",
            "/v1/backend/status"
        };
        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    // ── Start server ──
    printf("\n──────────────────────────────────────────────\n");
    printf("  1bit.systems — Agent Inference Server\n");
    printf("──────────────────────────────────────────────\n");
    printf("  Port:    %d\n", g_port);
    printf("  Backend: %s\n", mgr.active_info() ? mgr.active_info()->id.c_str() : "none");
    printf("  Strategy: %s\n", g_strategy_engine.name());
    printf("  Watchdog: %s\n", (g_watchdog && g_watchdog->running()) ? "running" : "stopped");
    printf("──────────────────────────────────────────────\n");
    printf("  Endpoints:\n");
    printf("    GET  /v1/health            — Backend status + metrics\n");
    printf("    GET  /v1/models            — List available models\n");
    printf("    POST /v1/chat/completions  — Chat with strategy routing\n");
    printf("    POST /v1/completions       — Legacy completion\n");
    printf("    GET  /v1/router            — Strategy engine status\n");
    printf("    POST /v1/strategy/select   — Change strategy at runtime\n");
    printf("    POST /v1/backend/select    — Select specific backend\n");
    printf("    GET  /v1/backend/status    — Full backend report\n");
    printf("──────────────────────────────────────────────\n");
    printf("\n  Try it:\n");
    printf("    curl http://127.0.0.1:%d/v1/router\n", g_port);
    printf("    curl -X POST http://127.0.0.1:%d/v1/chat/completions \\\n", g_port);
    printf("      -H \"X-Router-Strategy: cascade\" \\\n");
    printf("      -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'\n");
    printf("\n  Quick start: add --quick to skip full benchmark\n");
    printf("  Press Ctrl+C to stop.\n");
    printf("──────────────────────────────────────────────\n\n");

    if (!svr.listen("127.0.0.1", g_port)) {
        fprintf(stderr, "Failed to start server on port %d\n", g_port);
        return 1;
    }

    // ── Cleanup ──
    printf("\nShutting down...\n");
    if (g_watchdog) {
        g_watchdog->stop();
        delete g_watchdog;
        g_watchdog = nullptr;
    }
    mgr.destroy();
    printf("Done.\n");
    return 0;
}
