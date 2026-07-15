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
#include "rocm_cpp/tokenizer.h"

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
#include <getopt.h>
#include <fstream>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include "strategy_engine.h"
#include "agent_watchdog.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Globals ──
static std::atomic<bool> keep_running{true};
static std::string g_weights_dir = "/tmp/zaya_weights";
static int g_port = 8088;

// ── Strategy engine + agent watchdog (global for HTTP handler access) ──
static StrategyEngine g_strategy_engine;
static AgentWatchdog* g_watchdog = nullptr;

// ── Tokenizer: uses RCPP BPE tokenizer (.htok) when available, falls back
// to SimpleTokenizer (ASCII + UTF-8 byte passthrough) when not.
// The RCPP tokenizer is linked via librocm_cpp and reads .htok binary format.
struct SimpleTokenizer {
    int bos_id = 2;
    int eos_id = 1;
    bool use_bpe = false;
    rcpp_tokenizer_t* bpe_tok = nullptr;

    bool load(const std::string& vocab_path) {
        if (vocab_path.size() >= 4 && vocab_path.substr(vocab_path.size() - 4) == ".htok") {
            rcpp_tokenizer_t* tok = nullptr;
            rcpp_status_t st = rcpp_tokenizer_load(vocab_path.c_str(), &tok);
            if (st == RCPP_OK && tok) {
                bpe_tok = tok;
                use_bpe = true;
                bos_id = rcpp_tokenizer_bos_id(bpe_tok);
                eos_id = rcpp_tokenizer_eos_id(bpe_tok);
                fprintf(stderr, "Loaded BPE tokenizer from %s (BOS=%d, EOS=%d)\n",
                        vocab_path.c_str(), bos_id, eos_id);
                return true;
            }
            fprintf(stderr, "Failed to load BPE tokenizer from %s\n", vocab_path.c_str());
        }
        fprintf(stderr, "No .htok at %s — using fallback ASCII tokenizer\n", vocab_path.c_str());
        return false;
    }

    ~SimpleTokenizer() {
        if (bpe_tok) rcpp_tokenizer_free(bpe_tok);
    }

    std::vector<int> encode(const std::string& text) {
        if (use_bpe && bpe_tok) {
            std::vector<int> r(4096);
            size_t out_n = 0;
            rcpp_status_t st = rcpp_tokenizer_encode(bpe_tok, text.c_str(), text.size(),
                                                      1, r.data(), r.size(), &out_n);
            if (st == RCPP_OK && out_n > 0) {
                r.resize(out_n);
                return r;
            }
            return {bos_id};
        }
        // Fallback: ASCII + UTF-8 byte passthrough
        std::vector<int> r = {bos_id};
        for (unsigned char c : text) {
            if (c >= 32 && c <= 126)
                r.push_back((int)c + 100);
            else if (c != 0)
                r.push_back((int)c + 200);
        }
        return r;
    }

    /// Encode with per-token log-probabilities (for cascade strategy).
    /// Returns token IDs and fills logprobs_out with corresponding logprobs.
    std::vector<int> encode_with_logprobs(const std::string& text,
                                           std::vector<double>& logprobs_out) {
        logprobs_out.clear();
        if (use_bpe && bpe_tok) {
            std::vector<int> r(4096);
            std::vector<double> lp(4096);
            size_t out_n = 0;
            rcpp_status_t st = rcpp_tokenizer_encode_with_logprobs(
                bpe_tok, text.c_str(), text.size(),
                1, r.data(), lp.data(), r.size(), &out_n);
            if (st == RCPP_OK && out_n > 0) {
                r.resize(out_n);
                logprobs_out.assign(lp.begin(), lp.begin() + out_n);
                return r;
            }
            return {bos_id};
        }
        // Fallback: encode without logprobs
        auto tokens = encode(text);
        logprobs_out.resize(tokens.size(), -1.0);
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) {
        if (use_bpe && bpe_tok) {
            std::string r(4096, '\0');
            size_t out_len = 0;
            rcpp_status_t st = rcpp_tokenizer_decode(bpe_tok, tokens.data(), tokens.size(),
                                                      r.data(), r.size(), &out_len);
            if (st == RCPP_OK && out_len > 0) {
                r.resize(out_len);
                return r;
            }
            return "";
        }
        // Fallback: ASCII + UTF-8 byte passthrough
        std::string r;
        for (int v : tokens) {
            if (v == bos_id || v == eos_id) continue;
            if (v > 100 && v < 200)
                r += (char)(v - 100);
            else if (v > 200 && v < 456)
                r += (char)(v - 200);
            else {
                r += '['; r += std::to_string(v); r += ']';
            }
        }
        return r;
    }
};

static SimpleTokenizer g_tokenizer;

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
static json model_info_json(const BackendInfo* active) {
    json j;
    j["id"] = "zaya";
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
    int last_token = prompt_tokens.empty() ? g_tokenizer.bos_id : prompt_tokens.back();

    // Prefill: process prompt tokens
    for (size_t i = 0; i + 1 < prompt_tokens.size(); i++) {
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

        // ── Generate the token ──
        int next = mgr.generate(last_token);
        if (next < 0) {
            // Backend failed — try fallback
            if (strategy_engine && mgr.backends().size() > 1) {
                // Find first functional backend as fallback
                for (auto& b : mgr.backends()) {
                    if (b.available && b.functional && b.instance) {
                        mgr.select_backend(b.id);
                        active_backend_id = b.id;
                        // Re-try
                        next = mgr.generate(last_token);
                        if (next >= 0) break;
                    }
                }
            }
            if (next < 0) break;
        }

        output_tokens.push_back(next);
        last_token = next;

        // Extract logprob if tokenizer supports it (approximate from BPE)
        if (g_tokenizer.use_bpe && g_tokenizer.bpe_tok) {
            // Use token's BPE rank as proxy for logprob
            double lp = rcpp_tokenizer_logprob(g_tokenizer.bpe_tok, next);
            output_logprobs.push_back(lp);
        } else {
            output_logprobs.push_back(0.0);
        }

        if (next == g_tokenizer.eos_id) break;
    }

    float ms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    result["tokens"] = output_tokens;
    result["text"] = g_tokenizer.decode(output_tokens);
    result["gen_ms"] = ms;
    result["gen_tokens"] = (int)output_tokens.size();
    result["per_token_backend"] = per_token_backend;
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
static const char* kLockPath = "/tmp/unified_server.lock";

static bool pid_is_unified_server(pid_t pid) {
    char comm_path[64];
    snprintf(comm_path, sizeof(comm_path), "/proc/%d/comm", (int)pid);
    FILE* f = fopen(comm_path, "r");
    if (!f) return false;
    char comm[64] = {0};
    if (!fgets(comm, sizeof(comm), f)) { fclose(f); return false; }
    fclose(f);
    return strncmp(comm, "unified_server", 14) == 0;
}

static void acquire_singleton_lock() {
    int fd = open(kLockPath, O_CREAT | O_RDWR, 0644);
    if (fd < 0) {
        fprintf(stderr, "Fatal: could not open lock file %s (%s) — cannot guard against concurrent instances. Exiting.\n",
                kLockPath, strerror(errno));
        exit(EXIT_FAILURE);
    }

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        char buf[32] = {0};
        pread(fd, buf, sizeof(buf) - 1, 0);
        pid_t old_pid = (pid_t)atoi(buf);

        if (old_pid > 0 && pid_is_unified_server(old_pid)) {
            fprintf(stderr, "Found existing unified_server instance (pid %d) — stopping it\n", (int)old_pid);
            kill(old_pid, SIGTERM);
            for (int i = 0; i < 50; i++) {  // wait up to 5s for graceful shutdown
                if (kill(old_pid, 0) != 0) break;
                usleep(100 * 1000);
            }
            if (kill(old_pid, 0) == 0) {
                fprintf(stderr, "  pid %d didn't exit in time, sending SIGKILL\n", (int)old_pid);
                kill(old_pid, SIGKILL);
                usleep(500 * 1000);  // longer wait for SIGKILL to process (fixes #fix)
            }
        }

        // Retry now that the previous holder (if any) should be gone.
        // If flock still fails, the lock file may be stale from a before-SIGKILL state.
        // Remove and recreate it rather than failing immediately.
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            fprintf(stderr, "Warning: lock held after killing old instance — removing stale lock\n");
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

    if (ftruncate(fd, 0) != 0) { /* best-effort */ }
    char pid_buf[32];
    int n = snprintf(pid_buf, sizeof(pid_buf), "%d", (int)getpid());
    if (pwrite(fd, pid_buf, n, 0) != n) { /* best-effort */ }
    // fd intentionally leaked (kept open) for the process lifetime.
}

// ════════════════════════════════════════════════════════════════════════
//  Main
// ════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);

    // ── Parse CLI args ──
    static struct option long_opts[] = {
        {"port",    required_argument, nullptr, 'p'},
        {"weights", required_argument, nullptr, 'w'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "p:w:", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'p': g_port = atoi(optarg); break;
            case 'w': g_weights_dir = optarg; break;
        }
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
    printf("  Weights: %s\n", g_weights_dir.c_str());
    printf("  Port:    %d\n", g_port);
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

    // Phase 3: Initialize
    printf("\n── Initialize ──\n");
    ModelConfig cfg = default_model_config();
    bool inited = mgr.init(cfg, g_weights_dir);
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
    } else {
        printf("  ⚠  No backend initialized (weights missing or no hardware)\n");
        printf("     Server starts in discovery-only mode.\n");
    }

    // Phase 4: Benchmark available backends
    if (inited) {
        printf("\n── Benchmark ──\n");
        mgr.benchmark_all(3);  // 3 tokens is enough for a score

        // benchmark_all preserves the pre-existing (init'd) backend instance.
        // Now re-evaluate to pick the fastest backend per strategy.
        printf("\n── Select best backend ──\n");
        mgr.set_strategy(SelectionStrategy::FASTEST);

        auto* active = mgr.active_info();
        if (active) {
            printf("  Active: %s (%.1f ms/tok)\n",
                   active->id.c_str(), active->score);
        }
    }

    // ── HTTP Server ──
    httplib::Server svr;

    // ── CORS middleware ──
    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, X-Backend, X-Strategy, Authorization");
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    auto add_cors = [](httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
    };

    // ── GET /v1/health — Backend status dashboard ──
    svr.Get("/v1/health", [&](const httplib::Request& req, httplib::Response& res) {
        json j = health_json(mgr);
        j["version"] = "unified-server-1.0";
        j["model"] = "zaya";
        j["weights_dir"] = g_weights_dir;
        j["uptime"] = std::to_string(time(nullptr)) + "s";
        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    // ── GET /v1/models — List models ──
    svr.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
        auto* active = mgr.active_info();
        json j;
        j["object"] = "list";
        json models = json::array();
        models.push_back(model_info_json(active));
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
        mgr.set_strategy(strategy);

        // Check for strategy engine routing
        std::string strategy_name = resolve_strategy_name(req);
        bool use_strategy_engine = !strategy_name.empty();

        // Extract messages and build prompt + user message for content routing
        std::string prompt;
        std::string last_user_msg;
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
                        }
                    }
                }
                prompt += role + ": " + content + "\n";
                if (role == "user") last_user_msg = content;
            }
        } else if (body.contains("prompt")) {
            prompt = body["prompt"].get<std::string>();
            last_user_msg = prompt;
        }

        int max_tokens = body.value("max_tokens", 256);
        float temperature = body.value("temperature", 0.7f);

        // Tokenize with logprobs for cascade strategy
        std::vector<int> prompt_tokens;
        std::vector<double> prompt_logprobs;
        if (use_strategy_engine) {
            prompt_tokens = g_tokenizer.encode_with_logprobs(prompt, prompt_logprobs);
        } else {
            prompt_tokens = g_tokenizer.encode(prompt);
        }
        if (prompt_tokens.empty()) {
            prompt_tokens = {g_tokenizer.bos_id};
        }

        // Generate with strategy-aware routing
        StrategyEngine* se = use_strategy_engine ? &g_strategy_engine : nullptr;
        json gen_result = generate_completion(mgr, prompt_tokens, prompt_logprobs,
                                               max_tokens, backend_id,
                                               se, last_user_msg);

        // Build OpenAI-compatible response
        json response;
        response["id"] = "cmpl-" + std::to_string(time(nullptr));
        response["object"] = "chat.completion";
        response["created"] = time(nullptr);
        response["model"] = "zaya";

        json choice;
        choice["index"] = 0;
        json message;
        message["role"] = "assistant";
        message["content"] = gen_result.value("text", "");
        choice["message"] = message;
        choice["finish_reason"] = gen_result.contains("error") ? "error" : "stop";

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

        res.set_content(response.dump(2), "application/json");
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
        mgr.set_strategy(resolve_strategy(req));

        // Accept prompt or tokens
        std::vector<int> prompt_tokens;
        if (body.contains("tokens") && body["tokens"].is_array()) {
            for (auto& t : body["tokens"]) prompt_tokens.push_back(t.get<int>());
        } else if (body.contains("prompt")) {
            prompt_tokens = g_tokenizer.encode(body["prompt"].get<std::string>());
        }

        if (prompt_tokens.empty()) {
            prompt_tokens = {g_tokenizer.bos_id};
        }

        int max_tokens = body.value("n_predict", 256);
        float temperature = body.value("temperature", 0.7f);

        std::vector<double> empty_logprobs;
        json gen_result = generate_completion(mgr, prompt_tokens, empty_logprobs,
                                               max_tokens, backend_id);

        json response;
        response["tokens"] = gen_result["tokens"];
        response["text"] = gen_result["text"];
        response["gen_ms"] = gen_result["gen_ms"];
        response["tok_s"] = gen_result["tok_s"];
        response["backend_used"] = gen_result["backend_used"];

        res.set_header("X-Backend-Id", gen_result.value("backend_used", "unknown"));
        res.set_content(response.dump(2), "application/json");
        add_cors(res);
    });

    // ── GET /v1/router — Strategy engine status ──
    svr.Get("/v1/router", [&](const httplib::Request&, httplib::Response& res) {
        json j;
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
        auto perf_table = build_performance_table(mgr, "zaya");
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
        if (!backend_id.empty()) {
            ok = mgr.select_backend(backend_id);
        }

        json j;
        j["ok"] = ok;
        j["selected"] = backend_id;
        if (ok) {
            auto* active = mgr.active_info();
            j["active_backend"] = active ? active->id : "none";
            j["active_type"] = active ? backend_name(active->type) : "none";
        } else {
            j["error"] = "Backend '" + backend_id + "' not found or not functional";
        }
        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    // ── GET /v1/backend/status — Full backend report ──
    svr.Get("/v1/backend/status", [&](const httplib::Request& req, httplib::Response& res) {
        json j;
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
    svr.Get("/", [&](const httplib::Request& req, httplib::Response& res) {
        json j;
        j["service"] = "1bit.systems --- One binary, all backends, intelligent routing";
        j["version"] = "1.0";
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
    printf("\n  Press Ctrl+C to stop.\n");
    printf("──────────────────────────────────────────────\n\n");

    if (!svr.listen("0.0.0.0", g_port)) {
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
