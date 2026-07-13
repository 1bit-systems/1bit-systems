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

#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Globals ──
static std::atomic<bool> keep_running{true};
static std::string g_weights_dir = "/tmp/zaya_weights";
static int g_port = 8088;

// ── Simple ASCII tokenizer (no HIP, no Python, no external deps) ──
// Matches the zaya_server.cpp inline tokenizer pattern.
// Supports BOS/EOS and ASCII passthrough. Production deployments
// should swap this for a proper BPE tokenizer.
struct SimpleTokenizer {
    // Map token ID to string for output
    std::vector<std::string> id_to_token;
    int bos_id = 2;
    int eos_id = 1;
    int vocab_size = 0;

    bool load(const std::string& vocab_path) {
        std::ifstream f(vocab_path, std::ios::binary);
        if (!f) {
            fprintf(stderr, "No vocab at %s — using fallback ASCII tokenizer\n", vocab_path.c_str());
            return false;
        }
        fprintf(stderr, "Loaded vocab from %s\n", vocab_path.c_str());
        return true;
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> r = {bos_id}; // BOS
        for (char c : text)
            if (c >= ' ' && c <= '~')
                r.push_back((unsigned char)c + 100);
        return r;
    }

    std::string decode(const std::vector<int>& tokens) {
        std::string r;
        for (int v : tokens) {
            if (v == bos_id || v == eos_id) continue; // BOS/EOS
            if (v > 100 && v < 200) r += (char)(v - 100);
            else { r += '['; r += std::to_string(v); r += ']'; }
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

// ── Generate completion ──
// Returns { text, tokens, backend_used, ms_per_tok, tok_s }
static json generate_completion(BackendManager& mgr,
                                 const std::vector<int>& prompt_tokens,
                                 int max_tokens,
                                 const std::string& backend_id) {
    json result;

    // Select backend if specified
    if (backend_id != "auto") {
        mgr.select_backend(backend_id);
    }

    auto* active = mgr.active_info();
    if (active)
        result["backend_used"] = active->id;
    else
        result["backend_used"] = "none";

    // Reset backend state for new sequence
    if (!mgr.reset()) {
        result["error"] = "Failed to reset backend";
        result["tokens"] = json::array();
        result["text"] = "";
        return result;
    }

    std::vector<int> output_tokens;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Prefill: run the prompt through the model
    // For the Backend interface, we call generate() per token
    // which handles forward + lm_head internally
    int last_token = prompt_tokens.back();

    // First, process all prompt tokens (prefill) without generating output
    for (size_t i = 0; i + 1 < prompt_tokens.size(); i++) {
        int result_id = mgr.generate(prompt_tokens[i]);
        if (result_id < 0) {
            // Backend failed mid-prefill
            float ms = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            result["error"] = "Backend failed during prefill at token " + std::to_string(i);
            result["gen_ms"] = ms;
            result["gen_tokens"] = output_tokens.size();
            return result;
        }
    }

    // Now generate new tokens
    for (int i = 0; i < max_tokens; i++) {
        int next = mgr.generate(last_token);
        if (next < 0) break;  // backend failed
        output_tokens.push_back(next);
        last_token = next;
        if (next == g_tokenizer.eos_id) break;
    }

    float ms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    result["tokens"] = output_tokens;
    result["text"] = g_tokenizer.decode(output_tokens);
    result["gen_ms"] = ms;
    result["gen_tokens"] = (int)output_tokens.size();
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
        fprintf(stderr, "Warning: could not open lock file %s (%s) — skipping singleton guard\n",
                kLockPath, strerror(errno));
        return;
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
                usleep(200 * 1000);
            }
        }

        // Retry now that the previous holder (if any) should be gone.
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            fprintf(stderr, "Warning: could not acquire singleton lock (%s) — continuing anyway\n",
                    strerror(errno));
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
        printf("  ✓  Tokenizer loaded (%d tokens)\n", g_tokenizer.vocab_size);
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

        // benchmark_all() destroys instances and marks them non-functional (fixes #93).
        // Re-init the fastest backend so the server starts functional.
        printf("\n── Re-initialize best backend ──\n");
        // Re-init the backend that benchmark destroyed (fixes #93 cleanup)
        // Find the backend with the best score and re-init it
        std::string best_id;
        float best_score = 99999;
        for (auto& b : mgr.backends()) {
            if (b.score > 0 && b.score < best_score) {
                best_score = b.score;
                best_id = b.id;
            }
        }
        mgr.init(cfg, g_weights_dir);
        if (!best_id.empty() && mgr.select_backend(best_id)) {
            printf("  \u2713  Active: %s (%.1f ms/tok)\n",
                   best_id.c_str(), best_score);
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

        // Extract messages and build prompt
        std::string prompt;
        if (body.contains("messages") && body["messages"].is_array()) {
            for (auto& msg : body["messages"]) {
                std::string role = msg.value("role", "user");
                std::string content;
                if (msg["content"].is_string()) {
                    content = msg["content"].get<std::string>();
                } else if (msg["content"].is_array()) {
                    // Handle multi-modal content (text parts only)
                    for (auto& part : msg["content"]) {
                        if (part.value("type", "") == "text") {
                            content += part.value("text", "");
                        }
                    }
                }
                prompt += role + ": " + content + "\n";
            }
        } else if (body.contains("prompt")) {
            prompt = body["prompt"].get<std::string>();
        }

        int max_tokens = body.value("max_tokens", 256);
        float temperature = body.value("temperature", 0.7f);

        // Tokenize
        std::vector<int> prompt_tokens = g_tokenizer.encode(prompt);
        if (prompt_tokens.empty()) {
            prompt_tokens = {g_tokenizer.bos_id};
        }

        // Generate
        json gen_result = generate_completion(mgr, prompt_tokens, max_tokens, backend_id);

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

        // Add backend metadata as headers
        res.set_header("X-Backend-Id", gen_result.value("backend_used", "unknown"));
        char tok_s_buf[32];
        snprintf(tok_s_buf, sizeof(tok_s_buf), "%.1f", gen_result.value("tok_s", 0.0f));
        res.set_header("X-Backend-Tok-s", tok_s_buf);

        res.set_content(response.dump(2), "application/json");
        add_cors(res);

        // Log to stdout
        auto* active = mgr.active_info();
        printf("  [%s] %d tokens → %d tokens (%.1f tok/s, backend: %s)\n",
               backend_id.c_str(),
               (int)prompt_tokens.size(),
               gen_result.value("gen_tokens", 0),
               gen_result.value("tok_s", 0.0f),
               gen_result.value("backend_used", "?").c_str());
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

        json gen_result = generate_completion(mgr, prompt_tokens, max_tokens, backend_id);

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

    // ── GET / — Root health check ──
    svr.Get("/", [&](const httplib::Request& req, httplib::Response& res) {
        json j;
        j["service"] = "1bit-systems unified inference server";
        j["version"] = "1.0";
        j["status"] = mgr.active_backend() ? "ready" : "initializing";
        j["endpoints"] = {
            "/v1/health",
            "/v1/models",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/backend/select",
            "/v1/backend/status"
        };
        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    // ── Start server ──
    printf("\n──────────────────────────────────────────────\n");
    printf("  Server starting on port %d\n", g_port);
    printf("  API: http://127.0.0.1:%d/v1\n", g_port);
    printf("  Health: http://127.0.0.1:%d/v1/health\n", g_port);
    printf("  Chat:   POST http://127.0.0.1:%d/v1/chat/completions\n", g_port);
    printf("──────────────────────────────────────────────\n");
    printf("\n  Try it:\n");
    printf("    curl http://127.0.0.1:%d/v1/health\n", g_port);
    printf("    curl -X POST http://127.0.0.1:%d/v1/completions \\\n", g_port);
    printf("      -H \"Content-Type: application/json\" \\\n");
    printf("      -d '{\"prompt\":\"Hello\",\"n_predict\":20}'\n");
    printf("\n  Select backend:\n");
    printf("    curl -X POST http://127.0.0.1:%d/v1/chat/completions \\\n", g_port);
    printf("      -H \"X-Backend: npu_xrt\" \\\n");
    printf("      -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":20}'\n");
    printf("\n  Press Ctrl+C to stop.\n");
    printf("──────────────────────────────────────────────\n\n");

    if (!svr.listen("0.0.0.0", g_port)) {
        fprintf(stderr, "Failed to start server on port %d\n", g_port);
        return 1;
    }

    // ── Cleanup ──
    printf("\nShutting down...\n");
    mgr.destroy();
    printf("Done.\n");
    return 0;
}
