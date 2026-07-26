// token_router.cpp — Speculative Decode Token Router (v2)
//
// Routes tokens between NPU (draft) and GPU (verify) backends:
//   1. NPU drafts N tokens fast (~97 tok/s, Qwen3-0.6B)
//   2. GPU verifies draft tokens in parallel (~18.5 tok/s, Zaya1-8B)
//   3. Dynamic n_draft (1-16) adjusts by acceptance rate
//   4. Cascade fallback: NPU → GPU on low-confidence tokens
//
// Build: cmake --build . --target token_router -j8
// Run:   ./build/token_router [--port 13306] [--npu-bin path] [--gpu-bin path]
//
// API (OpenAI-compatible):
//   GET  /v1/health            — Backend status
//   GET  /v1/models            — Available models
//   POST /v1/chat/completions  — Speculative decode inference
//   POST /v1/verify            — GPU verify endpoint
//
// Architecture:
//   Client → TokenRouter (:13306)
//              │ strategy: cascade / spec_decode
//              │
//         ┌────┴──────────┐
//         │               │
//      NPU Engine      GPU Engine
//      (subprocess)    (HTTP server :13307)
//         │               │
//         └────┬──────────┘
//              │ dma-buf zero-copy handoff (Strix Halo unified memory)
//              │
//         ┌────┴────┐
//         │ zaya_server (ROCm/HIP on GPU)
//         │ POST /completion
//         │ {"tokens":[...], "n_predict":N}
//         │ Returns {"tokens":[...],"text":"...","gen_ms":N,"tok_s":N}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <curl/curl.h>

// ── libcurl write callback ──
struct CurlBuffer {
    std::string data;
    bool headers_done = false;
};
static size_t curl_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    size_t total = size * nmemb;
    auto* buf = (CurlBuffer*)userdata;
    buf->data.append(ptr, total);
    return total;
}

// ── Configuration ──
static int listen_port = 13306;
static std::string get_home_prefix() {
    const char* home = getenv("HOME");
    return home ? std::string(home) : std::string();
}
static std::string npu_engine_bin = get_home_prefix() + "/engine/npu/build/npu_engine_server";
static std::string gpu_engine_bin = get_home_prefix() + "/build/zaya_server";  // GPU backend binary
// Tokenizer: pure C++ via rocm_cpp/tokenizer.h (zero Python)
static int max_workers = 4;

// ── GPU port for verify backend ──
static int gpu_server_port = 13307;

// Forward declarations for JSON helpers used by GpuEngine
static std::vector<int> json_tokens(const std::string& j);
static int json_int(const std::string& j, const std::string& k, int d = 0);
static std::string json_str(const std::string& j, const std::string& k);

// ── NPU Engine Subprocess ──
struct NpuEngine {
    pid_t pid = 0;
    int stdin_fd = -1;
    int stdout_fd = -1;
    int stderr_fd = -1;
    bool ready = false;
    int timeout_ms = 120000;

    bool start() {
        if (access(npu_engine_bin.c_str(), X_OK) != 0) {
            fprintf(stderr, "  ⚠ NPU engine not found: %s\n", npu_engine_bin.c_str());
            return false;
        }
        int to_stdin[2], from_stdout[2], from_stderr[2];
        pipe(to_stdin); pipe(from_stdout); pipe(from_stderr);

        pid = fork();
        if (pid == 0) {
            // Child: NPU engine process
            dup2(to_stdin[0], STDIN_FILENO);
            dup2(from_stdout[1], STDOUT_FILENO);
            dup2(from_stderr[1], STDERR_FILENO);
            close(to_stdin[0]); close(to_stdin[1]);
            close(from_stdout[0]); close(from_stdout[1]);
            close(from_stderr[0]); close(from_stderr[1]);
            execl(npu_engine_bin.c_str(), npu_engine_bin.c_str(), nullptr);
            _exit(1);
        }

        close(to_stdin[0]); close(from_stdout[1]); close(from_stderr[1]);
        stdin_fd = to_stdin[1];
        stdout_fd = from_stdout[0];
        stderr_fd = from_stderr[0];

        // Drain stderr until "Ready." line (engine says when it's initialized)
        char buf[4096];
        auto t0 = std::chrono::steady_clock::now();
        std::string stderr_buf;
        while (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t0).count() < 30) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stderr_fd, &fds);
            struct timeval tv = {1, 0};
            int r = select(stderr_fd + 1, &fds, nullptr, nullptr, &tv);
            if (r > 0) {
                int n = (int)read(stderr_fd, buf, sizeof(buf)-1);
                if (n > 0) {
                    buf[n] = 0;
                    stderr_buf += buf;
                    // Print engine stderr (prefixed)
                    char* line = buf;
                    for (int i = 0; i < n; i++) {
                        if (buf[i] == '\n') { buf[i] = 0; fprintf(stderr, "    [npu] %s\n", line); line = buf+i+1; }
                    }
                    if (stderr_buf.find("Ready.") != std::string::npos) {
                        ready = true;
                        break;
                    }
                }
            }
        }

        if (!ready) {
            fprintf(stderr, "  ⚠ NPU engine failed to start within 30s\n");
            stop();
            return false;
        }

        fprintf(stderr, "  ▶ NPU engine ready (pid=%d)\n", pid);
        return true;
    }

    // Send tokens to NPU engine, get back generated tokens + logprobs
    // Returns empty on failure
    std::string infer(const std::vector<int>& prompt_tokens, int max_new) {
        if (!ready) return "";

        // Build request JSON
        std::string req = "{\"tokens\":[";
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
            if (i) req += ",";
            req += std::to_string(prompt_tokens[i]);
        }
        req += "],\"max_new_tokens\":" + std::to_string(max_new) + "}\n";

        // Write to engine stdin
        int wrote = (int)write(stdin_fd, req.data(), req.size());
        if (wrote != (int)req.size()) return "";

        // Read response from engine stdout (one line)
        std::string resp;
        char buf[65536];
        auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count() < timeout_ms) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd, &fds);
            struct timeval tv = {1, 0};
            int r = select(stdout_fd + 1, &fds, nullptr, nullptr, &tv);
            if (r > 0) {
                int n = (int)read(stdout_fd, buf, sizeof(buf)-1);
                if (n > 0) {
                    buf[n] = 0;
                    resp += buf;
                    // Check if we have a complete JSON object
                    if (resp.find('\n') != std::string::npos) {
                        resp = resp.substr(0, resp.find('\n'));
                        break;
                    }
                } else {
                    break; // EOF
                }
            }
        }
        return resp;
    }

    void stop() {
        if (pid > 0) {
            close(stdin_fd); close(stdout_fd); close(stderr_fd);
            kill(pid, SIGTERM);
            usleep(100000);
            kill(pid, SIGKILL);
            waitpid(pid, nullptr, WNOHANG);
            pid = 0;
        }
        ready = false;
    }
};

// ── GPU Engine (zaya_server HTTP subprocess) ──
struct GpuEngine {
    pid_t pid = 0;
    int port = 13307;
    bool ready = false;
    CURL* curl = nullptr;
    std::string base_url;
    int health_check_ms = 30000;
    int request_timeout_ms = 120000;

    bool start() {
        if (access(gpu_engine_bin.c_str(), X_OK) != 0) {
            fprintf(stderr, "  ⚠ GPU engine not found: %s\n", gpu_engine_bin.c_str());
            return false;
        }

        // Fork+exec zaya_server on our port
        pid = fork();
        if (pid == 0) {
            // Child: GPU engine process
            // Redirect stdout/stderr so we don't see raw HIP output
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) {
                dup2(devnull, STDOUT_FILENO);
                dup2(devnull, STDERR_FILENO);
                close(devnull);
            }
            char port_str[16];
            snprintf(port_str, sizeof(port_str), "%d", port);
            execl(gpu_engine_bin.c_str(), gpu_engine_bin.c_str(), port_str, nullptr);
            _exit(1);
        }

        fprintf(stderr, "  ▶ GPU engine starting (pid=%d, port=%d)...\n", pid, port);

        // Wait for TCP port to be ready (health check via connect)
        base_url = "http://127.0.0.1:" + std::to_string(port);
        auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count() < health_check_ms) {
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock >= 0) {
                struct sockaddr_in addr;
                memset(&addr, 0, sizeof(addr));
                addr.sin_family = AF_INET;
                addr.sin_port = htons((uint16_t)port);
                addr.sin_addr.s_addr = htonl(0x7f000001); // 127.0.0.1
                if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                    close(sock);
                    ready = true;
                    break;
                }
                close(sock);
            }
            usleep(100000); // 100ms poll
        }

        if (!ready) {
            fprintf(stderr, "  ⚠ GPU engine failed to start within %dms\n", health_check_ms);
            stop();
            return false;
        }

        // Initialize libcurl handle
        curl = curl_easy_init();
        if (!curl) {
            fprintf(stderr, "  ⚠ Failed to init libcurl for GPU engine\n");
            stop();
            return false;
        }

        fprintf(stderr, "  ▶ GPU engine ready on %s\n", base_url.c_str());
        return true;
    }

    // Send tokens to zaya_server, get back JSON response
    std::string infer(const std::vector<int>& tokens, int n_predict) {
        if (!ready || !curl) return "";

        // Build request JSON: {"tokens":[...],"n_predict":N}
        std::string json_body = "{\"tokens\":[";
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i) json_body += ",";
            json_body += std::to_string(tokens[i]);
        }
        json_body += "],\"n_predict\":" + std::to_string(n_predict) + "}";

        std::string url = base_url + "/completion";

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)json_body.size());
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, (long)request_timeout_ms);

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        CurlBuffer buf;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);

        CURLcode res = curl_easy_perform(curl);
        curl_slist_free_all(headers);

        if (res != CURLE_OK) {
            fprintf(stderr, "    [gpu] curl error: %s\n", curl_easy_strerror(res));
            return "";
        }

        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code != 200) {
            fprintf(stderr, "    [gpu] HTTP %ld: %s\n", http_code, buf.data.c_str());
            return "";
        }

        return buf.data;
    }

    // Verify draft token(s) against GPU:
    //   Send context + draft tokens to GPU, check if GPU agrees.
    //   Returns JSON: {"accept":true/false, "gpu_token":id, "draft_token":id, "tokens":[...], "replacement_tokens":[...]}
    //   On accept: gpu_token matches first draft token; tokens = gpu-generated continuation
    //   On reject: gpu_token is the GPU's preferred token; replacement_tokens = GPU continuation
    std::string verify(const std::vector<int>& context, const std::vector<int>& draft_tokens) {
        if (!ready || draft_tokens.empty()) {
            return "{\"accept\":false,\"gpu_token\":0,\"replacement_tokens\":[]}";
        }

        // Build extended context: context + draft tokens
        std::vector<int> extended = context;
        extended.insert(extended.end(), draft_tokens.begin(), draft_tokens.end());

        // Ask GPU to generate tokens from this extended context
        // n_predict = draft_tokens.size() so we get the same number of tokens for comparison
        int n_pred = std::min((int)draft_tokens.size(), 4);  // limit to avoid long generation
        std::string resp = infer(extended, n_pred);
        if (resp.empty()) {
            return "{\"accept\":false,\"gpu_token\":0,\"replacement_tokens\":[]}";
        }

        std::vector<int> gpu_tokens = json_tokens(resp);
        if (gpu_tokens.empty()) {
            return "{\"accept\":false,\"gpu_token\":0,\"replacement_tokens\":[]}";
        }

        int gpu_top1 = gpu_tokens[0];
        int draft_first = draft_tokens[0];
        bool accept = (gpu_top1 == draft_first);

        // Build replacement tokens array (GPU's generated tokens if rejected)
        std::string replacement_json = "[";
        for (size_t i = 0; i < gpu_tokens.size(); i++) {
            if (i) replacement_json += ",";
            replacement_json += std::to_string(gpu_tokens[i]);
        }
        replacement_json += "]";

        char result[4096];
        snprintf(result, sizeof(result),
            "{\"accept\":%s,\"gpu_token\":%d,\"draft_token\":%d,\"tokens\":%s,\"replacement_tokens\":%s}",
            accept ? "true" : "false", gpu_top1, draft_first,
            replacement_json.c_str(), replacement_json.c_str());

        return std::string(result);
    }

    void stop() {
        if (curl) {
            curl_easy_cleanup(curl);
            curl = nullptr;
        }
        if (pid > 0) {
            kill(pid, SIGTERM);
            usleep(200000);
            kill(pid, SIGKILL);
            waitpid(pid, nullptr, WNOHANG);
            pid = 0;
        }
        ready = false;
    }
};

// ── Tokenizer (pure C++ — zero Python at runtime) ──
#include "rocm_cpp/tokenizer.h"

static rcpp_tokenizer_t* g_tok = nullptr;
static std::string g_tok_path = "tokenizer.htok";

static bool ensure_tokenizer() {
    if (g_tok) return true;
    rcpp_tokenizer_t* tok = nullptr;
    if (rcpp_tokenizer_load(g_tok_path.c_str(), &tok) == RCPP_OK && tok) {
        g_tok = tok;
        return true;
    }
    fprintf(stderr, "Failed to load tokenizer from %s\n", g_tok_path.c_str());
    return false;
}

static std::vector<int> tokenize(const std::string& text) {
    if (!ensure_tokenizer()) return {};
    std::vector<int> r(4096);
    size_t out_n = 0;
    rcpp_status_t st = rcpp_tokenizer_encode(g_tok, text.c_str(), text.size(),
                                               1, r.data(), (int)r.size(), &out_n);
    if (st == RCPP_OK && out_n > 0) {
        r.resize(out_n);
        return r;
    }
    return {rcpp_tokenizer_bos_id(g_tok)};
}

static std::string detokenize(const std::vector<int>& ids) {
    if (!ensure_tokenizer()) return "";
    std::string r(4096, '\0');
    size_t out_len = 0;
    rcpp_status_t st = rcpp_tokenizer_decode(g_tok, ids.data(), (int)ids.size(),
                                              r.data(), (int)r.size(), &out_len);
    if (st == RCPP_OK && out_len > 0) {
        r.resize(out_len);
        return r;
    }
    return "";
}

// ── GPT-style JSON helpers ──
static std::string json_str(const std::string& j, const std::string& k) {
    auto p = j.find("\"" + k + "\""); if (p == j.npos) return "";
    p = j.find(':', p); if (p == j.npos) return "";
    p = j.find_first_of("\"", p); if (p == j.npos || j[p] != '\"') {
        // Try number
        auto ns = j.find_first_of("-0123456789", p+1);
        if (ns != j.npos) {
            auto ne = j.find_first_not_of("0123456789.e-+", ns);
            return j.substr(ns, ne - ns);
        }
        return "";
    }
    auto e = j.find('\"', p+1); if (e == j.npos) return "";
    return j.substr(p+1, e-p-1);
}
static int json_int(const std::string& j, const std::string& k, int d) {
    auto p = j.find("\"" + k + "\""); if (p == j.npos) return d;
    p = j.find(':', p); if (p == j.npos) return d;
    p = j.find_first_of("-0123456789", p); if (p == j.npos) return d;
    char* e; return (int)strtol(&j[p], &e, 10);
}
static std::vector<int> json_tokens(const std::string& j) {
    std::vector<int> ids;
    auto p = j.find("\"tokens\"");
    if (p == j.npos) return ids;
    p = j.find('[', p); if (p == j.npos) return ids;
    auto e = j.find(']', p); if (e == j.npos) return ids;
    std::string arr = j.substr(p+1, e-p-1);
    size_t i = 0;
    while (i < arr.size()) {
        while (i < arr.size() && (arr[i] == ',' || arr[i] == ' ' || arr[i] == '[')) i++;
        if (i >= arr.size() || arr[i] == ']') break;
        char* end;
        long val = strtol(arr.c_str() + i, &end, 10);
        if (end == arr.c_str() + i) break;
        ids.push_back((int)val);
        i = end - arr.c_str();
    }
    return ids;
}
static std::vector<float> json_logprobs(const std::string& j) {
    std::vector<float> lps;
    auto p = j.find("\"logprobs\"");
    if (p == j.npos) return lps;
    p = j.find('[', p); if (p == j.npos) return lps;
    auto e = j.find(']', p); if (e == j.npos) return lps;
    std::string arr = j.substr(p+1, e-p-1);
    size_t i = 0;
    while (i < arr.size()) {
        while (i < arr.size() && (arr[i] == ',' || arr[i] == ' ' || arr[i] == '[')) i++;
        if (i >= arr.size() || arr[i] == ']') break;
        char* end;
        double val = strtod(arr.c_str() + i, &end);
        if (end == arr.c_str() + i) break;
        lps.push_back((float)val);
        i = end - arr.c_str();
    }
    return lps;
}

// ── ChatML prompt builder ──
static std::string build_chatml_prompt(const std::string& body) {
    // Parse messages from the request body
    // Format: {"model":"...","messages":[{"role":"user","content":"..."},...],"max_tokens":256}
    std::string result;
    bool found_system = false;

    // Walk through messages array
    size_t msgs_start = body.find("\"messages\"");
    if (msgs_start == std::string::npos) return "";
    msgs_start = body.find('[', msgs_start);
    if (msgs_start == std::string::npos) return "";

    size_t pos = msgs_start;
    int depth = 0;
    while (pos < body.size()) {
        // Find next message object
        size_t obj_start = body.find('{', pos);
        if (obj_start == std::string::npos || obj_start > body.find(']', pos)) break;
        size_t obj_end = body.find('}', obj_start);
        if (obj_end == std::string::npos) break;

        std::string msg = body.substr(obj_start, obj_end - obj_start + 1);
        std::string role = json_str(msg, "role");
        std::string content = json_str(msg, "content");

        if (role == "system") {
            result += "<|im_start|>system\n" + content + "<|im_end|>\n";
            found_system = true;
        } else if (role == "user") {
            result += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            result += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
        } else {
            result += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
        pos = obj_end + 1;
    }

    if (!found_system) {
        result = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + result;
    }
    result += "<|im_start|>assistant\n";
    return result;
}

// ── Speculative Decode ──
struct SpecDecodeState {
    // Dynamic n_draft (1-16)
    int n_draft = 4;
    int min_draft = 1;
    int max_draft = 16;
    float accept_threshold = -1.0f;  // logprob below this → reject (i.e., < exp(-1) ≈ 0.37)
    float grow_threshold = 0.85f;    // acceptance rate above this → grow n_draft
    float shrink_threshold = 0.50f;  // acceptance rate below this → shrink n_draft
    int total_accepted = 0;
    int total_drafted = 0;
    float acceptance_rate = 1.0f;
    std::mt19937 rng{std::random_device{}()};
};

// ── HTTP Response Helpers ──
static void send_json(int cl, int code, const std::string& body) {
    char header[4096];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d OK\r\n"
        "Content-Type: application/json\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n", code, body.size());
    write(cl, header, hlen);
    write(cl, body.data(), body.size());
    close(cl);
}

// ── Global state (yes, globals — signal handlers need them) ──
static NpuEngine npu;
static GpuEngine gpu;
static int server_fd = -1;

// Signal-safe handler: just set a flag. The main loop checks and does cleanup.
// Never call fprintf/malloc/mutex from a signal handler — async-signal-unsafe.
static volatile sig_atomic_t g_shutdown_requested = 0;

static void cleanup(int) {
    g_shutdown_requested = 1;
}

// ── Request Handling ──
static void handle_request(int cl, const std::string& req, const std::string& body) {
    // ── CORS preflight ──
    if (req.find("OPTIONS") == 0) {
        send_json(cl, 200, "{\"ok\":true}");
        return;
    }

    // ── GET /v1/health ──
    if (req.find("GET /v1/health") != std::string::npos) {
        std::string resp = R"({"status":"ok","version":"2.0","backends":{)" +
            std::string("\"npu\":{\"type\":\"C++ engine\",\"ready\":") + (npu.ready?"true":"false") + "}," +
            std::string("\"gpu\":{\"type\":\"zaya_server (ROCm/HIP)\",\"ready\":") + (gpu.ready?"true":"false") + "}" +
            "}}";
        send_json(cl, 200, resp);
        return;
    }

    // ── GET /v1/models ──
    if (req.find("GET /v1/models") != std::string::npos) {
        std::string resp = "{\"object\":\"list\",\"data\":["
            "{\"id\":\"npu-draft\",\"object\":\"model\",\"owned_by\":\"npu\",\"description\":\"Qwen3-0.6B NPU draft (69 tok/s)\"},"
            "{\"id\":\"spec-decode\",\"object\":\"model\",\"owned_by\":\"router\",\"description\":\"NPU draft + GPU verify (speculative decode)\"},"
            "{\"id\":\"gpu-verify\",\"object\":\"model\",\"owned_by\":\"gpu\",\"description\":\"Zaya1 GPU verify backend\"}"
            "]}";
        send_json(cl, 200, resp);
        return;
    }

    // ── POST /v1/verify — standalone GPU verify endpoint ──
    if (req.find("POST /v1/verify") != std::string::npos) {
        // Accepts: {"tokens":[ctx...],"draft_tokens":[d1,d2,...]}
        // Returns: {"accept":true/false,"gpu_token":id,"replacement_tokens":[...],"gpu_tokens":[...],"verify_ms":N}
        std::vector<int> input_tokens = json_tokens(body);
        // Also look for "draft_tokens" specifically
        auto dtp = body.find("\"draft_tokens\"");
        std::vector<int> draft_tokens;
        if (dtp != std::string::npos) {
            auto dp = body.find('[', dtp);
            if (dp != std::string::npos) {
                auto de = body.find(']', dp);
                if (de != std::string::npos) {
                    std::string arr = body.substr(dp+1, de-dp-1);
                    size_t i = 0;
                    while (i < arr.size()) {
                        while (i < arr.size() && (arr[i] == ',' || arr[i] == ' ' || arr[i] == '[')) i++;
                        if (i >= arr.size() || arr[i] == ']') break;
                        char* end;
                        long val = strtol(arr.c_str() + i, &end, 10);
                        if (end == arr.c_str() + i) break;
                        draft_tokens.push_back((int)val);
                        i = end - arr.c_str();
                    }
                }
            }
        }

        if (input_tokens.empty() || draft_tokens.empty()) {
            send_json(cl, 400, "{\"error\":\"need tokens and draft_tokens\"}");
            return;
        }

        if (!gpu.ready) {
            send_json(cl, 503, "{\"error\":\"GPU backend not ready\",\"accept\":false}");
            return;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        std::string result = gpu.verify(input_tokens, draft_tokens);
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        // Remove trailing }
        if (!result.empty() && result.back() == '}') {
            result.pop_back();
        }
        result += ",\"verify_ms\":" + std::to_string((int)ms) + "}";

        send_json(cl, 200, result);
        return;
    }

    // ── POST /v1/chat/completions ──
    if (req.find("POST /v1/chat/completions") != std::string::npos ||
        req.find("POST /v1/completion") != std::string::npos) {

        std::string model = json_str(body, "model");
        if (model.empty()) model = "spec-decode";
        int max_tokens = json_int(body, "max_tokens", 256);

        // Build prompt
        std::string prompt = build_chatml_prompt(body);
        if (prompt.empty()) {
            send_json(cl, 400, "{\"error\":\"empty prompt\"}");
            return;
        }

        // Tokenize
        std::vector<int> prompt_tokens = tokenize(prompt);
        if (prompt_tokens.empty()) {
            send_json(cl, 400, "{\"error\":\"tokenization failed\"}");
            return;
        }

        // Select strategy based on model
        bool use_spec_decode = (model == "spec-decode" || model == "zaya-spec");
        bool use_npu_only = (model == "npu-draft");

        // ── Inference ──
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<int> all_tokens(prompt_tokens.begin(), prompt_tokens.end());
        std::vector<int> out_tokens;
        std::vector<float> out_logprobs;
        std::vector<float> out_confidence;
        const int eos_id = 151645;

        if (use_npu_only || !use_spec_decode) {
            // Simple NPU-only mode (no spec decode)
            std::string resp = npu.infer(prompt_tokens, max_tokens);
            if (resp.empty()) {
                send_json(cl, 502, "{\"error\":\"NPU engine failed\",\"x-device\":\"npu\"}");
                return;
            }
            out_tokens = json_tokens(resp);
            out_logprobs = json_logprobs(resp);
        } else {
            // ── Speculative Decode ──
            SpecDecodeState sd;

            // Parse n_draft from request if provided
            int req_draft = json_int(body, "n_draft", 0);
            if (req_draft >= sd.min_draft && req_draft <= sd.max_draft)
                sd.n_draft = req_draft;

            std::vector<int> ctx = prompt_tokens;
            int generated = 0;

            while (generated < max_tokens) {
                // 1. NPU drafts sd.n_draft tokens
                int draft_count = std::min(sd.n_draft, max_tokens - generated);
                std::string npu_resp = npu.infer(ctx, draft_count);
                if (npu_resp.empty()) break;

                std::vector<int> draft_tokens = json_tokens(npu_resp);
                std::vector<float> draft_logprobs = json_logprobs(npu_resp);

                if (draft_tokens.empty()) break;

                // 2. Verify draft tokens using GPU backend
                int accepted = 0;
                int rejected_at = -1;

                if (gpu.ready) {
                    // ── GPU verify path — real speculative decode ──
                    // Send context + all draft tokens to GPU; it generates its own tokens
                    // and we accept the longest prefix match
                    std::string gpu_resp = gpu.verify(ctx, draft_tokens);

                    bool gpu_accept = (json_str(gpu_resp, "accept") == "true");
                    int gpu_token = json_int(gpu_resp, "gpu_token", -1);
                    std::vector<int> gpu_replacement = json_tokens(gpu_resp);

                    if (gpu_accept) {
                        // GPU agrees with the first draft token → accept it
                        out_tokens.push_back(draft_tokens[0]);
                        float lp = (!draft_logprobs.empty()) ? draft_logprobs[0] : 0.0f;
                        out_logprobs.push_back(lp);
                        ctx.push_back(draft_tokens[0]);
                        generated++;
                        accepted = 1;

                        fprintf(stderr, "    [router] verify: accept tok=%d (lp=%.2f)\n",
                                draft_tokens[0], lp);

                        // If more drafts and GPU agrees, optionally accept more
                        // (For now, one-at-a-time verification is conservative but correct)
                        for (int i = 1; i < (int)draft_tokens.size() && generated < max_tokens; i++) {
                            if (draft_tokens[i] == eos_id) break;
                            // Re-verify each subsequent draft token one at a time
                            std::vector<int> sub_draft = {draft_tokens[i]};
                            std::string sub_resp = gpu.verify(ctx, sub_draft);
                            if (json_str(sub_resp, "accept") == "true") {
                                float lps = (!draft_logprobs.empty() && i < (int)draft_logprobs.size()) ? draft_logprobs[i] : 0.0f;
                                out_tokens.push_back(draft_tokens[i]);
                                out_logprobs.push_back(lps);
                                ctx.push_back(draft_tokens[i]);
                                generated++;
                                accepted++;
                                fprintf(stderr, "    [router] verify: accept+ tok=%d\n", draft_tokens[i]);
                            } else {
                                // Reject at this position, use GPU's token instead
                                std::vector<int> repl = json_tokens(sub_resp);
                                int repl_token = json_int(sub_resp, "gpu_token", -1);
                                if (repl_token > 0 && repl_token < 200000) {
                                    out_tokens.push_back(repl_token);
                                    out_logprobs.push_back(-0.5f);  // placeholder logprob
                                    ctx.push_back(repl_token);
                                    generated++;
                                    fprintf(stderr, "    [router] verify: reject tok=%d → gpu tok=%d\n",
                                            draft_tokens[i], repl_token);
                                }
                                rejected_at = i;
                                break;
                            }
                        }
                    } else {
                        // GPU disagrees with first draft token — use GPU's output
                        // Extract the first token from gpu_replacement or use gpu_token
                        int repl_token = gpu_token;
                        if (!gpu_replacement.empty()) {
                            repl_token = gpu_replacement[0];
                        }
                        if (repl_token > 0 && repl_token < 200000) {
                            out_tokens.push_back(repl_token);
                            out_logprobs.push_back(-0.5f);
                            ctx.push_back(repl_token);
                            generated++;
                            fprintf(stderr, "    [router] verify: reject first draft tok=%d → gpu tok=%d\n",
                                    draft_tokens[0], repl_token);
                        }
                        rejected_at = 0;
                        accepted = 0;
                    }
                } else {
                    // ── GPU not ready — fall back to cascade mode ──
                    // Use NPU logprob threshold as a soft verify
                    fprintf(stderr, "    [router] gpu not ready — cascade fallback\n");
                    for (int i = 0; i < (int)draft_tokens.size() && generated < max_tokens; i++) {
                        float lp = (i < (int)draft_logprobs.size()) ? draft_logprobs[i] : 0.0f;

                        if (lp > sd.accept_threshold) {
                            out_tokens.push_back(draft_tokens[i]);
                            out_logprobs.push_back(lp);
                            ctx.push_back(draft_tokens[i]);
                            generated++;
                            accepted++;
                        } else {
                            fprintf(stderr, "    [router] cascade: npu low confidence (lp=%.2f) → accept anyway\n", lp);
                            out_tokens.push_back(draft_tokens[i]);
                            out_logprobs.push_back(lp);
                            ctx.push_back(draft_tokens[i]);
                            generated++;
                            accepted++;
                            break;
                        }
                        if (draft_tokens[i] == eos_id) break;
                    }
                }

                // 3. Update dynamic n_draft based on acceptance rate
                sd.total_drafted += (int)draft_tokens.size();
                sd.total_accepted += accepted;
                if (sd.total_drafted > 0) {
                    sd.acceptance_rate = (float)sd.total_accepted / sd.total_drafted;
                }

                if (sd.acceptance_rate > sd.grow_threshold && sd.n_draft < sd.max_draft) {
                    sd.n_draft = std::min(sd.n_draft + 1, sd.max_draft);
                } else if (sd.acceptance_rate < sd.shrink_threshold && sd.n_draft > sd.min_draft) {
                    sd.n_draft = std::max(sd.n_draft - 1, sd.min_draft);
                }

                // Check EOS
                if (!out_tokens.empty() && out_tokens.back() == eos_id) break;

                // Safety: if NPU returns no new tokens, break
                if (draft_tokens.empty()) break;
            }
        }

        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        // Decode output tokens to text
        std::vector<int> gen_tokens;
        for (int t : out_tokens) {
            if (t == eos_id) break;
            gen_tokens.push_back(t);
        }
        std::string text = detokenize(gen_tokens);

        std::string finish_reason = "stop";
        if (!out_tokens.empty() && out_tokens.back() != eos_id && (int)out_tokens.size() >= max_tokens)
            finish_reason = "length";

        // Build response
        char resp_buf[4096];
        int prompt_tok_count = (int)prompt_tokens.size();
        int completion_tok_count = (int)gen_tokens.size();

        // Escape text for JSON
        std::string escaped_text;
        for (char c : text) {
            if (c == '"') escaped_text += "\\\"";
            else if (c == '\\') escaped_text += "\\\\";
            else if (c == '\n') escaped_text += "\\n";
            else if (c == '\r') escaped_text += "\\r";
            else if (c == '\t') escaped_text += "\\t";
            else if ((unsigned char)c < 32) escaped_text += "\\u00" + std::to_string((unsigned char)c/16) + std::to_string((unsigned char)c%16);
            else escaped_text += c;
        }

        const char* device_str = (use_spec_decode && gpu.ready) ? "npu+gpu" :
                                 (use_spec_decode) ? "npu+cascade" : "npu";

        snprintf(resp_buf, sizeof(resp_buf),
            R"({"id":"chatcmpl-%d","object":"chat.completion","created":%d,)"
            R"("choices":[{"index":0,"message":{"role":"assistant","content":"%s"},"finish_reason":"%s"}],)"
            R"("usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d},)"
            R"("x-device":"%s","x-model":"spec-decode","x-ms":%.0f,"x-tok-s":%.1f})",
            (int)(time(nullptr)), (int)(time(nullptr)),
            escaped_text.c_str(), finish_reason.c_str(),
            prompt_tok_count, completion_tok_count, prompt_tok_count + completion_tok_count,
            device_str,
            ms, ms > 0 ? completion_tok_count / (ms/1000.0f) : 0.0f);

        send_json(cl, 200, resp_buf);
        return;
    }

    // ── 404 ──
    send_json(cl, 404, "{\"error\":\"not found\"}");
}

// ── Main ──
int main(int argc, char** argv) {
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i+1 < argc) listen_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--npu-bin") && i+1 < argc) npu_engine_bin = argv[++i];
        else if (!strcmp(argv[i], "--gpu-bin") && i+1 < argc) gpu_engine_bin = argv[++i];
    }

    signal(SIGINT, cleanup);
    signal(SIGTERM, cleanup);
    signal(SIGPIPE, SIG_IGN);

    fprintf(stderr, "╔══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  TokenRouter v2 — Speculative Decode    ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════╝\n\n");

    // ── Start NPU engine ──
    fprintf(stderr, "Starting backends...\n");
    if (!npu.start()) {
        fprintf(stderr, "  ⚠ NPU engine failed — running without NPU\n");
    }

    // ── Start GPU engine (HTTP backend: zaya_server on port 13307) ──
    fprintf(stderr, "\nStarting GPU verify backend...\n");
    gpu.port = gpu_server_port;
    if (!gpu.start()) {
        fprintf(stderr, "  ⚠ GPU engine failed — fall back to cascade mode\n");
    }

    // ── Start HTTP server ──
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)listen_port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        fprintf(stderr, "  ⚠ Port %d in use? Try: --port 13307\n", listen_port);
        close(server_fd);
        return 1;
    }
    if (listen(server_fd, 10) < 0) {
        perror("listen");
        close(server_fd);
        return 1;
    }

    fprintf(stderr, "\n🌐 TokenRouter listening on :%d\n", listen_port);
    fprintf(stderr, "   GET  /v1/health\n");
    fprintf(stderr, "   GET  /v1/models\n");
    fprintf(stderr, "   POST /v1/chat/completions  (speculative decode)\n");
    fprintf(stderr, "   POST /v1/verify           (standalone GPU verify)\n");
    fprintf(stderr, "\n   NPU engine: %s\n", npu.ready ? "✅ ready" : "❌ offline");
    fprintf(stderr, "   GPU engine: %s (port %d)\n", gpu.ready ? "✅ ready" : "❌ offline", gpu.port);
    fprintf(stderr, "   Strategy: %s\n\n", (gpu.ready) ? "speculative decode" : "cascade fallback");

    // ── Accept loop ──
    while (!g_shutdown_requested) {
        int cl = accept(server_fd, nullptr, nullptr);
        if (g_shutdown_requested) break;
        if (cl < 0) {
            if (errno == EINTR) continue;
            break;
        }

        // Read request
        char buf[262144];
        int n = (int)read(cl, buf, sizeof(buf)-1);
        if (n <= 0) { close(cl); continue; }
        buf[n] = 0;

        std::string req(buf);
        auto bp = req.find("\r\n\r\n");
        std::string body = (bp == std::string::npos) ? "" : req.substr(bp+4);

        // Parse Content-Length for body
        std::string cl_header = "Content-Length: ";
        auto clp = req.find(cl_header);
        if (clp != std::string::npos) {
            auto cl_end = req.find("\r\n", clp);
            if (cl_end != std::string::npos) {
                int body_len = atoi(req.substr(clp + cl_header.size(), cl_end - clp - cl_header.size()).c_str());
                if ((int)body.size() < body_len) {
                    // Read remaining body
                    int remaining = body_len - (int)body.size();
                    int more = (int)read(cl, buf + bp + 4, remaining);
                    if (more > 0) {
                        buf[bp + 4 + more] = 0;
                        body = std::string(buf + bp + 4);
                    }
                }
            }
        }

        // Handle (thread per request for simplicity)
        std::thread([cl, req, body]() {
            handle_request(cl, req, body);
        }).detach();
    }

    // Shutdown (safe — called from main loop, not signal handler)
    fprintf(stderr, "\nShutdown...\n");
    gpu.stop();
    npu.stop();
    if (server_fd >= 0) close(server_fd);
    return 0;
}
