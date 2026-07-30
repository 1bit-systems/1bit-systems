// backend_npu.cpp — Real NPU inference backend via subprocess worker protocol.
//
// Spawns engine/npu/npu_engine_universal as a subprocess in --worker mode
// and communicates via its stdin/stdout binary protocol for GEMM operations.
// CPU-side handles: attention, RoPE, RMSNorm, residual add, SiLU gating,
// KV cache management, and lm_head.
//
// Model weights: loads only embed + norm tables directly from the model file
// (small ~few MB). The GB-scale GEMM weights are managed by the worker engine.
//
// Env vars:
//   NPU_MODEL_PATH   — path to model.q4nx (required)
//   NPU_ENGINE_BIN   — path to npu_engine_universal binary
//   NPU_XCLBIN_DIR   — xclbin directory (passed through to worker)

#include "backend.h"
#include "q4nx_reader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <mutex>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/select.h>
#include <signal.h>

// ── Math helpers (CPU fallback ops) ──
static constexpr float EPS = 1e-6f;
static inline void cn(float* x, int n) {
    for (int i = 0; i < n; i++) if (!std::isfinite(x[i])) x[i] = 0.0f;
}
static inline void rmsnorm(float* x, const float* w, int n) {
    cn(x, n); double ss = 0;
    for (int i = 0; i < n; i++) if (std::isfinite(x[i])) ss += (double)x[i] * x[i];
    float ir = 1.0f / sqrtf((float)(ss / n) + EPS);
    for (int i = 0; i < n; i++) x[i] = std::isfinite(x[i]) ? x[i] * ir * w[i] : 0.0f;
}
static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// RoPE cache (built once at init, thread-safe via mutex)
static std::vector<float> cos_cache, sin_cache;
static int rope_hd = 0;
static std::mutex rope_cache_mutex;
static void build_rope_cache(int max_pos, int head_dim, float theta) {
    std::lock_guard<std::mutex> lock(rope_cache_mutex);
    if (head_dim == rope_hd && !cos_cache.empty()) return; // already built — double-checked under lock
    rope_hd = head_dim;
    int hd2 = head_dim / 2;
    cos_cache.resize(max_pos * head_dim);
    sin_cache.resize(max_pos * head_dim);
    for (int p = 0; p < max_pos; p++) {
        for (int d = 0; d < hd2; d++) {
            float f = 1.0f / powf(theta, (float)d / hd2);
            float a = p * f;
            cos_cache[p * head_dim + d] = cosf(a);
            sin_cache[p * head_dim + d] = sinf(a);
        }
    }
}
static inline void rope(float* x, int head_dim, int pos) {
    int hd2 = head_dim / 2;
    for (int d = 0; d < hd2; d++) {
        float a = x[d], b = x[d + hd2];
        float c = cos_cache[pos * head_dim + d];
        float s = sin_cache[pos * head_dim + d];
        x[d] = a * c - b * s;
        x[d + hd2] = b * c + a * s;
    }
}

// CPU attention: Q[NQ,HD] @ K[seq,NKV,HD] -> scores, softmax, weighted V
static void attn_cpu(float* qo, float* at, int seq_len,
                     const float* kv_k, const float* kv_v,
                     int NQ, int NKV, int HD, int GQA) {
    constexpr int MAX_CTX = 4096;
    #pragma omp parallel for
    for (int hh = 0; hh < NQ; hh++) {
        int kvh = hh / GQA;
        float scores[MAX_CTX];
        float mx = -1e30f;
        for (int p = 0; p < seq_len; p++) {
            double s = 0;
            for (int d = 0; d < HD; d++)
                s += (double)qo[hh * HD + d] * kv_k[(size_t)p * NKV * HD + (size_t)kvh * HD + d];
            scores[p] = (float)(s / sqrtf((float)HD));
            if (scores[p] > mx) mx = scores[p];
        }
        double sw = 0;
        for (int p = 0; p < seq_len; p++) { scores[p] = expf(scores[p] - mx); sw += scores[p]; }
        float isw = (sw > 0 && seq_len > 0) ? 1.0f / (float)sw : 1.0f / (seq_len > 0 ? (float)seq_len : 1.0f);
        for (int d = 0; d < HD; d++) {
            float acc = 0;
            for (int p = 0; p < seq_len; p++)
                acc += scores[p] * kv_v[(size_t)p * NKV * HD + (size_t)kvh * HD + d];
            at[hh * HD + d] = acc * isw;
        }
    }
}

// ── Process helpers ──
// Read exactly `len` bytes from `fd`, bounded by `timeout_ms` total across
// the whole read (not per-call) — a single blocking read() has no timeout
// at all, so a hung NPU worker subprocess would block the calling httplib
// thread forever, eventually exhausting the whole server's thread pool.
// Also handles short reads (pipes don't guarantee delivering the full
// requested length in one read()), which a bare read() call did not.
static bool read_with_timeout(int fd, void* buf, size_t len, int timeout_ms) {
    size_t got = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (got < len) {
        int remaining_ms = timeout_ms - (int)std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (remaining_ms <= 0) return false;
        fd_set fds; FD_ZERO(&fds); FD_SET(fd, &fds);
        struct timeval tv = {remaining_ms / 1000, (remaining_ms % 1000) * 1000};
        int r = select(fd + 1, &fds, nullptr, nullptr, &tv);
        if (r <= 0) return false; // timeout or select error
        ssize_t n = read(fd, (char*)buf + got, len - got);
        if (n <= 0) return false; // EOF or read error
        got += (size_t)n;
    }
    return true;
}

// Wait up to timeout_ms for child to exit. Returns true if exited.
static bool wait_for_child(pid_t pid, int timeout_ms) {
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t0).count() < timeout_ms) {
        int status;
        pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) return true;
        if (r < 0) return true;
        usleep(10000); // 10ms poll interval
    }
    return false;
}

// ── NPU Worker subprocess ──
struct NpuWorker {
    pid_t pid = -1;
    int stdin_fd = -1;   // write to worker's stdin
    int stdout_fd = -1;  // read from worker's stdout
    bool ready = false;

    bool spawn(const std::string& model_path,
              int H, int NC, int NQ, int NKV, int HD, int IM, int NV) {
        // A worker that has already died (crash, or our own SIGKILL during
        // shutdown) turns every subsequent write() to its stdin pipe into
        // SIGPIPE, whose default action terminates the ENTIRE host process —
        // not just the failing write. The gemm() and shutdown() writes below
        // are exactly those writes. Ignore SIGPIPE process-wide, once, so those
        // writes return -1/EPIPE and are caught by the existing short-write
        // checks instead of crashing the server (AUDIT_ISSUES.md #3). Only
        // tools/token_router.cpp did this before, leaving unified_server /
        // zaya_server (the real NPU hosts) exposed. Function-local static init
        // is thread-safe and runs exactly once per process.
        static const bool _sigpipe_ignored = []{
            signal(SIGPIPE, SIG_IGN);
            return true;
        }();
        (void)_sigpipe_ignored;

        const char* engine_bin = getenv("NPU_ENGINE_BIN");
        std::string bin = engine_bin ? engine_bin : "./npu_engine_universal";

        int to_child[2], from_child[2];
        if (pipe(to_child) < 0) { perror("NPU: pipe(to_child)"); return false; }
        if (pipe(from_child) < 0) { perror("NPU: pipe(from_child)"); close(to_child[0]); close(to_child[1]); return false; }

        // Set env vars so worker loads matching dimensions (#445)
        auto set_env_int = [](const char* k, int v) {
            char buf[32]; snprintf(buf, sizeof(buf), "%d", v);
            setenv(k, buf, 1);
        };
        set_env_int("NPU_H", H);
        set_env_int("NPU_NC", NC);
        set_env_int("NPU_NH", NQ);
        set_env_int("NPU_NKV", NKV);
        set_env_int("NPU_HD", HD);
        set_env_int("NPU_IM", IM);
        set_env_int("NPU_NV", NV);

        pid = fork();
        if (pid < 0) { perror("NPU: fork"); close(to_child[0]); close(to_child[1]); close(from_child[0]); close(from_child[1]); return false; }

        if (pid == 0) {
            // Child: npu_engine_universal process
            close(to_child[1]); dup2(to_child[0], STDIN_FILENO); close(to_child[0]);
            close(from_child[0]); dup2(from_child[1], STDOUT_FILENO); close(from_child[1]);
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) dup2(devnull, STDERR_FILENO);
            execlp(bin.c_str(), bin.c_str(), model_path.c_str(), "--worker", (char*)nullptr);
            fprintf(stderr, "NPU: failed to exec %s\n", bin.c_str());
            _exit(1);
        }

        close(to_child[0]); close(from_child[1]);
        stdin_fd = to_child[1];
        stdout_fd = from_child[0];

        // Startup handshake: wait for "READY\n" from child (issue #365)
        char ready_buf[6];
        int ready_bytes = 0;
        auto t0 = std::chrono::steady_clock::now();
        while (ready_bytes < 6 && std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - t0).count() < 10) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd, &fds);
            struct timeval tv = {1, 0};
            if (select(stdout_fd + 1, &fds, nullptr, nullptr, &tv) > 0) {
                ssize_t n = read(stdout_fd, ready_buf + ready_bytes, 6 - ready_bytes);
                if (n > 0) ready_bytes += n;
                else if (n <= 0) break;
            }
        }
        if (ready_bytes >= 6 && memcmp(ready_buf, "READY\n", 6) == 0) {
            ready = true;
        } else {
            fprintf(stderr, "NPU: worker handshake failed (got %d bytes)\n", ready_bytes);
            kill(pid, SIGTERM); waitpid(pid, nullptr, 0);
            close(stdin_fd); close(stdout_fd); stdin_fd = stdout_fd = -1; pid = -1;
            return false;
        }
        return true;
    }

    // Send a GEMM operation, read back result. out_data auto-resized.
    // Bounded by GEMM_TIMEOUT_MS per read — a hung worker subprocess used to
    // block this call (and the calling httplib thread) forever (issue #20).
    static constexpr int GEMM_TIMEOUT_MS = 30000;
    bool gemm(int op, int layer, int batch, int in_dim,
              const float* in_data, std::vector<float>& out_data) {
        if (!ready || stdin_fd < 0 || stdout_fd < 0) return false;
        uint32_t hdr[4] = {(uint32_t)op, (uint32_t)layer, (uint32_t)batch, (uint32_t)in_dim};
        if (write(stdin_fd, hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) return false;
        if (write(stdin_fd, in_data, (size_t)batch * in_dim * sizeof(float)) !=
            (ssize_t)((size_t)batch * in_dim * sizeof(float))) return false;
        uint32_t resp[2];
        if (!read_with_timeout(stdout_fd, resp, sizeof(resp), GEMM_TIMEOUT_MS)) return false;
        if (resp[0] != 0) return false;
        uint32_t out_dim = resp[1];
        // Validate out_dim against known model dimensions to prevent
        // OOM from a buggy/compromised worker subprocess (AUDIT).
        static constexpr uint32_t MAX_SAFE_OUT_DIM = 256 * 1024; // 256K floats max
        if (out_dim > MAX_SAFE_OUT_DIM) {
            fprintf(stderr, "NPU: worker returned out_dim=%u > max=%u — rejecting\n",
                    out_dim, MAX_SAFE_OUT_DIM);
            return false;
        }
        out_data.resize((size_t)batch * out_dim);
        return read_with_timeout(stdout_fd, out_data.data(), (size_t)batch * out_dim * sizeof(float), GEMM_TIMEOUT_MS);
    }

    void shutdown() {
        if (pid > 0) {
            // Send quit command (op=0) and close stdin to signal EOF (issue #365)
            uint32_t quit[4] = {0, 0, 0, 0};
            if (stdin_fd >= 0) {
                write(stdin_fd, quit, sizeof(quit));
                close(stdin_fd);
                stdin_fd = -1;
            }
            // Wait up to 500ms for graceful exit after quit command
            if (!wait_for_child(pid, 500)) {
                kill(pid, SIGTERM);
                if (!wait_for_child(pid, 2000)) {
                    kill(pid, SIGKILL);
                    wait_for_child(pid, 1000);
                }
            }
            int status;
            waitpid(pid, &status, WNOHANG);
            pid = -1;
        }
        if (stdout_fd >= 0) { close(stdout_fd); stdout_fd = -1; }
        stdin_fd = stdout_fd = -1;
        ready = false;
    }

    ~NpuWorker() { shutdown(); }
};

// ── NPU Backend ──
struct NPUBackend : Backend {
    NpuWorker worker;
    Q4nxReader model;

    // Saved for worker respawn on idle-release (issue #1029)
    std::string worker_model_path_;

    // Idle-release: track last inference time so we can release the NPU
    // worker subprocess between requests. The worker holds /dev/accel/accel0;
    // releasing it when idle lets standalone tools (npu_engine_universal)
    // use the NPU without stopping unified_server.
    std::chrono::steady_clock::time_point last_use_;
    static constexpr int IDLE_RELEASE_SECS = 30;

    // Model dimensions
    int H = 0, NQ = 0, NKV = 0, HD = 0, GQA = 0, NV = 0, NC = 0;
    int qkv_total = 0, mlp_dim = 0;
    float rope_theta = 1000000.0f;
    int max_seq_len = 4096;

    // Weights loaded from model file (small: embed + norms)
    std::vector<float> embed;
    std::vector<float> final_norm;
    std::vector<std::vector<float>> in_norms;     // per-layer input norm
    std::vector<std::vector<float>> post_attn_norms; // per-layer post-attn norm
    std::vector<std::vector<float>> q_norms;       // per-layer Q norm (optional)
    std::vector<std::vector<float>> k_norms;       // per-layer K norm (optional)

    // KV cache
    struct KVCache { std::vector<float> k, v; int seq_len = 0; };
    std::vector<KVCache> kv_caches;

    // State buffers
    std::vector<float> hidden;
    std::vector<float> qkv_buf, attn_buf, o_buf, gateup_buf, up_buf, down_buf;
    std::vector<float> residual_buf, logits_buf;
    int pos = 0;

    bool has_q_norm = false, has_k_norm = false, gu_split = false;

    NPUBackend() { type = BackendType::NPU_XRT; name = "NPU XDNA (worker subprocess)"; }
    ~NPUBackend() override { destroy(); }
    bool can_infer() const override { return initialized; }

    bool init(const ModelConfig& cfg, const std::string& weights_dir) override {
        this->cfg = cfg;
        printf("NPU: Initializing worker subprocess...\n");

        const char* model_path = getenv("NPU_MODEL_PATH");
        std::string discovered_path;
        if (!model_path) {
            // Auto-discovery: search common paths for model.q4nx (#444)
            // 1. Current dir + common paths
            const char* home_model = getenv("HOME");
            static std::string home_model_path = (home_model && home_model[0]) ? std::string(home_model) + "/.local/share/1bit-systems/weights/model.q4nx" : "";
            static const char* search_paths[] = {
                "model.q4nx",
                "models/model.q4nx",
                "/opt/1bit/models/model.q4nx",
                home_model_path.c_str(),
            };
            // 2. FLM model directory (users already have models here)
            const char* home_env = getenv("HOME");
            std::string flm_dir = std::string(home_env ? home_env : "") + "/.config/flm/models";
            std::string flm_candidates[8];
            int n_flm = 0;
            if (!flm_dir.empty()) {
                auto* dir = opendir(flm_dir.c_str());
                if (dir) {
                    struct dirent* entry;
                    while ((entry = readdir(dir)) && n_flm < 8) {
                        if (entry->d_name[0] == '.') continue;
                        std::string mp = flm_dir + "/" + entry->d_name + "/model.q4nx";
                        if (access(mp.c_str(), R_OK) == 0)
                            flm_candidates[n_flm++] = mp;
                    }
                    closedir(dir);
                }
            }
            // 3. weights_dir fallback
            std::string wd_path = weights_dir + "/model.q4nx";
            // Try FLM models first (largest = most capable), then common paths
            for (int i = 0; i < n_flm; i++) {
                if (access(flm_candidates[i].c_str(), R_OK) == 0) {
                    discovered_path = flm_candidates[i];
                    model_path = discovered_path.c_str();
                    fprintf(stderr, "NPU: auto-discovered model at: %s\n", model_path);
                    break;
                }
            }
            if (!model_path) {
                const char* std_paths[] = {
                    wd_path.c_str(),
                    search_paths[0], search_paths[1], search_paths[2], search_paths[3],
                };
                for (const char* cand : std_paths) {
                    if (!cand || !cand[0]) continue;
                    if (access(cand, R_OK) == 0) {
                        discovered_path = cand;
                        model_path = discovered_path.c_str();
                        fprintf(stderr, "NPU: auto-discovered model at: %s\n", model_path);
                        break;
                    }
                }
            }
            if (!model_path) {
                fprintf(stderr, "NPU: no model found — set NPU_MODEL_PATH or place model.q4nx in FLM models dir (~/.config/flm/models/)\n");
                // FLM models dir is the recommended default — list it
                fprintf(stderr, "  • ~/.config/flm/models/<model>/model.q4nx\n");
                return false;
            }
        }

        // Read model dimensions from config
        H = cfg.hidden_size;
        NQ = cfg.num_attention_heads;
        NKV = cfg.num_kv_heads;
        HD = cfg.head_dim;
        GQA = NQ / NKV;
        NV = cfg.vocab_size;
        NC = cfg.num_layers;
        qkv_total = NQ * HD + NKV * HD * 2;
        mlp_dim = cfg.intermediate_size;
        rope_theta = cfg.rope_theta > 0 ? cfg.rope_theta : 1000000.0f;
        max_seq_len = cfg.max_seq_len > 0 ? cfg.max_seq_len : 4096;
        has_q_norm = cfg.has_q_norm;
        has_k_norm = cfg.has_k_norm;
        gu_split = cfg.gu_split;

        // Validate hidden size against stack buffer limits
        if (H > 8192) {
            fprintf(stderr, "NPU: hidden_size %d exceeds max 8192 — refusing to init\n", H);
            worker.shutdown();
            return false;
        }

        // Save model path for lazy worker start (issue #1029).
        // The worker subprocess holds /dev/accel/accel0 — we defer spawning
        // until the first inference request so the NPU is free when idle.
        worker_model_path_ = model_path;

        // Derive model tag from the ACTUAL filename (resolve symlinks)
        // so the worker finds the right xclbins (final_i8_<op>_<tag>.xclbin).
        {
            char realp[4096];
            const char* actual_path = model_path;
            if (realpath(model_path, realp)) actual_path = realp;
            std::string mp(actual_path);
            auto ls = mp.rfind('/');
            std::string tag = (ls != std::string::npos) ? mp.substr(ls + 1) : mp;
            auto dot = tag.rfind('.');
            if (dot != std::string::npos) tag = tag.substr(0, dot);
            // Lowercase, replace separators
            for (auto& c : tag) { c = tolower(c); if (c == '-' || c == '.' || c == '\\') c = '_'; }
            const char* sfxs[] = {"_npu2","_instruct","_it","_it_npu2"};
            for (auto sf : sfxs) {
                size_t sl = strlen(sf);
                if (tag.size() > sl && tag.substr(tag.size() - sl) == sf)
                    tag = tag.substr(0, tag.size() - sl);
            }
            setenv("NPU_MODEL_TAG", tag.c_str(), 1);
            printf("NPU: model tag set to '%s' (from %s)\n", tag.c_str(), actual_path);
        }

        // Load embed + norm weights from model file
        if (!model.open(model_path)) {
            fprintf(stderr, "NPU: failed to open model file\n");
            worker.shutdown();
            return false;
        }

        // Embed table (tied lm_head)
        {
            // JSON keys used by Q4NX format
            uint64_t emb_off = model.find_offset("model_embed_tokens_weight");
            if (!emb_off) emb_off = model.find_offset("model.embed_tokens.weight");
            if (!emb_off) emb_off = model.find_offset("gte");
            if (emb_off) {
                embed = model.read_floats(emb_off, (size_t)NV * H);
                printf("NPU: loaded embed %zu floats\n", embed.size());
            } else {
                fprintf(stderr, "NPU: cannot find embed in model\n");
            }
        }

        // Final norm
        {
            uint64_t fn_off = model.find_offset("model_norm_weight");
            if (!fn_off) fn_off = model.find_offset("model.norm.weight");
            if (fn_off) {
                final_norm = model.read_floats(fn_off, H);
                printf("NPU: loaded final_norm %zu floats\n", final_norm.size());
            }
        }

        // Per-layer norms
        in_norms.resize(NC);
        post_attn_norms.resize(NC);
        q_norms.resize(NC);
        k_norms.resize(NC);
        for (int l = 0; l < NC; l++) {
            char key[256];
            // Input layer norm
            snprintf(key, sizeof(key), "model.layers.%d.input_layernorm.weight", l);
            uint64_t off = model.find_offset(key);
            if (!off) { snprintf(key, sizeof(key), "model.layers.%d.self_attn.input_layernorm.weight", l); off = model.find_offset(key); }
            if (off) in_norms[l] = model.read_floats(off, H);

            // Post-attention norm
            snprintf(key, sizeof(key), "model.layers.%d.post_attention_layernorm.weight", l);
            off = model.find_offset(key);
            if (off) post_attn_norms[l] = model.read_floats(off, H);

            // Q norm (optional)
            if (has_q_norm) {
                snprintf(key, sizeof(key), "model.layers.%d.self_attn.q_norm.weight", l);
                off = model.find_offset(key);
                if (off) q_norms[l] = model.read_floats(off, HD);
            }
            // K norm (optional)
            if (has_k_norm) {
                snprintf(key, sizeof(key), "model.layers.%d.self_attn.k_norm.weight", l);
                off = model.find_offset(key);
                if (off) k_norms[l] = model.read_floats(off, HD);
            }
        }

        // Verify GEMM weights exist in model file (managed by worker subprocess).
        // The GB-scale QKV/O/GU/D projection weights are loaded by the NPU worker
        // engine via its own mmap; the backend only verifies they're present (#445).
        {
            char key[256];
            bool all_found = true;
            for (int l = 0; l < NC && all_found; l++) {
                snprintf(key, sizeof(key), "model.layers.%d.self_attn.q_proj.weight", l);
                if (!model.find_offset(key)) { all_found = false; break; }
                snprintf(key, sizeof(key), "model.layers.%d.self_attn.k_proj.weight", l);
                if (!model.find_offset(key)) { all_found = false; break; }
                snprintf(key, sizeof(key), "model.layers.%d.self_attn.v_proj.weight", l);
                if (!model.find_offset(key)) { all_found = false; break; }
                snprintf(key, sizeof(key), "model.layers.%d.self_attn.o_proj.weight", l);
                if (!model.find_offset(key)) { all_found = false; break; }
                snprintf(key, sizeof(key), "model.layers.%d.mlp.gate_proj.weight", l);
                if (!model.find_offset(key)) { all_found = false; break; }
                snprintf(key, sizeof(key), "model.layers.%d.mlp.up_proj.weight", l);
                if (!model.find_offset(key)) { all_found = false; break; }
                snprintf(key, sizeof(key), "model.layers.%d.mlp.down_proj.weight", l);
                if (!model.find_offset(key)) { all_found = false; break; }
            }
            if (!all_found) {
                fprintf(stderr, "NPU: GEMM weights missing from model file — worker may fail\n");
            } else {
                printf("NPU: verified GEMM weight offsets for %d layers\n", NC);
            }
        }

        // Build RoPE cache
        build_rope_cache(max_seq_len, HD, rope_theta);

        // Allocate KV cache
        kv_caches.resize(NC);
        for (int i = 0; i < NC; i++) {
            kv_caches[i].k.resize((size_t)max_seq_len * NKV * HD, 0);
            kv_caches[i].v.resize((size_t)max_seq_len * NKV * HD, 0);
        }

        // Allocate state buffers
        hidden.resize(H);
        qkv_buf.resize((size_t)qkv_total);
        attn_buf.resize((size_t)NQ * HD);
        o_buf.resize(H);
        gateup_buf.resize((size_t)mlp_dim * 2);
        up_buf.resize(gu_split ? (size_t)mlp_dim : 0);
        down_buf.resize(H);
        residual_buf.resize(H);
        logits_buf.resize(NV);

        printf("NPU: ready — %d layers, H=%d, V=%d, embed=%zu\n", NC, H, NV, embed.size());
        initialized = true;
        return true;
    }

    // Ensure the NPU worker subprocess is running. Called at the start of
    // every forward() call. If the worker was intentionally stopped (idle
    // release) or crashed, respawn it. Worker startup takes ~5s for model
    // dequant+pack; acceptable because it only happens on first request
    // after idle. See issue #1029.
    bool ensure_worker() {
        if (worker.ready) return true;
        printf("NPU: (re)starting worker subprocess...\n");
        if (!worker.spawn(worker_model_path_, H, NC, NQ, NKV, HD, mlp_dim, NV)) {
            fprintf(stderr, "NPU: failed to (re)spawn worker engine\n");
            return false;
        }
        return true;
    }

    bool reset() override {
        pos = 0;
        for (auto& kv : kv_caches) kv.seq_len = 0;
        // Signal the worker to reset its internal KV cache
        if (worker.ready) {
            std::vector<float> dummy;
            worker.gemm(31, 0, 0, 0, nullptr, dummy);
        }
        return true;
    }

    int generate(int token_id) override {
        if (!initialized || !ensure_worker()) return -1;
        last_use_ = std::chrono::steady_clock::now();

        // ── Fused forward in ONE subprocess call ──
        // Uses op=32: embed → all 36 layers → lm_head → next token.
        // 1 IPC call instead of 180. Eliminates ALL pipe overhead.
        std::vector<float> in_data(1, (float)token_id);
        std::vector<float> out_data;
        if (!worker.gemm(32, 0, 1, 1, in_data.data(), out_data)) {
            fprintf(stderr, "NPU: fused generate failed\n");
            return -1;
        }
        int next_token = (int)out_data[0];
        pos++;

        if (worker.ready) {
            auto now = std::chrono::steady_clock::now();
            auto idle_s = std::chrono::duration_cast<std::chrono::seconds>(
                now - last_use_).count();
            if (idle_s >= IDLE_RELEASE_SECS) {
                printf("NPU: idle for %lds, releasing worker subprocess\n", idle_s);
                worker.shutdown();
            }
        }
        return next_token;
    }

    // forward() and lm_head() replaced by fused generate() using op=32.
    // The old per-layer IPC path (op=1-5) is still available as fallback.
    bool forward(int token_id, float* hidden_out) override {
        (void)token_id; (void)hidden_out;
        fprintf(stderr, "NPU: forward() called but generate() should be used\n");
        return false;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        (void)hidden; (void)logits; (void)argmax;
        fprintf(stderr, "NPU: lm_head() called but generate() should be used\n");
        return false;
    }

    float benchmark(int tokens = 10) override {
        if (!initialized) return 0;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 100;
        for (int i = 0; i < tokens; i++) {
            tok = generate(tok);
            if (tok < 0) break;
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }

    void destroy() override {
        worker.shutdown();
        model.close();
        initialized = false;
    }
};

extern "C" Backend* create_npu_backend() { return new NPUBackend(); }
