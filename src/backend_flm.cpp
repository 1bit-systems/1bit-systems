// backend_flm.cpp — NPU inference via the FastFlowLM (FLM) subprocess.
//
// The project's own in-process NPU GEMM kernels are confirmed producing wrong
// output on real hardware (docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md) — this
// backend exists specifically to give NPU-class models a *working* path while
// that's unresolved, by shelling out to AMD's FastFlowLM (github.com/amd/fastflowlm)
// instead. Ported from the prototype at tests/backends/backend_npu_flm.cpp
// (InferenceBackend interface, toy char-offset "tokenizer") to the canonical
// Backend interface with real tokenizer bridging via the shared g_tokenizer
// (include/simple_tokenizer.h) — same vocabulary the rest of the server uses,
// so FLM's text output round-trips through re-tokenization correctly.
//
// Known architectural compromise, not a bug: FLM has no per-token logits API,
// only a whole-turn text REPL. This backend queries FLM for a fresh completion
// every time it sees new (non-self-generated) token content, and streams the
// re-tokenized response back one real token at a time. That means it re-queries
// once per prompt token during prefill (each such call's return value is
// discarded by the caller anyway — see tools/unified_server.cpp's
// generate_completion() prefill loop) rather than once per turn. Correct, but
// not maximally efficient; a natural follow-up is widening the Backend
// interface with an explicit prefill/decode boundary so this can query FLM
// exactly once per turn instead.

#include "backend.h"
#include "backend_detect.h"
#include "simple_tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <fcntl.h>
#include <signal.h>

// ── Process helpers ──
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

struct FlmBackend : Backend {
    std::string flm_bin_ = "/opt/fastflowlm/bin/flm";
    std::string model_tag_ = "qwen3:0.6b";
    int ready_timeout_s_ = 45;
    int query_timeout_ms_ = 120000;

    pid_t pid_ = -1;
    int stdin_fd_ = -1, stdout_fd_ = -1, stderr_fd_ = -1;

    // Per-turn state
    std::vector<int> prompt_tokens_;
    std::vector<int> response_tokens_;
    size_t response_pos_ = 0;
    int last_returned_ = -1;

    FlmBackend() { type = BackendType::NPU_FLM; name = "NPU via FastFlowLM"; }
    ~FlmBackend() override { destroy(); }

    bool find_flm_bin() {
        if (access(flm_bin_.c_str(), X_OK) == 0) return true;
        flm_bin_ = "/usr/bin/flm";
        return access(flm_bin_.c_str(), X_OK) == 0;
    }

    // FLM only serves its own bundled Qwen3 catalog, keyed by size tag —
    // not a general GGUF engine. Only route qwen3-architecture models here
    // (see model_router.h); this mapping picks the closest shipped size.
    static std::string tag_for_hidden(int hidden) {
        if (hidden <= 1024) return "qwen3:0.6b";
        if (hidden <= 1536) return "qwen3:1.7b";
        if (hidden <= 2560) return "qwen3:4b";
        return "qwen3:8b";
    }

    bool init(const ModelConfig& model_cfg, const std::string& weights_dir) override {
        (void)weights_dir;
        cfg = model_cfg;
        destroy();

        if (cfg.architecture != "qwen3") {
            fprintf(stderr, "FLM: architecture '%s' not in FLM's catalog (qwen3 only)\n",
                    cfg.architecture.c_str());
            return false;
        }
        if (!has_npu()) { fprintf(stderr, "FLM: no XDNA NPU detected\n"); return false; }
        if (!find_flm_bin()) { fprintf(stderr, "FLM: flm binary not found\n"); return false; }

        model_tag_ = tag_for_hidden(cfg.hidden);
        fprintf(stderr, "FLM: launching %s...\n", model_tag_.c_str());

        int to_child[2], from_child[2], err_child[2];
        if (pipe(to_child)) return false;
        if (pipe(from_child)) { close(to_child[0]); close(to_child[1]); return false; }
        if (pipe(err_child)) {
            close(to_child[0]); close(to_child[1]); close(from_child[0]); close(from_child[1]); return false;
        }

        pid_ = fork();
        if (pid_ == 0) {
            dup2(to_child[0], STDIN_FILENO);
            dup2(from_child[1], STDOUT_FILENO);
            dup2(err_child[1], STDERR_FILENO);
            close(to_child[0]); close(to_child[1]);
            close(from_child[0]); close(from_child[1]);
            close(err_child[0]); close(err_child[1]);
            execl(flm_bin_.c_str(), "flm", "run", model_tag_.c_str(), nullptr);
            _exit(1);
        }
        close(to_child[0]); close(from_child[1]); close(err_child[1]);
        stdin_fd_ = to_child[1];
        stdout_fd_ = from_child[0];
        stderr_fd_ = err_child[0];

        // Wait for the ">>> " prompt — model load takes several seconds.
        std::string buf; char c;
        auto t0 = std::chrono::steady_clock::now();
        bool ready = false;
        while (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t0).count() < ready_timeout_s_) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd_, &fds);
            struct timeval tv = {1, 0};
            if (select(stdout_fd_ + 1, &fds, nullptr, nullptr, &tv) > 0) {
                if (read(stdout_fd_, &c, 1) > 0) {
                    buf += c;
                    if (buf.size() >= 4 && buf.substr(buf.size() - 4) == ">>> ") { ready = true; break; }
                }
            }
        }
        // Drain stderr for visibility (FLM prints load progress there).
        char drain[4096];
        for (;;) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stderr_fd_, &fds);
            struct timeval tv = {0, 0};
            if (select(stderr_fd_ + 1, &fds, nullptr, nullptr, &tv) <= 0) break;
            int n = read(stderr_fd_, drain, sizeof(drain) - 1);
            if (n <= 0) break;
            drain[n] = 0;
            fprintf(stderr, "%s", drain);
        }
        if (!ready) { fprintf(stderr, "FLM: init timeout\n"); destroy(); return false; }

        initialized = true;
        fprintf(stderr, "FLM: ready (%s)\n", model_tag_.c_str());
        return true;
    }

    bool reset() override {
        prompt_tokens_.clear();
        response_tokens_.clear();
        response_pos_ = 0;
        last_returned_ = -1;
        return true;
    }

    // FLM's REPL interleaves the real answer with "[FLM] ..." progress/log
    // lines (some ANSI-colored) and repeats the answer a second time under
    // a "[FLM]  Model RAW Output:" marker. Extract just the first clean,
    // non-"[FLM]" span of text — everything from after the leading log
    // lines up to the next "[FLM]" line (real behavior verified by
    // capturing raw output directly: `flm run <tag>` piped input/output).
    static std::string strip_ansi(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); i++) {
            if ((unsigned char)s[i] == 0x1b && i + 1 < s.size() && s[i + 1] == '[') {
                size_t j = i + 2;
                while (j < s.size() && !isalpha((unsigned char)s[j])) j++;
                i = j; // skip terminating letter too (loop's ++ advances past it)
                continue;
            }
            out += s[i];
        }
        return out;
    }

    static std::string extract_answer(const std::string& raw) {
        std::string clean = strip_ansi(raw);
        std::vector<std::string> lines;
        size_t start = 0;
        for (size_t i = 0; i <= clean.size(); i++) {
            if (i == clean.size() || clean[i] == '\n') {
                lines.push_back(clean.substr(start, i - start));
                start = i + 1;
            }
        }
        std::string answer;
        bool collecting = false;
        for (auto& line : lines) {
            bool is_log = line.rfind("[FLM]", 0) == 0;
            if (is_log) {
                if (collecting) break; // hit the "Model RAW Output:" duplicate — stop
                continue;              // still in the leading log lines — skip
            }
            if (line.empty() && !collecting) continue; // skip blank lines before the answer starts
            collecting = true;
            if (!answer.empty()) answer += '\n';
            answer += line;
        }
        while (!answer.empty() && (answer.back() == '\n' || answer.back() == ' '))
            answer.pop_back();
        return answer;
    }

    // Whole-turn text exchange over the ">>> "-delimited REPL.
    std::string query_flm(const std::string& prompt) {
        std::string req = prompt + "\n";
        if (write(stdin_fd_, req.c_str(), req.size()) < 0) return "";

        std::string resp;
        char c;
        auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count() < query_timeout_ms_) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd_, &fds);
            struct timeval tv = {1, 0};
            if (select(stdout_fd_ + 1, &fds, nullptr, nullptr, &tv) > 0) {
                if (read(stdout_fd_, &c, 1) > 0) {
                    resp += c;
                    if (resp.size() >= 4 && resp.substr(resp.size() - 4) == ">>> ") break;
                }
            } else break;
        }
        if (resp.size() >= 4 && resp.substr(resp.size() - 4) == ">>> ")
            resp = resp.substr(0, resp.size() - 4);
        // Drop the echoed input line (REPL echoes what was piped to it).
        auto nl = resp.find('\n');
        if (nl != std::string::npos) resp = resp.substr(nl + 1);
        return extract_answer(resp);
    }

    bool forward(int, float*) override { return false; } // no hidden states — use generate()
    bool lm_head(const float*, float*, int*) override { return false; } // use generate()

    int generate(int token_id) override {
        if (!initialized) return -1;

        if (token_id == last_returned_ && !response_tokens_.empty()) {
            // Continuing to drain an already-computed response.
            if (response_pos_ < response_tokens_.size()) {
                last_returned_ = response_tokens_[response_pos_++];
                return last_returned_;
            }
            last_returned_ = g_tokenizer.eos_id;
            return last_returned_;
        }

        // New prompt content (prefill, or the boundary token before decode
        // begins — see the file-level comment for why we can't tell which).
        prompt_tokens_.push_back(token_id);
        std::string prompt = g_tokenizer.decode(prompt_tokens_);
        if (prompt.empty()) prompt = " ";

        // Every distinct new-token call would otherwise re-query FLM, which
        // both wastes round-trips AND pollutes FLM's own session context with
        // discarded partial-prompt exchanges before the real one. Under a
        // real BPE tokenizer each token is already word/subword-sized, so
        // this is a minor inefficiency — but under the char-level ASCII
        // fallback (verified: with no .htok available, each character is
        // its own token) it becomes a query storm, one per character, that
        // measurably degrades answer quality. Mitigate by only actually
        // querying at a word/punctuation boundary (or if nothing has been
        // queried yet this turn) — cuts query count roughly to one per word
        // instead of one per token, without needing a prefill/decode signal
        // the Backend interface doesn't provide.
        char last_ch = prompt.empty() ? ' ' : prompt.back();
        bool at_boundary = last_ch == ' ' || last_ch == '\n' || ispunct((unsigned char)last_ch);
        if (!at_boundary && !response_tokens_.empty()) {
            // Mid-word continuation of a prompt we've already queried once —
            // keep the previous answer as our best guess rather than
            // re-querying (avoids the storm; the final boundary call before
            // decode will produce the real, complete-prompt answer).
            last_returned_ = response_tokens_[0];
            return last_returned_;
        }

        std::string text = query_flm(prompt);
        response_tokens_ = g_tokenizer.encode(text);
        response_pos_ = 0;
        if (response_tokens_.empty()) { last_returned_ = g_tokenizer.eos_id; return last_returned_; }
        last_returned_ = response_tokens_[response_pos_++];
        return last_returned_;
    }

    void destroy() override {
        if (pid_ > 0) {
            // Send /exit command and close stdin to signal EOF (issue #365)
            const char* exit_cmd = "/exit\n";
            if (stdin_fd_ >= 0) {
                write(stdin_fd_, exit_cmd, strlen(exit_cmd));
                close(stdin_fd_);
                stdin_fd_ = -1;
            }
            // Wait up to 500ms for graceful exit after /exit command
            if (!wait_for_child(pid_, 500)) {
                kill(pid_, SIGTERM);
                if (!wait_for_child(pid_, 2000)) {
                    kill(pid_, SIGKILL);
                    wait_for_child(pid_, 1000);
                }
            }
            int status;
            waitpid(pid_, &status, WNOHANG);
            pid_ = -1;
        }
        if (stdout_fd_ >= 0) { close(stdout_fd_); stdout_fd_ = -1; }
        if (stderr_fd_ >= 0) { close(stderr_fd_); stderr_fd_ = -1; }
        stdin_fd_ = stdout_fd_ = stderr_fd_ = -1;
        initialized = false;
    }

    float benchmark(int tokens) override {
        if (!initialized) return -1.0f;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = g_tokenizer.bos_id;
        for (int i = 0; i < tokens; i++) tok = generate(tok);
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return tokens > 0 ? ms / tokens : 0.0f;
    }

    bool can_infer() const override { return initialized; }
};

extern "C" Backend* create_flm_backend() { return new FlmBackend(); }
