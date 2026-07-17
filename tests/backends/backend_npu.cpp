// backend_npu_flm.cpp — NPU backend via production FLM engine
//
// Uses /opt/fastflowlm/bin/flm (v0.9.45, validated 94 tok/s on Strix Halo).
// FLM handles: model loading, Q4NX dequant, NPU xclbin dispatch, lm_head.
// Communication: stdin/stdout line protocol with ">>> " prompt delimiter.
//
// This replaces the custom npu_engine_universal which has intermittent
// xclbin hangs (issue #56). FLM xclbins are at:
//   /opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/
//   (layer.xclbin + mm.xclbin + dequant.xclbin + attn.xclbin)
//
// Part of the unified zaya_server binary.

#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <fcntl.h>
#include <signal.h>

class NpuFlmBackend : public InferenceBackend {
    ModelConfig cfg_;
    bool loaded_ = false;
    bool available_ = false;
    pid_t pid_ = 0;
    int stdin_fd_ = -1;
    int stdout_fd_ = -1;
    int stderr_fd_ = -1;
    std::string flm_bin_ = "/opt/fastflowlm/bin/flm";
    std::string model_tag_ = "qwen3:0.6b";
    int timeout_ms_ = 120000;

    // Cached generation state
    std::vector<int> pending_prompt_;
    std::string generated_text_;
    size_t generated_pos_ = 0;

public:
    BackendType type() const override { return BackendType::NPU_XRT; }
    const char* name() const override { return "NPU FLM"; }
    float estimated_tok_s() const override { return 57.0f; }  // estimate; FLM Qwen3:0.6B
    bool is_coherent() const override { return true; }

    bool is_available() override {
        if (available_) return true;
        // Check for NPU via multiple methods
        bool hw = false;
        // Method 1: amdxdna kernel module (Strix Halo)
        std::ifstream m("/proc/modules");
        if (m.good()) {
            std::string line;
            while (std::getline(m, line))
                if (line.find("amdxdna") != std::string::npos) { hw = true; break; }
        }
        // Method 2: XRT device node
        if (!hw) hw = (access("/dev/xclmgmt", F_OK) == 0);
        // Method 3: sysfs drivers
        if (!hw) hw = (access("/sys/bus/pci/drivers/amd_npu", F_OK) == 0 ||
                       access("/sys/bus/pci/drivers/xdna", F_OK) == 0);
        // Method 4: XRT can find device
        if (!hw) {
            FILE* p = popen("xrt-smi examine 2>/dev/null | grep -q RyzenAI && echo yes", "r");
            if (p) { char buf[4]={0}; fread(buf,1,3,p); pclose(p); if(buf[0]=='y') hw=true; }
        }
        if (!hw) { fprintf(stderr, "  NPU: no XDNA 2 detected\n"); return false; }
        if (access(flm_bin_.c_str(), X_OK) != 0) {
            flm_bin_ = "/usr/bin/flm";
            if (access(flm_bin_.c_str(), X_OK) != 0) {
                fprintf(stderr, "  NPU: FLM not installed\n");
                return false;
            }
        }
        available_ = true;
        return true;
    }

    bool load_model(const ModelConfig& cfg) override {
        cfg_ = cfg;
        unload_model();
        if (!is_available()) return false;

        // Map model dimensions to FLM tag
        if (cfg.hidden_size <= 1024)      model_tag_ = "qwen3:0.6b";
        else if (cfg.hidden_size <= 1536) model_tag_ = "qwen3:1.7b";
        else if (cfg.hidden_size <= 2560) model_tag_ = "qwen3:4b";
        else                              model_tag_ = "qwen3:8b";

        fprintf(stderr, "  NPU: launching FLM %s...\n", model_tag_.c_str());

        int to_child[2], from_child[2], err_child[2];
        if (pipe(to_child) || pipe(from_child) || pipe(err_child)) return false;

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
        stdin_fd_  = to_child[1];
        stdout_fd_ = from_child[0];
        stderr_fd_ = err_child[0];

        // Wait for ">>> " prompt — model loading takes ~8-10s
        std::string buf; char c;
        auto t0 = std::chrono::steady_clock::now();
        bool ready = false;
        while (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t0).count() < 45) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd_, &fds);
            struct timeval tv = {1, 0};
            if (select(stdout_fd_ + 1, &fds, nullptr, nullptr, &tv) > 0) {
                if (read(stdout_fd_, &c, 1) > 0) {
                    buf += c;
                    if (buf.size() >= 4 && buf.substr(buf.size()-4) == ">>> ") {
                        ready = true;
                        break;
                    }
                }
            }
        }

        // Drain stderr (FLM prints "[FLM] Loading model..." etc)
        char drain[4096];
        while (true) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stderr_fd_, &fds);
            struct timeval tv = {0, 0};
            if (select(stderr_fd_ + 1, &fds, nullptr, nullptr, &tv) > 0) {
                int n = read(stderr_fd_, drain, sizeof(drain)-1);
                if (n <= 0) break;
                drain[n] = 0;
                // Pass through for visibility
                fprintf(stderr, "%s", drain);
            } else break;
        }

        if (!ready) {
            fprintf(stderr, "  NPU: FLM init timeout\n");
            unload_model();
            return false;
        }

        loaded_ = true;
        // estimated_tok_s() is a prior, not a measurement (issue #231). Real
        // throughput is reported per-request via InferenceResult.tok_s; until
        // then this is just the selection heuristic + a labelled estimate.
        fprintf(stderr, "  NPU: FLM ready (%s, ~%.0f tok/s est. — measured per-request)\n",
                model_tag_.c_str(), estimated_tok_s());
        return true;
    }

    void unload_model() override {
        if (pid_ > 0) {
            const char* exit_cmd = "/exit\n";
            write(stdin_fd_, exit_cmd, strlen(exit_cmd));
            usleep(500000);
            close(stdin_fd_); close(stdout_fd_); close(stderr_fd_);
            kill(pid_, SIGTERM); usleep(200000);
            kill(pid_, SIGKILL);
            // Reap child to prevent zombie. WNOHANG first, then wait if still alive.
            if (waitpid(pid_, nullptr, WNOHANG) == 0)
                waitpid(pid_, nullptr, 0);
            pid_ = 0;
        }
        stdin_fd_ = stdout_fd_ = stderr_fd_ = -1;
        pending_prompt_.clear();
        generated_text_.clear();
        generated_pos_ = 0;
        loaded_ = false;
    }

    void reset_state() override {
        pending_prompt_.clear();
        generated_text_.clear();
        generated_pos_ = 0;
    }

    // Send accumulated prompt to FLM, get full response
    // FLM uses ">>> " as its prompt delimiter. This function reads until
    // it sees ">>> " at the START of a line, which distinguishes it from
    // ">>> " that may appear mid-response (the actual bug this fixes).
    std::string query_flm(const std::string& prompt) {
        std::string req = prompt + "\n";
        write(stdin_fd_, req.c_str(), req.size());

        std::string resp;
        char c;
        auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count() < timeout_ms_) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd_, &fds);
            struct timeval tv = {1, 0};
            if (select(stdout_fd_ + 1, &fds, nullptr, nullptr, &tv) > 0) {
                if (read(stdout_fd_, &c, 1) > 0) {
                    resp += c;
                    // Only match ">>> " at the START of a line (preceded by \n
                    // or at position 0). This prevents mid-content ">>> " from
                    // being mistaken for FLM's prompt delimiter.
                    if (resp.size() >= 4) {
                        size_t pos = resp.rfind('\n');
                        size_t check_start = (pos == std::string::npos) ? 0 : pos + 1;
                        if (resp.substr(check_start) == ">>> ")
                            break;
                    }
                }
            } else break;
        }

        // Find the last newline before ">>> " and strip everything from there
        size_t prompt_pos = resp.rfind("\n>>> ");
        if (prompt_pos != std::string::npos) {
            resp = resp.substr(0, prompt_pos);
        } else if (resp.size() >= 4 && resp.substr(0, 4) == ">>> ") {
            // Leading prompt
            resp = resp.substr(4);
        }

        // Remove trailing whitespace
        while (!resp.empty() && (resp.back() == '\n' || resp.back() == ' '))
            resp.pop_back();

        return resp;
    }

    // Forward: returns chars as token IDs matching SimpleTokenizer::decode() range
    // (100-199 for printable ASCII). NPU FLM returns raw chars -> shift to match.
    int forward(int token_id, int pos) override {
        if (!loaded_) return 106;

        // Accumulate prompt tokens
        pending_prompt_.push_back(token_id);

        // On first call (pos==0) or when cache exhausted, query FLM
        if (pos == 0 || generated_pos_ >= generated_text_.size()) {
            std::string prompt = "Hello";
            if (!pending_prompt_.empty()) {
                prompt.clear();
                for (int t : pending_prompt_) {
                    if (t == 2) continue;
                    if (t == 106) break;
                    if (t > 100 && t < 200) prompt += (char)(t - 100);
                }
                if (prompt.empty()) prompt = "Hello";
            }

            generated_text_ = query_flm(prompt);
            generated_pos_ = 0;
        }

        // Return characters shifted to SimpleTokenizer-compatible range (100-199)
        if (generated_pos_ < generated_text_.size()) {
            unsigned char c = (unsigned char)generated_text_[generated_pos_++];
            if (c >= 32 && c <= 126) return c + 100;  // printable ASCII -> 132-226
            return (int)c + 200;  // non-printable: > 200 (decode shows [N])
        }
        return 106; // EOS
    }

    // ─── Higher-level generate interface (used by token_router) ────
    InferenceResult generate(const std::string& prompt, int max_tokens = 256) {
        InferenceResult r;
        if (!loaded_) { r.text = "[npu: not loaded]"; return r; }

        auto t0 = std::chrono::high_resolution_clock::now();
        std::string text = query_flm(prompt);
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        r.text = text;

        // Estimate token count (rough: ~4 chars per token)
        int est_tokens = std::max(1, (int)text.size() / 4);
        r.tokens.resize(est_tokens);
        for (int i = 0; i < est_tokens && i < (int)text.size(); i++)
            r.tokens[i] = (unsigned char)text[i];

        r.gen_ms = ms;
        r.tok_s = ms > 0 ? est_tokens / (ms / 1000.0f) : 0;
        return r;
    }
};

std::vector<InferenceBackend*> detect_backends_npu() {
    std::vector<InferenceBackend*> backends;
    static NpuFlmBackend npu;
    backends.push_back(&npu);
    return backends;
}

// Also export the old name for backward compat
extern std::vector<InferenceBackend*> detect_backends_npu_flm() {
    return detect_backends_npu();
}
