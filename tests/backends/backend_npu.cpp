// backend_npu.cpp — AMD XDNA 2 NPU backend via subprocess engine
//
// The NPU engine (engine/npu/src/npu_engine_universal.cpp) runs as a
// separate process because it links XRT + xclbins. This backend:
//   1. Detects XDNA 2 NPU hardware
//   2. Launches the pre-built NPU engine binary as a subprocess
//   3. Communicates via stdin/stdout JSON protocol
//   4. Translates to our InferenceBackend interface
//
// Protocol:
//   → {"tokens":[...], "max_new_tokens": N}\n
//   ← {"tokens":[...], "logprobs":[...]}\n
//
// Part of the unified zaya_server binary.

#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>

namespace {
    std::string json_str(const std::string& j, const std::string& k) {
        auto p = j.find("\"" + k + "\"");
        if (p == std::string::npos) return "";
        p = j.find(':', p); if (p == std::string::npos) return "";
        p = j.find_first_of("\"", p);
        if (p == std::string::npos || j[p] != '\"') {
            auto ns = j.find_first_of("-0123456789", p + 1);
            if (ns != std::string::npos) {
                auto ne = j.find_first_not_of("0123456789.e-+", ns);
                return j.substr(ns, ne - ns);
            }
            return "";
        }
        auto e = j.find('\"', p + 1);
        if (e == std::string::npos) return "";
        return j.substr(p + 1, e - p - 1);
    }
    int json_int(const std::string& j, const std::string& k, int d = 0) {
        auto s = json_str(j, k);
        if (s.empty()) return d;
        return atoi(s.c_str());
    }
}

class NpuBackend : public InferenceBackend {
    ModelConfig cfg_;
    bool loaded_ = false;
    bool available_ = false;
    pid_t engine_pid_ = 0;
    int stdin_fd_ = -1;
    int stdout_fd_ = -1;
    std::string engine_bin_;
    int timeout_ms_ = 120000;
    std::string model_variant_;
    std::vector<int> pending_tokens_;

public:
    BackendType type() const override { return BackendType::NPU; }
    const char* name() const override { return "NPU XDNA 2"; }
    float estimated_tok_s() const override { return 69.0f; }
    bool is_coherent() const override { return true; }

    NpuBackend() {
        engine_bin_ = "engine/npu/build/npu_engine";
        if (access(engine_bin_.c_str(), X_OK) != 0)
            engine_bin_ = "./build/npu_engine";
        if (access(engine_bin_.c_str(), X_OK) != 0)
            engine_bin_ = "/home/bcloud/engine/npu/build/npu_engine";
    }

    bool is_available() override {
        if (available_) return true;
        bool hw_found = false;
        if (access("/dev/xclmgmt", F_OK) == 0) {
            hw_found = true;
            fprintf(stderr, "  NPU: XRT device node found (/dev/xclmgmt)\n");
        }
        std::ifstream sf("/sys/class/drm/renderD128/device/vendor");
        if (sf.good()) {
            std::string vendor;
            sf >> vendor;
            if (vendor == "0x1002") { hw_found = true; fprintf(stderr, "  NPU: AMD GPU/NPU node detected\n"); }
        }
        if (access("/sys/bus/pci/drivers/amd_npu", F_OK) == 0 ||
            access("/sys/bus/pci/drivers/xdna", F_OK) == 0) {
            hw_found = true;
            fprintf(stderr, "  NPU: AMD NPU driver loaded\n");
        }
        if (!hw_found) { fprintf(stderr, "  NPU: no XDNA 2 hardware detected\n"); return false; }
        if (access(engine_bin_.c_str(), X_OK) != 0) {
            fprintf(stderr, "  NPU: engine binary not found at %s\n", engine_bin_.c_str());
            fprintf(stderr, "  NPU: (build with: engine/npu/build_npu.sh)\n");
            return false;
        }
        available_ = true;
        return true;
    }

    bool load_model(const ModelConfig& cfg) override {
        cfg_ = cfg;
        unload_model();
        if (!is_available()) return false;

        model_variant_ = "qwen3_0_6b";
        if (cfg.model_name.find("0.6B") != std::string::npos || cfg.hidden_size <= 1536)
            model_variant_ = "qwen3_0_6b";
        else if (cfg.model_name.find("8B") != std::string::npos || cfg.hidden_size >= 4096)
            model_variant_ = "qwen3_8b";
        else if (cfg.model_name.find("4B") != std::string::npos || cfg.model_name.find("VL") != std::string::npos)
            model_variant_ = "qwen3_vl_4b";
        else if (cfg.model_name.find("Llama") != std::string::npos || cfg.model_name.find("llama") != std::string::npos)
            model_variant_ = "llama";
        else if (cfg.model_name.find("Gemma") != std::string::npos || cfg.model_name.find("gemma") != std::string::npos)
            model_variant_ = "gemma4_e2b";

        std::string model_bin = "engine/npu/build/npu_engine_" + model_variant_;
        if (access(model_bin.c_str(), X_OK) == 0) engine_bin_ = model_bin;
        else if (access(engine_bin_.c_str(), X_OK) != 0) {
            fprintf(stderr, "  NPU: no engine binary for variant %s\n", model_variant_.c_str());
            return false;
        }

        fprintf(stderr, "  NPU: starting engine (variant=%s, bin=%s)...\n", model_variant_.c_str(), engine_bin_.c_str());

        int to_stdin[2], from_stdout[2];
        if (pipe(to_stdin) != 0 || pipe(from_stdout) != 0) return false;

        engine_pid_ = fork();
        if (engine_pid_ == 0) {
            dup2(to_stdin[0], STDIN_FILENO);
            dup2(from_stdout[1], STDOUT_FILENO);
            close(to_stdin[0]); close(to_stdin[1]);
            close(from_stdout[0]); close(from_stdout[1]);
            execl(engine_bin_.c_str(), engine_bin_.c_str(), nullptr);
            _exit(1);
        }
        close(to_stdin[0]); close(from_stdout[1]);
        stdin_fd_ = to_stdin[1];
        stdout_fd_ = from_stdout[0];

        fprintf(stderr, "  NPU: waiting for engine init (pid=%d)...\n", engine_pid_);
        usleep(3000000);

        int status;
        if (waitpid(engine_pid_, &status, WNOHANG) == engine_pid_) {
            fprintf(stderr, "  NPU: engine exited prematurely (status=%d)\n", WEXITSTATUS(status));
            engine_pid_ = 0;
            close(stdin_fd_); close(stdout_fd_);
            return false;
        }

        loaded_ = true;
        fprintf(stderr, "  NPU: engine ready (model=%s)\n", model_variant_.c_str());
        return true;
    }

    void unload_model() override {
        if (engine_pid_ > 0) {
            close(stdin_fd_); close(stdout_fd_);
            kill(engine_pid_, SIGTERM); usleep(200000);
            kill(engine_pid_, SIGKILL); waitpid(engine_pid_, nullptr, WNOHANG);
            engine_pid_ = 0;
        }
        stdin_fd_ = -1; stdout_fd_ = -1;
        loaded_ = false;
    }

    void reset_state() override { pending_tokens_.clear(); }

    int forward(int token_id, int pos) override {
        pending_tokens_.push_back(token_id);
        static std::vector<int> cached_tokens_;
        static size_t cache_pos_ = 0;

        if (pos == 0 || cache_pos_ >= cached_tokens_.size()) {
            if (pending_tokens_.empty()) return 106;

            std::string req = "{\"tokens\":[";
            for (size_t i = 0; i < pending_tokens_.size(); i++) {
                if (i) req += ",";
                req += std::to_string(pending_tokens_[i]);
            }
            req += "],\"max_new_tokens\":16}\n";

            ssize_t wrote = write(stdin_fd_, req.data(), req.size());
            if (wrote != (ssize_t)req.size()) return 106;

            std::string resp;
            char buf[65536];
            auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t0).count() < timeout_ms_) {
                fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd_, &fds);
                struct timeval tv = {1, 0};
                int r = select(stdout_fd_ + 1, &fds, nullptr, nullptr, &tv);
                if (r > 0) {
                    ssize_t n = read(stdout_fd_, buf, sizeof(buf) - 1);
                    if (n > 0) { buf[n] = 0; resp += buf;
                        if (resp.find('\n') != std::string::npos) { resp = resp.substr(0, resp.find('\n')); break; }
                    } else break;
                }
            }

            cached_tokens_.clear();
            if (!resp.empty()) {
                auto tp = resp.find("\"tokens\"");
                if (tp != std::string::npos) {
                    auto ap = resp.find('[', tp);
                    if (ap != std::string::npos) {
                        auto ae = resp.find(']', ap);
                        if (ae != std::string::npos) {
                            std::string arr = resp.substr(ap + 1, ae - ap - 1);
                            size_t i = 0;
                            while (i < arr.size()) {
                                while (i < arr.size() && (arr[i] == ',' || arr[i] == ' ')) i++;
                                if (i >= arr.size()) break;
                                char* end;
                                cached_tokens_.push_back((int)strtol(arr.c_str() + i, &end, 10));
                                i = end - arr.c_str();
                            }
                        }
                    }
                }
            }
            cache_pos_ = 0;
            pending_tokens_.clear();
            for (int t : cached_tokens_) pending_tokens_.push_back(t);
        }

        if (cache_pos_ < cached_tokens_.size()) return cached_tokens_[cache_pos_++];
        return 106;
    }
};

std::vector<InferenceBackend*> detect_backends_npu() {
    std::vector<InferenceBackend*> backends;
    static NpuBackend npu;
    backends.push_back(&npu);
    return backends;
}
