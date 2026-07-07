#pragma once
/** FusedServerTarget — TargetModelInterface wrapping npu_engine_fused_server subprocess.
 *
 * Spawns the proven fused server binary and communicates via stdin/stdout pipe protocol.
 * Loads the model file locally for embedding lookups and lm_head computation.
 *
 * Protocol:
 *   LAYER <layer> <pos> <batch>  (reads BF16 hidden, writes BF16 output, batch=1 only)
 *   LM_HEAD <batch>               (reads F32 hidden, writes I32 token IDs, batch=1 only)
 *   EXIT
 *
 * Build: link with -lxrt_coreutil -luuid -lm -ldl
 */

#include "spec_decode.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

// BF16 helpers
static inline float bf16_to_f32(uint16_t v) {
    if (v == 0) return 0.0f;
    uint32_t b = (uint32_t)v << 16; float f;
    memcpy(&f, &b, 4); return f;
}
static inline uint16_t f32_to_bf16(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    return (uint16_t)((b + 0x8000) >> 16);
}

#include "npu_fused_target.h"  // reuse Q4NXModel

static const char* kFusedServerPath_ = "/home/bcloud/engine/npu/build/npu_engine_fused_server";

class FusedServerTarget : public TargetModelInterface {
public:
    static constexpr int H = 1024;
    static constexpr int NC = 28;
    static constexpr int NV = 151936;
    static constexpr int B = H * 2;  // bytes per bf16 hidden block

    FusedServerTarget(const char* model_path,
                      const char* /*xclbin_dir*/,
                      const char* /*weights_dir*/,
                      const int32_t* target_layer_ids,
                      int32_t num_target_layers)
        : target_layer_ids_(target_layer_ids, target_layer_ids + num_target_layers) {
        
        // Load model for embedding lookups
        if (!model_.load(model_path)) {
            fprintf(stderr, "[FusedServerTarget] Failed to load model\n");
            return;
        }
        
        // Spawn fused server
        int in_pipe[2], out_pipe[2];
        if (pipe(in_pipe) < 0 || pipe(out_pipe) < 0) { perror("pipe"); return; }
        
        pid_ = fork();
        if (pid_ < 0) { perror("fork"); return; }
        
        if (pid_ == 0) {
            dup2(in_pipe[0], STDIN_FILENO);
            dup2(out_pipe[1], STDOUT_FILENO);
            close(in_pipe[1]); close(out_pipe[0]);
            close(in_pipe[0]); close(out_pipe[1]);
            
            // The server reads FUSED_MODEL, FUSED_XCLBIN_DIR, FUSED_WEIGHTS_DIR from env
            // Those are set globally, so they should work for the child process too.
            execlp(kFusedServerPath_, kFusedServerPath_, nullptr);
            fprintf(stderr, "[FusedServerTarget] exec failed: %s\n", kFusedServerPath_);
            _exit(1);
        }
        
        stdin_fd_ = in_pipe[1];
        stdout_fd_ = out_pipe[0];
        close(in_pipe[0]); close(out_pipe[1]);
        
        // Wait for "READY"
        char buf[4096];
        std::string accum;
        auto t0 = std::chrono::steady_clock::now();
        while (true) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd_, &fds);
            struct timeval tv = { 10, 0 };
            int ret = select(stdout_fd_ + 1, &fds, nullptr, nullptr, &tv);
            if (ret <= 0) {
                fprintf(stderr, "[FusedServerTarget] timeout\n");
                cleanup(); return;
            }
            int n = (int)read(stdout_fd_, buf, sizeof(buf) - 1);
            if (n <= 0) break;
            buf[n] = 0;
            accum += buf;
            if (accum.find("READY") != std::string::npos) {
                printf("[FusedServerTarget] ready\n");
                ready_ = true;
                break;
            }
        }
        
        layer_hidden_snapshots_.resize(NC, std::vector<float>(H));
    }

    ~FusedServerTarget() override { cleanup(); }

    bool ready() const { return ready_; }

    // ── TargetModelInterface ──

    void forward(const int32_t* input_ids, int32_t seq_len,
                 float* logits, float* hidden_states) override {
        kv_pos_ = 0;
        run_positions(input_ids, seq_len, logits, hidden_states, /*logits_all=*/false);
    }

    void forward_with_kv(const int32_t* input_ids, int32_t n_tokens, int32_t /*past_len*/,
                          float* logits, float* hidden_states) override {
        run_positions(input_ids, n_tokens, logits, hidden_states, /*logits_all=*/true);
    }

    void get_layer_hidden(const float* /*all_hidden*/, int32_t /*num_layers*/,
                           const int32_t* target_ids, int32_t num_target,
                           float* out) override {
        for (int i = 0; i < num_target; i++) {
            int layer = target_ids[i];
            memcpy(out + (size_t)i * H, layer_hidden_snapshots_[layer].data(), H * 4);
        }
    }

    void commit_accepted(int32_t /*start_pos*/, int32_t /*n_accept*/) override {}

private:
    void cleanup() {
        if (pid_ > 0) {
            write(stdin_fd_, "EXIT\n", 5);
            close(stdin_fd_);
            close(stdout_fd_);
            waitpid(pid_, nullptr, WNOHANG);
            pid_ = 0;
        }
    }

    void send_all(const void* data, size_t nbytes) {
        const char* p = (const char*)data;
        size_t remain = nbytes;
        while (remain > 0) {
            int n = (int)write(stdin_fd_, p, remain);
            if (n <= 0) { fprintf(stderr, "[FusedServerTarget] write error\n"); break; }
            p += n; remain -= n;
        }
    }

    void recv_all(void* data, size_t nbytes) {
        char* p = (char*)data;
        size_t remain = nbytes;
        while (remain > 0) {
            int n = (int)read(stdout_fd_, p, remain);
            if (n <= 0) { fprintf(stderr, "[FusedServerTarget] read error\n"); break; }
            p += n; remain -= n;
        }
    }

    // Run one or more positions through the fused server
    void run_positions(const int32_t* tokens, int n,
                        float* out_logits, float* out_hidden,
                        bool logits_all) {
        if (!ready_) return;

        for (int pi = 0; pi < n; pi++) {
            int pos = kv_pos_ + pi;
            
            // Get BF16 embedding for this token
            model_.embed_lookup_bf16(tokens[pi], bf16_embed_);
            
            // Run all 28 layers — read output after EACH layer for draft feature snapshots
            for (int l = 0; l < NC; l++) {
                // Send LAYER command
                char cmd[64];
                snprintf(cmd, sizeof(cmd), "LAYER %d %d 1\n", l, pos);
                send_all(cmd, strlen(cmd));
                send_all(bf16_embed_, B);  // input hidden state (bf16)
                
                // Read output (bf16) — becomes next layer's input
                recv_all(bf16_embed_, B);
                
                // Snapshot at target layer positions for draft features
                for (int ti = 0; ti < (int)target_layer_ids_.size(); ti++) {
                    if (target_layer_ids_[ti] == l) {
                        for (int i = 0; i < H; i++)
                            layer_hidden_snapshots_[l][i] = bf16_to_f32(bf16_embed_[i]);
                        break;
                    }
                }
            }
            
            // If we need per-position logits
            if (logits_all && out_logits) {
                float* lp = out_logits + (size_t)pi * NV;
                compute_logits(bf16_embed_, lp);
            }
        }
        
        kv_pos_ += n;
        
        // Last position's hidden state
        if (out_hidden) {
            for (int i = 0; i < H; i++)
                out_hidden[i] = bf16_to_f32(bf16_embed_[i]);
        }
        
        // Compute logits for last position (if not already done per-position)
        if (out_logits && !logits_all) {
            compute_logits(bf16_embed_, out_logits);
        }
    }

    void compute_logits(const uint16_t* bf16_hidden, float* logits) {
        if (!logits) return;
        
        // Convert BF16 to float
        std::vector<float> h(H);
        for (int i = 0; i < H; i++) h[i] = bf16_to_f32(bf16_hidden[i]);
        
        // RMSNorm
        double ss = 0;
        for (int i = 0; i < H; i++) if (std::isfinite(h[i])) ss += (double)h[i] * h[i];
        float ir = 1.0f / sqrtf((float)(ss / H) + 1e-6f);
        for (int i = 0; i < H; i++) {
            if (!std::isfinite(h[i])) h[i] = 0.0f;
            else h[i] *= ir * model_.final_norm_w[i];
        }
        
        // LM head matmul (vocab-parallel)
        #pragma omp parallel for
        for (int v = 0; v < NV; v++) {
            double s = 0;
            const float* wrow = model_.lm_head_f32 + (size_t)v * H;
            for (int k = 0; k < H; k++) s += (double)h[k] * wrow[k];
            logits[v] = (float)s;
        }
    }

    npu_fused_detail::Q4NXModel model_;
    pid_t pid_ = 0;
    int stdin_fd_ = -1, stdout_fd_ = -1;
    bool ready_ = false;
    int kv_pos_ = 0;
    std::vector<std::vector<float>> layer_hidden_snapshots_;
    std::vector<int32_t> target_layer_ids_;
    // Buffer for BF16 data (reused across calls)
    uint16_t bf16_embed_[1024];
};
