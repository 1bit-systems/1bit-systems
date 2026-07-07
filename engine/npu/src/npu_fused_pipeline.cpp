/** NPU Fused Pipeline — QKV→Attention→FFN via split/fused engine.
 *  Demonstrates the fused inference loop using npu_engine_split or npu_engine_fused_server.
 *  
 *  Usage: npu_fused_pipeline [--engine split|server|fused] [--batch N] [--tokens N]
 *
 *  This is the C++ equivalent of fused_bench.py — measures the full
 *  QKV→Attention→FFN pipeline throughput and estimates the pipeline-
 *  overlapped tok/s target.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/wait.h>

// ─── Model config (Qwen3-0.6B) ──────────────────────────────────────────
static constexpr int H = 1536;
static constexpr int NH = 12;
static constexpr int NKV = 2;
static constexpr int HD = 128;
static constexpr int IM = 4096;
static constexpr int NC = 28;
static constexpr int QKV = NH * HD + 2 * NKV * HD;  // 12*128 + 2*2*128 = 1536+512=2048
static constexpr int NV = 151936;
static constexpr int MAX_BATCH = 128;

static const char* kModelPath = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
static const char* kEngineDir = "/home/bcloud/engine/npu/build";
static const char* kXclbinDir = "/home/bcloud/npu-sandbox/npu-infer/build/int8";

// ─── NPU Server client ──────────────────────────────────────────────────

struct NpuClient {
    pid_t pid;
    int stdin_fd;
    int stdout_fd;
    bool ready;
    
    bool spawn(const char* engine_type) {
        int in_pipe[2], out_pipe[2];
        if (pipe(in_pipe) < 0 || pipe(out_pipe) < 0) { perror("pipe"); return false; }
        
        pid = fork();
        if (pid < 0) { perror("fork"); return false; }
        
        if (pid == 0) {
            // Child: connect pipes to stdin/stdout
            dup2(in_pipe[0], STDIN_FILENO);
            dup2(out_pipe[1], STDOUT_FILENO);
            close(in_pipe[1]); close(out_pipe[0]);
            close(in_pipe[0]); close(out_pipe[1]);
            
            std::string engine_path = std::string(kEngineDir) + "/npu_engine_";
            std::vector<const char*> args;
            
            if (strcmp(engine_type, "split") == 0) {
                engine_path += "split";
                args = { engine_path.c_str(), kModelPath, "--xclbin-dir", kXclbinDir, nullptr };
            } else if (strcmp(engine_type, "fused") == 0) {
                engine_path += "fused_server";
                args = { engine_path.c_str(), nullptr };
            } else {
                engine_path += "server";
                args = { engine_path.c_str(), kModelPath, "--server", nullptr };
            }
            
            execvp(args[0], (char* const*)args.data());
            fprintf(stderr, "exec failed: %s\n", engine_path.c_str());
            _exit(1);
        }
        
        // Parent
        stdin_fd = in_pipe[1];
        stdout_fd = out_pipe[0];
        close(in_pipe[0]); close(out_pipe[1]);
        
        // Wait for "READY" on stdout
        char buf[4096];
        std::string accum;
        auto t0 = std::chrono::steady_clock::now();
        while (true) {
            fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd, &fds);
            struct timeval tv = { 1, 0 };
            int ret = select(stdout_fd + 1, &fds, nullptr, nullptr, &tv);
            if (ret <= 0) {
                fprintf(stderr, "Timeout waiting for server ready\n");
                kill(pid, SIGTERM); waitpid(pid, nullptr, 0);
                return false;
            }
            int n = (int)read(stdout_fd, buf, sizeof(buf) - 1);
            if (n <= 0) break;
            buf[n] = 0;
            accum += buf;
            if (accum.find("READY") != std::string::npos) {
                ready = true;
                break;
            }
        }
        return ready;
    }
    
    ~NpuClient() {
        if (pid > 0) {
            write(stdin_fd, "EXIT\n", 5);
            close(stdin_fd);
            close(stdout_fd);
            waitpid(pid, nullptr, WNOHANG);
        }
    }
    
    void send_cmd(const char* fmt, ...) {
        char cmd[512];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(cmd, sizeof(cmd), fmt, ap);
        va_end(ap);
        write(stdin_fd, cmd, strlen(cmd));
    }
    
    void send_binary(const float* data, int count) {
        write(stdin_fd, data, count * 4);
    }
    
    void recv_float(float* out, int count) {
        int bytes = count * 4;
        int pos = 0;
        while (pos < bytes) {
            int n = (int)read(stdout_fd, (char*)out + pos, bytes - pos);
            if (n <= 0) { fprintf(stderr, "Short read: %d/%d\n", pos, bytes); break; }
            pos += n;
        }
    }
    
    void recv_int32(int32_t* out, int count) {
        recv_float((float*)out, count); // same byte count
    }
    
    void qkv(int layer, int pos, int batch, const float* hidden, float* qkv_out) {
        char cmd[128];
        snprintf(cmd, sizeof(cmd), "QKV %d %d %d\n", layer, pos, batch);
        send_cmd(cmd);
        send_binary(hidden, batch * H);
        recv_float(qkv_out, batch * QKV);
    }
    
    void ffn(int layer, int pos, int batch, const float* attn_in, float* hidden_out) {
        char cmd[128];
        snprintf(cmd, sizeof(cmd), "FFN %d %d %d\n", layer, pos, batch);
        send_cmd(cmd);
        send_binary(attn_in, batch * NH * HD);
        recv_float(hidden_out, batch * H);
    }
    
    void attention_cpu(int layer, int pos, int batch, const float* qkv_in, float* attn_out) {
        char cmd[128];
        snprintf(cmd, sizeof(cmd), "ATTENTION %d %d %d\n", layer, pos, batch);
        send_cmd(cmd);
        send_binary(qkv_in, batch * QKV);
        recv_float(attn_out, batch * NH * HD);
    }
    
    void lm_head(int batch, const float* hidden, int32_t* token_ids) {
        char cmd[128];
        snprintf(cmd, sizeof(cmd), "LM_HEAD %d\n", batch);
        send_cmd(cmd);
        send_binary(hidden, batch * H);
        recv_int32(token_ids, batch);
    }
    
    // Fused server: single LAYER call does QKV→Attn→O→GU→D
    void fused_layer(int layer, int pos, const float* hidden, float* hidden_out) {
        char cmd[128];
        snprintf(cmd, sizeof(cmd), "LAYER %d %d 1\n", layer, pos);
        send_cmd(cmd);
        // Fused server expects BF16 input
        std::vector<uint16_t> bf16_in(H);
        for (int i = 0; i < H; i++) {
            uint32_t b; memcpy(&b, &hidden[i], 4);
            bf16_in[i] = (uint16_t)((b + 0x8000) >> 16);
        }
        write(stdin_fd, bf16_in.data(), H * 2);
        // Fused server outputs BF16
        std::vector<uint16_t> bf16_out(H);
        int pos_bytes = 0, total = H * 2;
        while (pos_bytes < total) {
            int n = (int)read(stdout_fd, (char*)bf16_out.data() + pos_bytes, total - pos_bytes);
            if (n <= 0) { fprintf(stderr, "Short read: %d/%d\n", pos_bytes, total); break; }
            pos_bytes += n;
        }
        for (int i = 0; i < H; i++) {
            uint32_t b = (uint32_t)bf16_out[i] << 16;
            memcpy(&hidden_out[i], &b, 4);
        }
    }
};

// ─── CPU Attention reference ────────────────────────────────────────────

static void attention_cpu_ref(const float* q, const float* k_cache, const float* v_cache,
                              float* out, int NH, int NKV, int HD, int GQA, int seq_len, int max_pos) {
    // Simplified single-query attention
    for (int hh = 0; hh < NH; hh++) {
        int kvh = hh / GQA;
        double max_score = -1e30;
        std::vector<double> scores(seq_len);
        for (int p = 0; p < seq_len; p++) {
            if (p >= max_pos) { scores[p] = -1e30; continue; }
            double s = 0;
            for (int d = 0; d < HD; d++)
                s += (double)q[hh * HD + d] * k_cache[p * NKV * HD + kvh * HD + d];
            s /= sqrt((double)HD);
            scores[p] = s;
            if (s > max_score) max_score = s;
        }
        double sum = 0;
        for (int p = 0; p < seq_len; p++) { scores[p] = exp(scores[p] - max_score); sum += scores[p]; }
        double inv_sum = sum > 0 ? 1.0 / sum : 1.0 / seq_len;
        for (int d = 0; d < HD; d++) {
            double acc = 0;
            for (int p = 0; p < seq_len; p++)
                acc += scores[p] * v_cache[p * NKV * HD + kvh * HD + d];
            out[hh * HD + d] = (float)(acc * inv_sum);
        }
    }
}

// ─── Benchmarks ─────────────────────────────────────────────────────────

static double bench_qkv(NpuClient& npu, int batch) {
    std::vector<float> hidden(batch * H);
    std::vector<float> qkv_buf(batch * QKV);
    for (int i = 0; i < batch * H; i++) hidden[i] = 0.01f * (float)(rand() % 100);
    
    auto t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < NC; l++)
        npu.qkv(l, 0, batch, hidden.data(), qkv_buf.data());
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / NC;
    double tok_s = batch / (ms / 1000.0);
    printf("  QKV B=%d: %.2fms/layer → %.0f tok/s\n", batch, ms, tok_s);
    return ms;
}

static double bench_ffn(NpuClient& npu, int batch) {
    std::vector<float> attn_in(batch * NH * HD);
    std::vector<float> hidden(batch * H);
    for (int i = 0; i < batch * NH * HD; i++) attn_in[i] = 0.01f * (float)(rand() % 100);
    
    auto t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < NC; l++)
        npu.ffn(l, 0, batch, attn_in.data(), hidden.data());
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / NC;
    double tok_s = batch / (ms / 1000.0);
    printf("  FFN B=%d: %.2fms/layer → %.0f tok/s\n", batch, ms, tok_s);
    return ms;
}

static void bench_pipeline(NpuClient& npu, int batch) {
    printf("\n── Fused pipeline QKV→Attention→FFN (B=%d) ──\n", batch);
    std::vector<float> hidden(batch * H);
    std::vector<float> qkv_buf(batch * QKV);
    std::vector<float> attn_buf(batch * NH * HD);
    std::vector<float> result(batch * H);
    for (int i = 0; i < batch * H; i++) hidden[i] = 0.01f * (float)(rand() % 100);
    
    std::vector<double> qkv_ms(NC), attn_ms(NC), ffn_ms(NC);
    
    auto t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < NC; l++) {
        // Phase 1: QKV
        auto t1 = std::chrono::steady_clock::now();
        npu.qkv(l, l, batch, hidden.data(), qkv_buf.data());
        qkv_ms[l] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        
        // Phase 2: CPU Attention (stand-in for GPU flash attention)
        auto t2 = std::chrono::steady_clock::now();
        npu.attention_cpu(l, l, batch, qkv_buf.data(), attn_buf.data());
        attn_ms[l] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
        
        // Phase 3: FFN
        auto t3 = std::chrono::steady_clock::now();
        npu.ffn(l, l, batch, attn_buf.data(), result.data());
        ffn_ms[l] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t3).count();
        
        hidden = result;
    }
    double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    
    // Averages
    double avg_qkv = 0, avg_attn = 0, avg_ffn = 0;
    for (int l = 0; l < NC; l++) { avg_qkv += qkv_ms[l]; avg_attn += attn_ms[l]; avg_ffn += ffn_ms[l]; }
    avg_qkv /= NC; avg_attn /= NC; avg_ffn /= NC;
    
    printf("  QKV:       %.2fms/layer (%.1fms total)\n", avg_qkv, avg_qkv * NC);
    printf("  Attention: %.2fms/layer (%.1fms total)\n", avg_attn, avg_attn * NC);
    printf("  FFN:       %.2fms/layer (%.1fms total)\n", avg_ffn, avg_ffn * NC);
    printf("  Total:     %.1fms → %.0f tok/s (sequential)\n", total_ms, batch / (total_ms / 1000.0));
    
    // Pipeline overlap: QKV(N+1) || Attn(N) → FFN(N)
    double pipe_crit = std::max({avg_qkv, avg_attn, avg_ffn});
    double pipe_ms = avg_qkv + (NC - 1) * std::max({avg_attn, avg_ffn, avg_qkv}) + std::max(avg_attn, avg_ffn);
    double pipe_tok_s = batch / (pipe_ms / 1000.0);
    
    printf("\n── Pipeline overlapped ──\n");
    printf("  Critical path: %.2fms/layer\n", pipe_crit);
    printf("  Pipeline: %.1fms → %.0f tok/s\n", pipe_ms, pipe_tok_s);
    printf("  Target 273 tok/s gap: %.0f (%.0f%%)\n", 273 - pipe_tok_s, (273 / pipe_tok_s - 1) * 100);
}

static void bench_fused_layer(NpuClient& npu) {
    printf("\n── Fused single-layer pipeline ──\n");
    std::vector<float> hidden(H);
    std::vector<float> result(H);
    for (int i = 0; i < H; i++) hidden[i] = 0.01f * (float)(rand() % 100);
    
    auto t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < NC; l++)
        npu.fused_layer(l, l, hidden.data(), result.data());
    double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    double per_layer = total_ms / NC;
    double tok_s = 1000.0 / per_layer;  // batch=1, one token per full-layer call
    printf("  %.2fms/layer → %.0f tok/s (fused, B=1)\n", per_layer, tok_s);
    printf("  With pipeline overlap: ~%.0f tok/s\n", tok_s * 1.3);  // ~30% overlap gain
}

// ─── Main ───────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* engine_type = "server";
    int batch = 64;
    int n_tokens = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--engine") == 0 && i + 1 < argc) engine_type = argv[++i];
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) n_tokens = atoi(argv[++i]);
    }
    
    if (batch > MAX_BATCH) batch = MAX_BATCH;
    if (batch < 1) batch = 1;
    bool is_fused = (strcmp(engine_type, "fused") == 0);
    
    printf("═══ NPU Fused Pipeline ═══\n");
    printf("Engine: %s | B=%d | L=%d | H=%d\n", engine_type, batch, NC, H);
    printf("Model: %s\n", kModelPath);
    printf("\nStarting NPU server...\n");
    
    NpuClient npu;
    if (!npu.spawn(engine_type)) {
        fprintf(stderr, "Failed to start NPU engine\n");
        return 1;
    }
    printf("  Server ready\n");
    
    if (is_fused) {
        // Fused engine: single LAYER call per layer
        printf("\n── Warmup ──\n");
        std::vector<float> h0(H);
        npu.fused_layer(0, 0, h0.data(), h0.data());
        printf("  OK\n");
        bench_fused_layer(npu);
    } else {
        // Warmup
        printf("\n── Warmup ──\n");
        std::vector<float> h0(H);
        std::vector<float> q0(QKV);
        std::vector<float> a0(NH * HD);
        npu.qkv(0, 0, 1, h0.data(), q0.data());
        npu.ffn(0, 0, 1, a0.data(), h0.data());
        printf("  OK\n");
        
        // Micro-benchmarks
        printf("\n── Micro-benchmarks ──\n");
        double qkv_ms = bench_qkv(npu, batch);
        double ffn_ms = bench_ffn(npu, batch);
        printf("  Q+F:  %.2fms/layer → %.0f tok/s (NPU-only estimate)\n",
            qkv_ms + ffn_ms, batch / ((qkv_ms + ffn_ms) * NC / 1000.0));
        
        // Full pipeline
        bench_pipeline(npu, batch);
        
        // Summary
        printf("\n═══ Summary (B=%d) ═══\n", batch);
        printf("  NPU QKV + FFN:  %.2f + %.2f = %.2f ms/layer\n", qkv_ms, ffn_ms, qkv_ms + ffn_ms);
        printf("  Seq tok/s:      %.0f\n", batch / ((qkv_ms + ffn_ms + 0.5) * NC / 1000.0));
        printf("  Pipe tok/s:     ~%.0f (with GPU overlap)\n",
            batch / ((qkv_ms + (NC-1) * std::max(0.5, ffn_ms) + std::max(0.5, ffn_ms)) / 1000.0));
    }
    printf("  Target:         273 tok/s\n");
    
    return 0;
}
