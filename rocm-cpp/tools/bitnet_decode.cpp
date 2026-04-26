// bitnet_decode — minimal end-to-end BitNet-2B-4T forward pass using only
// librocm_cpp. No MLX, no halo-1bit, no external ML framework.
//
// Usage: bitnet_decode <model.h1b> [start_token_id=1] [num_new_tokens=16]
//
// Greedy (argmax) decode. Prints generated token IDs + per-token latency.
// Doesn't include a tokenizer — caller provides token IDs directly.
//
// Pipeline matches BitNet-b1.58:
//   input_norm → QKV proj → RoPE → attention → attn_sub_norm → O proj
//   → residual → post_attn_norm → gate/up proj → fused relu² GLU + ffn_sub_norm
//   → down proj → residual → final_norm → tied LM head → argmax
//
// Residual stream is FP32 throughout. The raw relu²(gate)*up intermediate
// reaches ~1e9 on real weights, so the ReLU² GLU is fused with its
// ffn_sub_norm inside the kernel (FP32 internal, FP16 output). The PyTorch
// reference (absmean quant, 1/mean(|W|) scale, ReLU² GLU, sub_norms) gives
// the exact same top-5 tokens for every input we've checked.

#include "rocm_cpp/ck_gemm.h"
#include "rocm_cpp/bitnet_model.h"
#include "rocm_cpp/sherry.h"
#include "rocm_cpp/tokenizer.h"
#include "rocm_cpp/kv_rotorquant.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <ctime>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

// Bonsai Q1_0_g128 / TQ2_0_g128 ternary GEMV launchers — forward-declared
// here (the HIP port agent's rocm_cpp/bonsai.h hasn't landed yet).
// `__attribute__((weak))` lets us link even when the kernels haven't been
// built into librocm_cpp.so yet — the Bonsai dispatch path checks the
// symbol pointer for null and reports a clean error instead of SEGV.
extern "C" void bonsai_q1_gemv_launch(
    const uint8_t* packed_weights,
    const uint16_t* act_fp16,
    uint16_t* out_fp16,
    int N_out, int K_in,
    void* stream) __attribute__((weak));
extern "C" void bonsai_tq2_gemv_launch(
    const uint8_t* packed_weights,
    const uint16_t* act_fp16,
    uint16_t* out_fp16,
    int N_out, int K_in,
    void* stream) __attribute__((weak));

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP %d %s:%d\n",_s,__FILE__,__LINE__); return 1;}} while(0)
#define RC_OK(e)  do { auto _s=(e); if(_s!=RCPP_OK){fprintf(stderr,"rcpp err %d at %s:%d\n",(int)_s,__FILE__,__LINE__); return 1;}} while(0)

namespace {

// Post-scale for the fp16-Sherry GEMV path.
//
// The clean-room sherry_ternary_gemv_launch kernel emits pure signed-sum fp16
// — it does NOT fold the per-row fp32 scale that lives on the .h1b layer. The
// halo-1bit decoder does fold that scale in (it's INT8-input with per-row
// scale baked into the accumulator). Here we apply it as a tiny post-pass so
// both paths produce comparable magnitudes.
//
// out[n] := clamp_fp16(fp32(out[n]) * row_scales[n]).
//
// One thread per row; N is always small (hidden_size, intermediate_size, or
// head-count * head-dim — all ≤ 6912 on the 2B model). A single wave-round
// block is plenty; no reduction needed.
__global__ void sherry_fp16_apply_row_scale_kernel(
    __half*          __restrict__ y,           // [N]  inout
    const float*     __restrict__ row_scales,  // [N]  in
    int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float v = (float)y[n] * row_scales[n];
    // Match the in-kernel clamp in src/sherry_gemv.hip — downstream RMSNorm
    // / softmax NaN-poisons on ±Inf.
    constexpr float FP16_MAX = 65504.0f;
    float vc = v;
    if (vc >  FP16_MAX) vc =  FP16_MAX;
    if (vc < -FP16_MAX) vc = -FP16_MAX;
    y[n] = __float2half(vc);
}

// Dispatch helper: fp16 activations + packed Sherry weights -> fp16 output,
// with per-row fp32 scale applied as a fused post-pass. Signature mirrors the
// integer-input path so bitnet_decode's forward lambda can use a single call
// site per projection.
//
// Returns RCPP_OK unconditionally — sherry_ternary_gemv_launch early-returns
// silently on malformed (N, K). The constraint K % 32 == 0 is validated at
// model-load time (read_ternary_sherry) and matches BitNet-2B shapes
// (hs=2560, is=6912).
rcpp_status_t bitnet_sherry_fp16_gemv(
    const void* packed_weights_dev,
    const void* act_fp16_dev,
    const void* row_scales_dev,
    void*       out_fp16_dev,
    int N, int K,
    hipStream_t stream)
{
    sherry_ternary_gemv_launch(
        static_cast<const uint8_t*>(packed_weights_dev),
        static_cast<const uint16_t*>(act_fp16_dev),
        static_cast<uint16_t*>(out_fp16_dev),
        N, K,
        static_cast<void*>(stream));

    const int BLOCK = 128;
    dim3 grid((unsigned)((N + BLOCK - 1) / BLOCK), 1, 1);
    dim3 block((unsigned)BLOCK, 1, 1);
    hipLaunchKernelGGL(sherry_fp16_apply_row_scale_kernel,
                       grid, block, 0, stream,
                       static_cast<__half*>(out_fp16_dev),
                       static_cast<const float*>(row_scales_dev),
                       N);
    return RCPP_OK;
}

// Bonsai Q1 / TQ2 GEMV dispatch. fp16-in / fp16-out, no row_scales tensor
// (scales are inline per-128-weight block). When the kernel is not linked
// in (weak symbol resolved to null), memset the output to zero and warn.
rcpp_status_t bonsai_gemv_dispatch(
    rcpp_weight_format_t fmt,
    const void* packed_weights_dev,
    const void* act_fp16_dev,
    void*       out_fp16_dev,
    int N, int K,
    hipStream_t stream)
{
    using Fn = void (*)(const uint8_t*, const uint16_t*, uint16_t*,
                        int, int, void*);
    Fn fn = nullptr;
    if      (fmt == RCPP_WEIGHT_FORMAT_BONSAI_TQ2) fn = &bonsai_tq2_gemv_launch;
    else if (fmt == RCPP_WEIGHT_FORMAT_BONSAI_Q1)  fn = &bonsai_q1_gemv_launch;
    if (!fn) {
        static bool once = false;
        if (!once) {
            fprintf(stderr,
                "[bitnet_decode] bonsai_*_gemv_launch not linked — HIP kernel pass has not landed yet.\n"
                "  Zeroing output; logits will be meaningless until librocm_cpp.so re-exports the symbol.\n");
            once = true;
        }
        (void)hipMemsetAsync(out_fp16_dev, 0, (size_t)N * sizeof(uint16_t), stream);
        return RCPP_OK;
    }
    fn(static_cast<const uint8_t*>(packed_weights_dev),
       static_cast<const uint16_t*>(act_fp16_dev),
       static_cast<uint16_t*>(out_fp16_dev),
       N, K,
       static_cast<void*>(stream));
    return RCPP_OK;
}

}  // namespace

// Load a whitespace-separated list of token IDs from a stream.
// Used when --prompt arg starts with @ (file path) or "-" (stdin).
static std::vector<int> read_token_ids(std::istream& in) {
    std::vector<int> ids;
    int t;
    while (in >> t) ids.push_back(t);
    return ids;
}

int main(int argc, char** argv) {
    // CLI:
    //   bitnet_decode <model.h1b> <prompt> <num_new_tokens> [tokenizer.htok]
    //   bitnet_decode <model.h1b>                                 # defaults
    //   bitnet_decode --model <model.h1b> --ctx <N> --iters <N>   # bench mode
    //
    // <prompt> forms:
    //   --text "your prompt"      — encode via librocm_cpp tokenizer (.htok)
    //   @file.toks                — whitespace-separated ints from file
    //   -                         — whitespace-separated ints from stdin
    //   <int>                     — single start_tok (legacy)
    //
    // Bench mode (--model + --ctx + --iters): runs --iters new tokens after
    // a ctx-sized warmup prefill, prints ONE LAST LINE of "<tok/s>" to stdout.
    // sherry-bench.sh parses this; no other output goes to stdout in this mode.

    // Peek for --model / --ctx / --iters / --prompt / --max-tokens up front;
    // if present, drive the positional/prompt machinery from those flags.
    // --prompt / --max-tokens are flag-form aliases for the positional
    // (<prompt> <num_new>) triple and let orchestration scripts use a
    // uniform `--model X --prompt Y --max-tokens N` invocation across
    // BitNet / Sherry / Bonsai models.
    const char* bench_model = nullptr;
    int bench_ctx   = 0;
    int bench_iters = 0;
    const char* cli_prompt = nullptr;
    int cli_max_tokens = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"      && i + 1 < argc) { bench_model    = argv[++i]; }
        else if (a == "--ctx"        && i + 1 < argc) { bench_ctx      = std::atoi(argv[++i]); }
        else if (a == "--iters"      && i + 1 < argc) { bench_iters    = std::atoi(argv[++i]); }
        else if (a == "--prompt"     && i + 1 < argc) { cli_prompt     = argv[++i]; }
        else if (a == "--max-tokens" && i + 1 < argc) { cli_max_tokens = std::atoi(argv[++i]); }
    }
    const bool bench_mode =
        (bench_model != nullptr && bench_ctx > 0 && bench_iters > 0);

    const char* path = bench_model ? bench_model
                     : (argc > 1   ? argv[1]
                                   : "/home/bcloud/halo-1bit/models/halo-1bit-2b.h1b");
    // --prompt "..." routes through the same --text code path below.
    // Signal it by swapping prompt_arg to "--text"; the --text branch reads
    // cli_prompt / cli_max_tokens when argc_use < 4.
    const char* prompt_arg = bench_mode      ? "1"
                           : cli_prompt      ? "--text"
                           : (argc > 2       ? argv[2]
                                             : "1");
    int num_tokens = bench_mode ? bench_iters
                   : cli_max_tokens > 0 ? cli_max_tokens
                   : 16;
    // Default tokenizer: same dir as model, .h1b -> .htok. Makes the binary
    // relocatable — no more build-time $HOME leak. Overridable via --tokenizer
    // flag or (in --text mode) the trailing positional arg.
    std::string derived_tok_path = path;
    if (derived_tok_path.size() > 4 &&
        derived_tok_path.compare(derived_tok_path.size() - 4, 4, ".h1b") == 0) {
        derived_tok_path.replace(derived_tok_path.size() - 4, 4, ".htok");
    } else {
        derived_tok_path += ".htok";
    }
    const char* tok_path = derived_tok_path.c_str();

    // Collect --stop "seq" / --temp <f> / --top-k <int> / --seed <int> flags.
    // Positional args must come BEFORE any named flag; argc_use caps the
    // positional scan so trailing flag blocks don't leak into chat/system
    // positional slots.
    std::vector<std::string> stop_seqs;
    float temperature = 0.0f;          // 0 = greedy argmax (default, legacy)
    int   top_k_val    = 0;            // 0 = disabled
    float top_p_val    = 1.0f;         // 1.0 = disabled (keep all mass)
    float rep_penalty  = 1.0f;         // 1.0 = disabled (no penalty)
    int   rep_last_n   = 64;           // window for repetition penalty
    uint64_t sampler_seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    int         server_port = 0;       // >0 enables HTTP server mode
    std::string server_bind = "127.0.0.1";
    bool        kv_int8 = false;       // --kv-int8 : halve KV DRAM + bandwidth
    bool        kv_rotor = false;      // --kv-rotor : PQ3 rotorquant KV (5.33x DRAM)
    int argc_use = argc;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--stop"         && i + 1 < argc) { stop_seqs.emplace_back(argv[++i]); }
        else if (a == "--temp"         && i + 1 < argc) { temperature  = (float)std::atof(argv[++i]); }
        else if (a == "--top-k"        && i + 1 < argc) { top_k_val    = std::atoi(argv[++i]); }
        else if (a == "--top-p"        && i + 1 < argc) { top_p_val    = (float)std::atof(argv[++i]); }
        else if (a == "--rep-penalty"  && i + 1 < argc) { rep_penalty  = (float)std::atof(argv[++i]); }
        else if (a == "--rep-last-n"   && i + 1 < argc) { rep_last_n   = std::atoi(argv[++i]); }
        else if (a == "--seed"         && i + 1 < argc) { sampler_seed = (uint64_t)std::atoll(argv[++i]); }
        else if (a == "--server"       && i + 1 < argc) { server_port  = std::atoi(argv[++i]); }
        else if (a == "--bind"         && i + 1 < argc) { server_bind  = argv[++i]; }
        else if (a == "--tokenizer"    && i + 1 < argc) { tok_path     = argv[++i]; }
        else if (a == "--kv-int8")                      { kv_int8      = true; if (argc_use > i) argc_use = i; continue; }
        else if (a == "--kv-rotor")                     { kv_rotor     = true; if (argc_use > i) argc_use = i; continue; }
        else continue;
        if (argc_use > i - 1) argc_use = i - 1;
    }

    // Tokenize a chunk of text with NO bos; returns IDs.
    auto tokenize = [&](rcpp_tokenizer_t* tok, const char* text) -> std::vector<int> {
        std::vector<int> buf(4096);
        size_t count = 0;
        rcpp_tokenizer_encode(tok, text, std::strlen(text), /*add_bos=*/0,
                              buf.data(), buf.size(), &count);
        if (count > buf.size()) { buf.resize(count);
            rcpp_tokenizer_encode(tok, text, std::strlen(text), 0, buf.data(), buf.size(), &count);
        }
        buf.resize(count);
        return buf;
    };

    std::vector<int> prompt_ids;
    if (std::string(prompt_arg) == "--text") {
        // layout: bitnet_decode <model> --text "<prompt>" <num_new> [tokenizer.htok]
        //     or: bitnet_decode --model <m> --prompt "<text>" --max-tokens N
        const char* text = nullptr;
        if (cli_prompt) {
            text = cli_prompt;
            if (cli_max_tokens > 0) num_tokens = cli_max_tokens;
        } else {
            if (argc_use < 4) { fprintf(stderr, "usage: --text \"<prompt text>\" <num_new> [tokenizer.htok]\n"); return 1; }
            text = argv[3];
            num_tokens = argc_use > 4 ? std::atoi(argv[4]) : 32;
            if (argc_use > 5) tok_path = argv[5];
        }
        rcpp_tokenizer_t* tok = nullptr;
        if (rcpp_tokenizer_load(tok_path, &tok) != RCPP_OK) {
            // Models without a .htok sidecar (Bonsai today) can still drive
            // a smoke run with BOS alone. Emit a loud warning + downgrade to
            // single-token-start so the decode pipe still runs.
            fprintf(stderr,
                "[tokenizer] WARN cannot load %s — falling back to BOS-only prompt. "
                "Actual text prompt discarded.\n", tok_path);
            prompt_ids.push_back(1);  // BOS in every tokenizer we've seen
        } else {
            prompt_ids.push_back(rcpp_tokenizer_bos_id(tok));
            auto text_ids = tokenize(tok, text);
            prompt_ids.insert(prompt_ids.end(), text_ids.begin(), text_ids.end());
            rcpp_tokenizer_free(tok);
            fprintf(stderr, "[tokenizer] \"%s\" -> %zu tokens\n", text, prompt_ids.size());
        }
    } else if (std::string(prompt_arg) == "--chat") {
        // layout: bitnet_decode <model> --chat "<user msg>" <num_new> [system_msg]
        // Applies BitNet's chat template: "User: msg<|eot_id|>Assistant: "
        // (verified against tokenizer_config.json for BitNet-b1.58-2B-4T)
        if (argc_use < 4) { fprintf(stderr, "usage: --chat \"<user msg>\" <num_new> [\"<system>\"] [--stop \"seq\" ...]\n"); return 1; }
        const char* user_msg = argv[3];
        num_tokens = argc_use > 4 ? std::atoi(argv[4]) : 128;
        const char* system_msg = argc_use > 5 ? argv[5] : nullptr;

        rcpp_tokenizer_t* tok = nullptr;
        if (rcpp_tokenizer_load(tok_path, &tok) != RCPP_OK) {
            fprintf(stderr, "cannot load tokenizer .htok: %s\n", tok_path); return 1;
        }
        const int BOS = rcpp_tokenizer_bos_id(tok);
        const int EOT = 128009;  // <|eot_id|>

        prompt_ids.push_back(BOS);
        if (system_msg) {
            std::string s = std::string("System: ") + system_msg;
            auto ids = tokenize(tok, s.c_str());
            prompt_ids.insert(prompt_ids.end(), ids.begin(), ids.end());
            prompt_ids.push_back(EOT);
        }
        {
            std::string s = std::string("User: ") + user_msg;
            auto ids = tokenize(tok, s.c_str());
            prompt_ids.insert(prompt_ids.end(), ids.begin(), ids.end());
            prompt_ids.push_back(EOT);
        }
        {
            auto ids = tokenize(tok, "Assistant: ");
            prompt_ids.insert(prompt_ids.end(), ids.begin(), ids.end());
        }
        rcpp_tokenizer_free(tok);
        fprintf(stderr, "[chat] user=\"%s\"%s -> %zu prompt tokens\n",
                user_msg, system_msg ? " (with system)" : "", prompt_ids.size());
    } else if (std::string(prompt_arg) == "--server") {
        // Server mode: don't build a single prompt — each HTTP request
        // carries its own conversation in OpenAI format. Prime the cache
        // with just BOS; per-request handler resets cache_pos back to 1.
        //
        //   bitnet_decode <model> --server [port=8080] [default_max_tokens=256]
        server_port = argc_use > 3 ? std::atoi(argv[3]) : 8080;
        num_tokens  = argc_use > 4 ? std::atoi(argv[4]) : 256;
        rcpp_tokenizer_t* tok = nullptr;
        if (rcpp_tokenizer_load(tok_path, &tok) != RCPP_OK) {
            fprintf(stderr, "cannot load tokenizer .htok: %s\n", tok_path); return 1;
        }
        prompt_ids.push_back(rcpp_tokenizer_bos_id(tok));
        rcpp_tokenizer_free(tok);
        fprintf(stderr, "[server] OpenAI-compat API on %s:%d (default max_tokens=%d)\n",
                server_bind.c_str(), server_port, num_tokens);
        if (server_bind == "0.0.0.0") {
            fprintf(stderr, "[server] WARNING: bound to 0.0.0.0 — publicly reachable.\n");
            fprintf(stderr, "[server] WARNING: no auth, no TLS, no rate limit. Use a reverse proxy.\n");
        }
    } else if (std::string(prompt_arg) == "--repl") {
        // REPL mode — interactive multi-turn chat with persistent KV cache.
        //   bitnet_decode <model> --repl [max_new_per_turn] [tokenizer.htok]
        // The decode loop is re-entered with new user turns appended to
        // the existing KV cache. Model stays loaded across turns; only
        // the forward-pass positions advance.
        num_tokens = argc_use > 3 ? std::atoi(argv[3]) : 256;
        if (argc_use > 4) tok_path = argv[4];
        rcpp_tokenizer_t* tok = nullptr;
        if (rcpp_tokenizer_load(tok_path, &tok) != RCPP_OK) {
            fprintf(stderr, "cannot load tokenizer .htok: %s\n", tok_path); return 1;
        }
        // Seed with just BOS — the first user message is read from stdin below.
        prompt_ids.push_back(rcpp_tokenizer_bos_id(tok));
        rcpp_tokenizer_free(tok);
        fprintf(stderr, "[repl] %d max tokens/turn. Ctrl-D or 'quit' to exit.\n", num_tokens);
    } else if (std::string(prompt_arg) == "--ppl") {
        // PPL mode — score a text file, report mean NLL / perplexity.
        //   bitnet_decode <model> --ppl <file.txt> [max_tokens] [tokenizer.htok]
        // Single pass, truncated to max_tokens (default = model max_len-1).
        if (argc_use < 4) { fprintf(stderr, "usage: --ppl <file.txt> [max_tokens] [tokenizer.htok]\n"); return 1; }
        const char* ppl_path = argv[3];
        num_tokens = argc_use > 4 ? std::atoi(argv[4]) : 4095;
        if (argc_use > 5) tok_path = argv[5];
        rcpp_tokenizer_t* tok = nullptr;
        if (rcpp_tokenizer_load(tok_path, &tok) != RCPP_OK) {
            fprintf(stderr, "cannot load tokenizer .htok: %s\n", tok_path); return 1;
        }
        std::ifstream f(ppl_path);
        if (!f) { fprintf(stderr, "cannot open --ppl file: %s\n", ppl_path); return 1; }
        std::stringstream ss; ss << f.rdbuf();
        std::string text = ss.str();
        prompt_ids.push_back(rcpp_tokenizer_bos_id(tok));
        auto text_ids = tokenize(tok, text.c_str());
        // Cap so KV cache fits; leave 1 slot for safety.
        if ((int)text_ids.size() > num_tokens) text_ids.resize(num_tokens);
        prompt_ids.insert(prompt_ids.end(), text_ids.begin(), text_ids.end());
        rcpp_tokenizer_free(tok);
        fprintf(stderr, "[ppl] %s: %zu chars -> %zu tokens\n",
                ppl_path, text.size(), prompt_ids.size());
    } else if (prompt_arg[0] == '@') {
        num_tokens = argc > 3 ? std::atoi(argv[3]) : 16;
        std::ifstream f(prompt_arg + 1);
        if (!f) { fprintf(stderr, "cannot open prompt file: %s\n", prompt_arg + 1); return 1; }
        prompt_ids = read_token_ids(f);
    } else if (std::string(prompt_arg) == "-") {
        num_tokens = argc > 3 ? std::atoi(argv[3]) : 16;
        prompt_ids = read_token_ids(std::cin);
    } else if (bench_mode) {
        // Bench mode: synthesize a bench_ctx-token prompt (BOS repeated). We
        // measure tok/s on the DECODE loop, which the run_turn helper reports
        // after prefill. bench_iters is already stamped into num_tokens above.
        num_tokens = bench_iters;
        prompt_ids.assign((size_t)bench_ctx, 1);
    } else {
        num_tokens = argc > 3 ? std::atoi(argv[3]) : 16;
        prompt_ids = { std::atoi(prompt_arg) };
    }
    if (prompt_ids.empty()) { fprintf(stderr, "prompt is empty\n"); return 1; }
    const int start_tok = prompt_ids.front();

    rcpp_bitnet_model_t m;
    if (rcpp_bitnet_load_h1b(path, &m) != RCPP_OK) {
        fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }

    // halo-ai Lane B/B'/B'': select ternary GEMV by resolved weight_format.
    //   HALO_V2      → halo kernel (2 bpw packing, int8 acts)
    //   SHERRY_I8    → sherry int8-act decoder (1.25 bpw, per-row scale baked in)
    //   TQ1          → tq1-halo kernel (1.6 bpw base-3, lossless for ternary)
    //   SHERRY_FP16  → clean-room fp16-in/fp16-out sherry kernel + post-scale
    //                  (bitnet_sherry_fp16_gemv above). Activations stay fp16,
    //                  no INT8 quant in that path.
    const bool is_sherry_fp16 = (m.weight_format == RCPP_WEIGHT_FORMAT_SHERRY_FP16);
    const bool is_bonsai_q1   = (m.weight_format == RCPP_WEIGHT_FORMAT_BONSAI_Q1);
    const bool is_bonsai_tq2  = (m.weight_format == RCPP_WEIGHT_FORMAT_BONSAI_TQ2);
    const bool is_bonsai      = is_bonsai_q1 || is_bonsai_tq2;
    // `arch` drives the attention preamble (per-head q/k norm vs
    // attn_sub_norm) and the FFN activation (SwiGLU vs squared-ReLU GLU +
    // ffn_sub_norm) — orthogonal to the ternary GEMV dispatch. A
    // BitNet-repacked Bonsai .h1b has arch=BITNET + weight_format=BONSAI_TQ2.
    const bool is_qwen3       = (m.arch == RCPP_ARCH_QWEN3);
    fprintf(stderr, "[bitnet_decode] arch=%s weight_format=%d\n",
            is_qwen3 ? "qwen3" : "bitnet", (int)m.weight_format);

    // Int8-activation dispatch (HALO_V2 / SHERRY_I8 / TQ1). Unused on the
    // SHERRY_FP16 / BONSAI_* paths — forward lambda branches before calling.
    //
    // Round-4 perf change: route through *_devscale entries. They take the
    // x_scale device pointer rather than a host float, so the per-layer
    // hipMemcpy-D2H of x_scale_dev disappears (was ~150 host stalls per
    // token at 30 layers). Math is bit-equivalent; the scalar is read once
    // per workgroup at the writeback. See kernels/ternary_gemv_devscale.hip.
    const auto ternary_gemv_i8 =
        (m.weight_format == RCPP_WEIGHT_FORMAT_TQ1)       ? rcpp_ternary_gemv_tq1_halo_f16_devscale
      : (m.weight_format == RCPP_WEIGHT_FORMAT_SHERRY_I8) ? rcpp_ternary_gemv_sherry_f16_devscale
                                                          : rcpp_ternary_gemv_halo_f16_devscale;
    // TQ1 expects K padded to multiple of 20 (u32-aligned row bytes). Other
    // formats use K directly — align_k becomes a no-op for them.
    const int k_pad_unit = (m.weight_format == RCPP_WEIGHT_FORMAT_TQ1) ? 20 : 1;
    auto align_k = [&](int k) { return ((k + k_pad_unit - 1) / k_pad_unit) * k_pad_unit; };

    const int hs  = m.hidden_size;
    const int is  = m.intermediate_size;
    const int nh  = m.num_heads;
    const int nkv = m.num_kv_heads;
    const int hd  = hs / nh;
    const int hs_k = align_k(hs);   // K for GEMVs that take hs as input dim (Q/K/V/O/gate/up)
    const int is_k = align_k(is);   // K for down_proj (takes is as input dim)
    const int L   = m.num_layers;
    const int V   = m.vocab_size;
    const bool repl_mode = (std::string(prompt_arg) == "--repl");
    const bool ppl_mode  = (std::string(prompt_arg) == "--ppl");
    const int prompt_len = (int)prompt_ids.size();
    // REPL and server both need a big KV slab — REPL for multi-turn growth,
    // server because per-request max_tokens isn't known until HTTP time.
    // 4096 matches BitNet-2B-4T's max_position_embeddings; RoPE theta 500k
    // means positions up to that bound are trained.
    const bool server_mode = (server_port > 0);
    const int max_len = (repl_mode || server_mode) ? 4096
                                                   : prompt_len + num_tokens;
    const float scale = 1.0f / std::sqrt((float)hd);

    fprintf(stderr, "[bitnet_decode] prompt_len=%d new_tokens=%d max_ctx=%d\n",
            prompt_len, num_tokens, max_len);
    (void)start_tok;  // preserved above for logging continuity

    // ---- Scratch buffers on device ----
    // x_fp32 is the FP32 residual stream (the dominant numerical-stability
    // knob in deep transformers). Sublayer math and KV cache stay FP16.
    float    *x_fp32;
    _Float16 *x, *normed, *x_i8_scratch_fp16;
    int8_t   *x_i8;
    float    *x_scale_dev;
    float    *q_raw, *k_raw, *v_raw, *o_raw, *gate_raw, *up_raw, *down_raw;
    _Float16 *q_fp16, *k_fp16, *v_fp16, *o_fp16, *gate_fp16, *up_fp16, *down_fp16;
    _Float16 *silu_out;
    int8_t   *silu_i8;
    float    *silu_scale_dev;
    float    *logits;
    int      *next_tok_dev;

    HIP_OK(hipMalloc(&x_fp32,        hs * 4));
    HIP_OK(hipMalloc(&x,             hs * 2));
    HIP_OK(hipMalloc(&normed,        hs * 2));
    HIP_OK(hipMalloc(&x_i8_scratch_fp16, hs * 2));  // unused slot, kept for parity
    HIP_OK(hipMalloc(&x_i8,          hs_k));
    HIP_OK(hipMemsetAsync(x_i8, 0, hs_k, nullptr));
    HIP_OK(hipMalloc(&x_scale_dev,   4));
    HIP_OK(hipMalloc(&q_raw,         nh * hd * 4));
    HIP_OK(hipMalloc(&k_raw,         nkv * hd * 4));
    HIP_OK(hipMalloc(&v_raw,         nkv * hd * 4));
    HIP_OK(hipMalloc(&q_fp16,        nh * hd * 2));
    HIP_OK(hipMalloc(&k_fp16,        nkv * hd * 2));
    HIP_OK(hipMalloc(&v_fp16,        nkv * hd * 2));
    HIP_OK(hipMalloc(&o_raw,         hs * 4));
    HIP_OK(hipMalloc(&o_fp16,        hs * 2));
    HIP_OK(hipMalloc(&gate_raw,      is * 4));
    HIP_OK(hipMalloc(&up_raw,        is * 4));
    HIP_OK(hipMalloc(&down_raw,      hs * 4));
    HIP_OK(hipMalloc(&gate_fp16,     is * 2));
    HIP_OK(hipMalloc(&up_fp16,       is * 2));
    HIP_OK(hipMalloc(&down_fp16,     hs * 2));
    HIP_OK(hipMalloc(&silu_out,      is * 2));
    HIP_OK(hipMalloc(&silu_i8,       is_k));
    HIP_OK(hipMemsetAsync(silu_i8, 0, is_k, nullptr));
    HIP_OK(hipMalloc(&silu_scale_dev, 4));
    HIP_OK(hipMalloc(&logits,        V * 4));
    HIP_OK(hipMalloc(&next_tok_dev,  4));

    // ---- KV cache (per layer) ----
    // Default: FP16 per-token per-kv-head per-head-dim.
    // --kv-int8  : INT8 tensor + per-(pos, kv_head) FP16 scale. Halves DRAM/BW.
    // --kv-rotor : PQ3 packed-3bit (8 idx / 3 bytes). 5.33x DRAM vs fp16.
    if (kv_int8 && kv_rotor) {
        fprintf(stderr, "[bitnet_decode] --kv-int8 and --kv-rotor are mutually exclusive\n");
        return 1;
    }
    std::vector<_Float16*> K_caches(L, nullptr), V_caches(L, nullptr);
    std::vector<int8_t*>   K_caches_i8(L, nullptr), V_caches_i8(L, nullptr);
    std::vector<_Float16*> K_scales(L, nullptr),   V_scales(L, nullptr);
    std::vector<uint8_t*>  K_caches_pq3(L, nullptr), V_caches_pq3(L, nullptr);
    const size_t kv_size     = (size_t)max_len * nkv * hd * sizeof(_Float16);
    const size_t kv_size_i8  = (size_t)max_len * nkv * hd * sizeof(int8_t);
    const size_t sc_size     = (size_t)max_len * nkv * sizeof(_Float16);
    const size_t kv_size_pq3 = (size_t)max_len * nkv * ((size_t)hd * 3 / 8);
    // PQ3 requires head_dim % 8 == 0.
    if (kv_rotor && (hd & 7) != 0) {
        fprintf(stderr, "[kv-rotor] head_dim=%d not a multiple of 8; bailing\n", hd);
        return 1;
    }
    for (int l = 0; l < L; ++l) {
        if (kv_int8) {
            HIP_OK(hipMalloc(&K_caches_i8[l], kv_size_i8));
            HIP_OK(hipMalloc(&V_caches_i8[l], kv_size_i8));
            HIP_OK(hipMalloc(&K_scales[l],    sc_size));
            HIP_OK(hipMalloc(&V_scales[l],    sc_size));
        } else if (kv_rotor) {
            HIP_OK(hipMalloc(&K_caches_pq3[l], kv_size_pq3));
            HIP_OK(hipMalloc(&V_caches_pq3[l], kv_size_pq3));
        } else {
            HIP_OK(hipMalloc(&K_caches[l], kv_size));
            HIP_OK(hipMalloc(&V_caches[l], kv_size));
        }
    }
    if (kv_int8) {
        const double fp16_mb = (double)(kv_size) * L * 2 / (1024.0 * 1024.0);
        const double i8_mb   = (double)(kv_size_i8 + sc_size) * L * 2 / (1024.0 * 1024.0);
        fprintf(stderr, "[kv-int8] KV cache: %.1f MB (vs %.1f MB fp16, %.2fx)\n",
                i8_mb, fp16_mb, fp16_mb / std::max(i8_mb, 1e-9));
    }
    if (kv_rotor) {
        const double fp16_mb = (double)(kv_size) * L * 2 / (1024.0 * 1024.0);
        const double pq3_mb  = (double)(kv_size_pq3) * L * 2 / (1024.0 * 1024.0);
        fprintf(stderr, "[kv-rotor] KV cache: %.1f MB (vs %.1f MB fp16, %.2fx)\n",
                pq3_mb, fp16_mb, fp16_mb / std::max(pq3_mb, 1e-9));
    }

    // History the sampler sees — used for repetition penalty. Grows as
    // forward_token is called; seeded with prompt IDs before the loop.
    std::vector<int> sampler_history;
    sampler_history.reserve(prompt_len + num_tokens);

    std::mt19937_64 rng(sampler_seed);
    if (temperature > 0.0f) {
        fprintf(stderr, "[sampler] temp=%.3f top_k=%d top_p=%.3f rep=%.3f/%d seed=%llu\n",
                temperature, top_k_val, top_p_val, rep_penalty, rep_last_n,
                (unsigned long long)sampler_seed);
    }

    // Host-side sampler (used only when temperature > 0). Reads the
    // FP32 logits buffer from device, applies the full filter chain
    // (repetition penalty -> top-k mask -> softmax(temp) -> top-p mask
    // -> multinomial), returns the sampled token id. One hipMemcpy per
    // token (~V*4 bytes, sub-1% of the 12 ms/tok decode cost) in
    // exchange for zero sort-on-GPU complexity.
    std::vector<float> logits_host(V);
    auto sample_host = [&](const std::vector<int>& recent) -> int {
        HIP_OK(hipMemcpy(logits_host.data(), logits, V * 4, hipMemcpyDeviceToHost));

        // Repetition penalty: downweight logits of recently-emitted tokens.
        // rep_penalty > 1 : discourage repeat (divide positive, multiply negative)
        // rep_penalty < 1 : encourage repeat (rare).
        if (rep_penalty != 1.0f && rep_last_n > 0) {
            int start = std::max(0, (int)recent.size() - rep_last_n);
            for (int i = start; i < (int)recent.size(); ++i) {
                int id = recent[i];
                if (id >= 0 && id < V) {
                    float& l = logits_host[id];
                    l = (l > 0.0f) ? (l / rep_penalty) : (l * rep_penalty);
                }
            }
        }

        // Top-k: keep only the top k, mask rest to -inf.
        if (top_k_val > 0 && top_k_val < V) {
            std::vector<float> tmp(logits_host);
            std::nth_element(tmp.begin(), tmp.begin() + (V - top_k_val), tmp.end());
            float thresh = tmp[V - top_k_val];
            for (float& l : logits_host) if (l < thresh) l = -INFINITY;
        }

        // Softmax with temperature.
        float m = -INFINITY;
        for (float l : logits_host) if (l > m) m = l;
        double sum = 0.0;
        for (float& l : logits_host) { l = std::exp((l - m) / temperature); sum += l; }
        const float inv = (float)(1.0 / (sum > 0 ? sum : 1.0));
        for (float& l : logits_host) l *= inv;

        // Top-p: sort descending, keep smallest prefix with cumsum >= p.
        if (top_p_val > 0.0f && top_p_val < 1.0f) {
            std::vector<int> idx(V);
            for (int i = 0; i < V; ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b) { return logits_host[a] > logits_host[b]; });
            float csum = 0.0f;
            int cutoff = V;
            for (int i = 0; i < V; ++i) {
                csum += logits_host[idx[i]];
                if (csum >= top_p_val) { cutoff = i + 1; break; }
            }
            // Zero out the tail.
            for (int i = cutoff; i < V; ++i) logits_host[idx[i]] = 0.0f;
            // Renormalize the kept head.
            float keep_sum = 0.0f;
            for (int i = 0; i < cutoff; ++i) keep_sum += logits_host[idx[i]];
            if (keep_sum > 0) {
                float s = 1.0f / keep_sum;
                for (int i = 0; i < cutoff; ++i) logits_host[idx[i]] *= s;
            }
        }

        // Multinomial draw.
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        float r = u(rng);
        float acc = 0.0f;
        for (int i = 0; i < V; ++i) {
            acc += logits_host[i];
            if (acc >= r) return i;
        }
        return V - 1;
    };

    // Ternary GEMV dispatcher. In the int8 paths (HALO_V2/SHERRY_I8/TQ1) this
    // forwards the (packed, x_i8, x_scale_dev, row_scales, out, N, K) args
    // to the int8-act kernel via the *_devscale entries — x_scale is read
    // off the device pointer inside the kernel, no host stall. In the
    // SHERRY_FP16 path it ignores x_i8/x_scale_dev and drives the
    // fp16-in/fp16-out clean-room kernel off `normed` directly, then
    // applies row_scales in the post-pass helper.
    auto ternary_gemv = [&](const void* packed, const int8_t* x_i8_in,
                            const float* x_scale_dev_arg,
                            const float* row_scales, const void* normed_fp16,
                            void* out_fp16, int N, int K) -> rcpp_status_t {
        if (is_bonsai) {
            return bonsai_gemv_dispatch(m.weight_format, packed, normed_fp16,
                                        out_fp16, N, K, /*stream=*/nullptr);
        }
        if (is_sherry_fp16) {
            return bitnet_sherry_fp16_gemv(packed, normed_fp16, row_scales,
                                           out_fp16, N, K, /*stream=*/nullptr);
        }
        return ternary_gemv_i8(packed, x_i8_in, x_scale_dev_arg, row_scales,
                               out_fp16, N, K, /*stream=*/nullptr);
    };

    // ---- Forward pass for one token at position pos ----
    auto forward_token = [&](int token_id, int pos) -> int {
        // Seed the FP32 residual stream from the FP16 embedding.
        RC_OK(rcpp_embedding_lookup_fp16(m.embedding_dev, token_id, x, hs, nullptr));
        HIP_OK(hipMemsetAsync(x_fp32, 0, hs * sizeof(float), nullptr));
        RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, x, hs, nullptr));

        for (int l = 0; l < L; ++l) {
            rcpp_bitnet_layer_t& ly = m.layers[l];

            // --- Attention block ---
            RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, ly.input_norm_dev, normed,
                                                m.rms_norm_eps, hs, nullptr));
            // INT8 quant only matters for the int8 paths; the SHERRY_FP16 path
            // reads `normed` directly. We still run it so x_i8 / x_scale_dev
            // are valid if we ever hop formats mid-decode (we don't today, but
            // the cost is one small kernel per layer — negligible vs 7 GEMVs).
            // Round-4: x_scale stays on the device — no host hipMemcpy needed
            // because the *_devscale GEMV reads it from x_scale_dev.
            RC_OK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs, nullptr));

            // Q/K/V projections — fused FP16 output (halo-ai Lane A)
            RC_OK(ternary_gemv(ly.q_packed_dev, x_i8, x_scale_dev, ly.q_scales_dev, normed, q_fp16, nh*hd,  hs_k));
            RC_OK(ternary_gemv(ly.k_packed_dev, x_i8, x_scale_dev, ly.k_scales_dev, normed, k_fp16, nkv*hd, hs_k));
            RC_OK(ternary_gemv(ly.v_packed_dev, x_i8, x_scale_dev, ly.v_scales_dev, normed, v_fp16, nkv*hd, hs_k));

            // Qwen3 attention preamble — per-head RMSNorm on Q and K, applied
            // BEFORE RoPE (matches HuggingFace qwen3/modeling_qwen3.py
            // `Qwen3Attention.forward`: `q = self.q_norm(q.view(B, -1, nh, hd))`
            // then rope). BitNet / Sherry skip this block.
            if (is_qwen3 && ly.attn_q_norm_dev) {
                for (int h = 0; h < nh; ++h) {
                    _Float16* qh = q_fp16 + (size_t)h * hd;
                    RC_OK(rcpp_rmsnorm_fp16(qh, ly.attn_q_norm_dev, qh,
                                            m.rms_norm_eps, hd, nullptr));
                }
            }
            if (is_qwen3 && ly.attn_k_norm_dev) {
                for (int h = 0; h < nkv; ++h) {
                    _Float16* kh = k_fp16 + (size_t)h * hd;
                    RC_OK(rcpp_rmsnorm_fp16(kh, ly.attn_k_norm_dev, kh,
                                            m.rms_norm_eps, hd, nullptr));
                }
            }

            // RoPE on Q and K
            RC_OK(rcpp_rope_fp16(q_fp16, pos, m.rope_theta, nh,  hd, nullptr));
            RC_OK(rcpp_rope_fp16(k_fp16, pos, m.rope_theta, nkv, hd, nullptr));

            // Append this token's K/V to the per-layer cache at slot 'pos'
            if (kv_int8) {
                // Quantize K/V [nkv, hd] to INT8 with per-kv-head FP16 scale,
                // writing directly into the cache at offset 'pos'.
                RC_OK(rcpp_quantize_fp16_to_i8_rowscale(
                    k_fp16,
                    K_caches_i8[l] + (size_t)pos * nkv * hd,
                    K_scales[l]    + (size_t)pos * nkv,
                    nkv, hd, nullptr));
                RC_OK(rcpp_quantize_fp16_to_i8_rowscale(
                    v_fp16,
                    V_caches_i8[l] + (size_t)pos * nkv * hd,
                    V_scales[l]    + (size_t)pos * nkv,
                    nkv, hd, nullptr));
                // Split-KV Flash-Decoding over int8 storage: same per-(pos,
                // kv_head) scale broadcast as the single-block kernel, but
                // grid = (num_q_heads, num_kv_tiles) recovers occupancy at
                // long ctx. Drop-in replacement; bit-for-bit equivalent
                // online softmax to the single-block path.
                RC_OK(rcpp_kv_cache_attn_decode_fd_i8(
                    q_fp16,
                    K_caches_i8[l], V_caches_i8[l],
                    K_scales[l],    V_scales[l],
                    o_fp16, nh, nkv, hd, pos+1, scale, nullptr));
            } else if (kv_rotor) {
                // Rotorquant PQ3 — one (pos, kv_head) row per new token. The
                // requantize kernel writes exactly nkv rows of (hd*3/8) bytes
                // at the tail of the cache; a seq_len=1 launch with per-layer
                // layer_idx gives the right rotation seed.
                const size_t pq3_row = (size_t)hd * 3 / 8;
                RC_OK((rcpp_status_t)0);  // keep macro symmetry; no status here
                rcpp_kv_requantize_pq3  (k_fp16,
                    K_caches_pq3[l] + (size_t)pos * nkv * pq3_row,
                    /*seq_len=*/1, nkv, hd, /*layer_idx=*/l, nullptr);
                rcpp_kv_requantize_pq3_v(v_fp16,
                    V_caches_pq3[l] + (size_t)pos * nkv * pq3_row,
                    1, nkv, hd, l, nullptr);
                int rrc = rcpp_kv_cache_attn_decode_fd_pq3(
                    q_fp16,
                    K_caches_pq3[l], V_caches_pq3[l],
                    o_fp16, nh, nkv, hd, pos+1, /*layer_idx=*/l, scale, nullptr);
                if (rrc != 0) { fprintf(stderr, "kv-rotor rc=%d layer=%d\n", rrc, l); return 1; }
            } else {
                HIP_OK(hipMemcpy(K_caches[l] + (size_t)pos * nkv * hd, k_fp16, nkv*hd*2, hipMemcpyDeviceToDevice));
                HIP_OK(hipMemcpy(V_caches[l] + (size_t)pos * nkv * hd, v_fp16, nkv*hd*2, hipMemcpyDeviceToDevice));
                RC_OK(rcpp_kv_cache_attn_decode_fd(q_fp16, K_caches[l], V_caches[l],
                                                   o_fp16, nh, nkv, hd, pos+1, scale, nullptr));
            }

            if (is_qwen3) {
                // Qwen3 has no attn_sub_norm — attention output feeds O proj
                // directly. Reuse `normed` as the O-proj output buffer so we
                // don't trample o_fp16 mid-compute when residual-adds it.
                // Round-4: x_scale stays on device; no host hipMemcpy here.
                RC_OK(rcpp_quantize_fp16_to_i8(o_fp16, x_i8, x_scale_dev, hs, nullptr));
                RC_OK(ternary_gemv(ly.o_packed_dev, x_i8, x_scale_dev, ly.o_scales_dev,
                                   o_fp16, normed, hs, align_k(nh*hd)));
                RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, normed, hs, nullptr));
            } else {
                // BitNet b1.58: attn_sub_norm on attention output before O proj.
                RC_OK(rcpp_rmsnorm_fp16(o_fp16, ly.attn_sub_norm_dev, normed,
                                        m.rms_norm_eps, hs, nullptr));
                RC_OK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs, nullptr));
                RC_OK(ternary_gemv(ly.o_packed_dev, x_i8, x_scale_dev, ly.o_scales_dev, normed, o_fp16, hs, align_k(nh*hd)));
                RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, o_fp16, hs, nullptr));
            }

            // --- FFN block ---
            RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, ly.post_attn_norm_dev, normed,
                                                m.rms_norm_eps, hs, nullptr));
            RC_OK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs, nullptr));

            RC_OK(ternary_gemv(ly.gate_packed_dev, x_i8, x_scale_dev, ly.gate_scales_dev, normed, gate_fp16, is, hs_k));
            RC_OK(ternary_gemv(ly.up_packed_dev,   x_i8, x_scale_dev, ly.up_scales_dev,   normed, up_fp16,   is, hs_k));

            if (is_qwen3) {
                // Qwen3 FFN activation: SwiGLU = silu(gate) * up. No
                // ffn_sub_norm between GLU and down_proj.
                // rcpp_silu_glu_fp16's C API signature is (up, gate, y) but
                // the kernel computes `silu(first_arg) * second_arg`
                // (see tests/test_prim_and_attn.cpp:160 — ref is
                // `silu(u) * g`). Canonical Qwen3/HF SwiGLU is
                // `silu(gate) * up`, so we pass `gate` as the first arg
                // and `up` as the second. Passing them in the "natural"
                // name order computes `silu(up) * gate`, which is a
                // factor-swap bug that produces near-uniform logits.
                RC_OK(rcpp_silu_glu_fp16(gate_fp16, up_fp16, silu_out, is, nullptr));
            } else {
                // BitNet-b1.58 FFN activation: relu²(gate) * up — fused with
                // ffn_sub_norm in FP32 to avoid FP16 overflow of the raw
                // product (magnitude ~1e9 on real weights; FP16 max 6.5e4).
                RC_OK(rcpp_relu2_glu_rmsnorm_fp16(gate_fp16, up_fp16, ly.ffn_sub_norm_dev,
                                                  silu_out, m.rms_norm_eps, is, nullptr));
            }
            // Round-4: silu_scale stays on device; ternary_gemv reads it from
            // silu_scale_dev via the *_devscale entry.
            RC_OK(rcpp_quantize_fp16_to_i8(silu_out, silu_i8, silu_scale_dev, is, nullptr));

            RC_OK(ternary_gemv(ly.down_packed_dev, silu_i8, silu_scale_dev, ly.down_scales_dev, silu_out, down_fp16, hs, is_k));
            RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, down_fp16, hs, nullptr));
        }

        // Final norm reads FP32 residual, emits FP16 → tied LM head GEMV.
        RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, m.final_norm_weight_dev, normed,
                                            m.rms_norm_eps, hs, nullptr));
        RC_OK(rcpp_fp16_gemv(m.embedding_dev, normed, logits, V, hs, nullptr));

        // Fast path: greedy argmax on device. Full sampler chain (rep
        // penalty / top-k / top-p / multinomial) runs host-side when
        // temperature > 0 — see sample_host lambda above main loop.
        int next_tok;
        if (temperature <= 0.0f) {
            RC_OK(rcpp_argmax_fp32(logits, next_tok_dev, V, nullptr));
            HIP_OK(hipMemcpy(&next_tok, next_tok_dev, 4, hipMemcpyDeviceToHost));
        } else {
            next_tok = sample_host(sampler_history);
        }
        return next_tok;
    };

    // Load the tokenizer up front — used for streaming detokenization,
    // --stop suffix match, and the REPL's per-turn encode of user input.
    rcpp_tokenizer_t* dec_tok = nullptr;
    rcpp_tokenizer_load(tok_path, &dec_tok);

    sampler_history = prompt_ids;

    // Stop conditions:
    //   * Special IDs 128001 <|end_of_text|> / 128009 <|eot_id|>
    //   * User-supplied --stop "seq" string(s), matched against the
    //     detokenized tail of the generated window
    const int stop_a = 128001, stop_b = 128009;

    // Turn state — advances across REPL turns.
    int cache_pos  = 0;          // next free slot in the KV cache
    int last_tok   = 0;          // last token processed (prefill tail)
    double last_decode_ms     = 0.0;   // bench-mode handle on decode latency
    size_t last_decode_tokens = 0;     // bench-mode handle on decode count
    std::vector<char> tail_buf(8192);
    std::vector<char> stream_buf(16 * 1024);

    // When out_text != nullptr, generated text is appended there
    // instead of streaming to stdout — used by the HTTP non-streaming
    // path. When on_stream is set, each incremental byte-range goes
    // through the callback (used by the HTTP SSE path). Both can be
    // set together; the callback still fires.
    using StreamCb = std::function<void(const std::string&)>;
    auto run_turn = [&](const std::vector<int>& new_tokens, int max_new,
                        std::string* out_text = nullptr,
                        StreamCb on_stream = nullptr) -> int {
        const bool silent = (out_text != nullptr) || (on_stream != nullptr);
        // Prefill the new tokens (positions cache_pos..cache_pos+N-1).
        double prefill_ms = 0.0;
        for (size_t i = 0; i < new_tokens.size(); ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            last_tok = forward_token(new_tokens[i], cache_pos + (int)i);
            HIP_OK(hipDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            prefill_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        cache_pos += (int)new_tokens.size();
        fprintf(stderr, "[bitnet_decode] prefill %zu tok in %.2f ms (%.1f tok/s) [pos=%d]\n",
                new_tokens.size(), prefill_ms,
                prefill_ms > 0 ? 1000.0 * new_tokens.size() / prefill_ms : 0.0, cache_pos);

        if (max_new <= 0) return 0;

        // Decode loop: streaming text to stdout, IDs+stats to stderr.
        std::vector<int> generated;
        generated.reserve(max_new);
        fprintf(stderr, "[bitnet_decode] tokens:");
        double decode_ms = 0.0;
        int cur_tok = last_tok;
        bool hit_eos = false;
        std::string stop_hit;
        size_t printed_bytes = 0;
        for (int step = 0; step < max_new; ++step) {
            auto t0 = std::chrono::high_resolution_clock::now();
            int next_tok = forward_token(cur_tok, cache_pos + step);
            HIP_OK(hipDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            decode_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            generated.push_back(next_tok);
            sampler_history.push_back(next_tok);
            fprintf(stderr, " %d", next_tok);
            fflush(stderr);
            if (dec_tok) {
                size_t tlen = 0;
                rcpp_tokenizer_decode(dec_tok, generated.data(), generated.size(),
                                      stream_buf.data(), stream_buf.size(), &tlen);
                tlen = std::min(tlen, stream_buf.size());
                if (tlen > printed_bytes) {
                    std::string delta(stream_buf.data() + printed_bytes,
                                      tlen - printed_bytes);
                    if (out_text)   out_text->append(delta);
                    if (on_stream)  on_stream(delta);
                    if (!silent) {
                        fwrite(delta.data(), 1, delta.size(), stdout);
                        fflush(stdout);
                    }
                    printed_bytes = tlen;
                }
            }
            if (next_tok == stop_a || next_tok == stop_b) { hit_eos = true; break; }
            cur_tok = next_tok;

            if (!stop_seqs.empty() && dec_tok) {
                size_t win = std::min((size_t)64, generated.size());
                size_t tlen = 0;
                rcpp_tokenizer_decode(dec_tok,
                                      generated.data() + (generated.size() - win),
                                      win, tail_buf.data(), tail_buf.size(), &tlen);
                tlen = std::min(tlen, tail_buf.size());
                std::string tail(tail_buf.data(), tlen);
                for (const auto& s : stop_seqs) {
                    if (tail.size() >= s.size() &&
                        tail.compare(tail.size() - s.size(), s.size(), s) == 0) {
                        stop_hit = s; break;
                    }
                }
                if (!stop_hit.empty()) break;
            }
        }
        fprintf(stderr, "\n");
        if (!silent) { printf("\n"); fflush(stdout); }
        cache_pos += (int)generated.size();
        last_tok = generated.empty() ? last_tok : generated.back();
        if (hit_eos)
            fprintf(stderr, "[bitnet_decode] EOS (%d) after %zu new tokens\n",
                    last_tok, generated.size());
        else if (!stop_hit.empty())
            fprintf(stderr, "[bitnet_decode] --stop \"%s\" after %zu new tokens\n",
                    stop_hit.c_str(), generated.size());
        if (!generated.empty())
            fprintf(stderr, "[bitnet_decode] decode %zu tok in %.2f ms (%.2f ms/tok, %.1f tok/s)\n",
                    generated.size(), decode_ms, decode_ms/generated.size(),
                    1000.0 * generated.size() / decode_ms);
        last_decode_ms     = decode_ms;
        last_decode_tokens = generated.size();
        return 0;
    };

    // PPL mode: feed each token, read fp32 logits, accumulate -log_softmax
    // at the next token's id. exp(mean NLL) = perplexity.
    if (ppl_mode) {
        const size_t N = prompt_ids.size();
        if (N < 2) { fprintf(stderr, "[ppl] need >= 2 tokens, got %zu\n", N); return 1; }
        double total_nll = 0.0;
        size_t scored = 0;
        auto t_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i + 1 < N; ++i) {
            (void)forward_token((int)prompt_ids[i], (int)i);
            HIP_OK(hipDeviceSynchronize());
            HIP_OK(hipMemcpy(logits_host.data(), logits, V * 4, hipMemcpyDeviceToHost));
            float max_l = logits_host[0];
            for (int v = 1; v < V; ++v) if (logits_host[v] > max_l) max_l = logits_host[v];
            double sum_exp = 0.0;
            for (int v = 0; v < V; ++v) sum_exp += std::exp((double)(logits_host[v] - max_l));
            const int target = prompt_ids[i + 1];
            double logp = (double)(logits_host[target] - max_l) - std::log(sum_exp);
            total_nll += -logp;
            ++scored;
            if ((scored & 63) == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double el = std::chrono::duration<double>(now - t_start).count();
                fprintf(stderr, "\r[ppl] %zu/%zu  mean_nll=%.4f  ppl=%.2f  %.1f tok/s",
                        scored, N - 1, total_nll / scored, std::exp(total_nll / scored),
                        scored / std::max(el, 1e-9));
                fflush(stderr);
            }
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        double el = std::chrono::duration<double>(t_end - t_start).count();
        double mean_nll = total_nll / (double)scored;
        fprintf(stderr, "\n[ppl] scored=%zu mean_nll=%.6f perplexity=%.4f elapsed=%.2fs tok/s=%.1f\n",
                scored, mean_nll, std::exp(mean_nll), el, scored / el);
        printf("{\"scored\":%zu,\"mean_nll\":%.6f,\"perplexity\":%.6f,\"elapsed_s\":%.3f,\"tok_per_s\":%.2f}\n",
               scored, mean_nll, std::exp(mean_nll), el, scored / el);
        for (int l = 0; l < L; ++l) {
            if (kv_int8) {
                hipFree(K_caches_i8[l]); hipFree(V_caches_i8[l]);
                hipFree(K_scales[l]);    hipFree(V_scales[l]);
            } else if (kv_rotor) {
                hipFree(K_caches_pq3[l]); hipFree(V_caches_pq3[l]);
            } else {
                hipFree(K_caches[l]); hipFree(V_caches[l]);
            }
        }
        return 0;
    }

    // Run the first (or only) turn. In REPL / server modes the "initial
    // prompt" is just BOS — do a prefill-only pass (max_new=0) and let
    // the loop below drive real generation once the user / HTTP
    // request arrives.
    (void)run_turn(prompt_ids, (repl_mode || server_mode) ? 0 : num_tokens);

    if (bench_mode) {
        // Emit just the tok/s number as the LAST stdout line so the bench
        // script (tail -1) captures a clean float. Other output went to
        // stderr.
        const double tok_s = (last_decode_ms > 0 && last_decode_tokens > 0)
            ? (1000.0 * (double)last_decode_tokens / last_decode_ms)
            : 0.0;
        printf("%.2f\n", tok_s);
        fflush(stdout);
        return 0;
    }

    // REPL loop: read stdin, wrap in chat template, feed + decode. Keep
    // going until EOF or "quit".
    if (repl_mode) {
        rcpp_tokenizer_t* rtok = nullptr;
        if (rcpp_tokenizer_load(tok_path, &rtok) != RCPP_OK) { fprintf(stderr, "repl: tokenizer fail\n"); return 1; }
        auto encode_chunk = [&](const char* s) {
            std::vector<int> buf(2048);
            size_t n = 0;
            rcpp_tokenizer_encode(rtok, s, std::strlen(s), 0, buf.data(), buf.size(), &n);
            buf.resize(std::min(n, buf.size()));
            return buf;
        };
        const int EOT = 128009;
        std::string line;
        while (true) {
            fprintf(stderr, "\n> ");
            fflush(stderr);
            if (!std::getline(std::cin, line)) break;
            if (line == "quit" || line == "exit") break;
            if (line.empty()) continue;
            auto user_ids = encode_chunk((std::string("User: ") + line).c_str());
            user_ids.push_back(EOT);
            auto assist_pre = encode_chunk("Assistant: ");
            std::vector<int> turn;
            turn.reserve(user_ids.size() + assist_pre.size());
            turn.insert(turn.end(), user_ids.begin(), user_ids.end());
            turn.insert(turn.end(), assist_pre.begin(), assist_pre.end());
            if (cache_pos + (int)turn.size() + num_tokens >= max_len) {
                fprintf(stderr, "\n[repl] context full (%d/%d), resetting\n",
                        cache_pos, max_len);
                break;
            }
            (void)run_turn(turn, num_tokens);
        }
        rcpp_tokenizer_free(rtok);
    }

    if (server_mode) {
        // OpenAI-compatible HTTP server. Single-threaded inference —
        // requests serialize through the one model instance. Each
        // request resets the KV cache and runs from scratch.
        rcpp_tokenizer_t* stok = nullptr;
        if (rcpp_tokenizer_load(tok_path, &stok) != RCPP_OK) { return 1; }
        auto srv_encode = [&](const std::string& s) {
            std::vector<int> buf(4096); size_t n = 0;
            rcpp_tokenizer_encode(stok, s.c_str(), s.size(), 0,
                                  buf.data(), buf.size(), &n);
            buf.resize(std::min(n, buf.size())); return buf;
        };
        const int BOS = rcpp_tokenizer_bos_id(stok);
        const int EOT = 128009;

        std::mutex gen_mu;  // serialize generation across concurrent requests

        httplib::Server svr;

        svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("OK\n", "text/plain");
        });

        svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
            nlohmann::json j = {{"object", "list"}, {"data", nlohmann::json::array({
                {{"id", "bitnet-b1.58-2b-4t"}, {"object", "model"}, {"owned_by", "halo-ai"}}
            })}};
            res.set_content(j.dump(), "application/json");
        });

        svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
            // shared_ptr<unique_lock> so the streaming branch can copy-capture
            // it into the chunked content provider and keep the mutex held for
            // the duration of GPU work (provider runs AFTER the handler returns).
            // httplib stores the provider as std::function, which requires a
            // copyable callable — unique_lock alone isn't.
            auto lk = std::make_shared<std::unique_lock<std::mutex>>(gen_mu);
            nlohmann::json j;
            try { j = nlohmann::json::parse(req.body); }
            catch (const std::exception& e) {
                res.status = 400;
                res.set_content(std::string("{\"error\":\"bad json: ") + e.what() + "\"}",
                                "application/json");
                return;
            }

            // Build prompt token stream from OpenAI-format messages using
            // BitNet's chat template.
            std::vector<int> req_prompt = { BOS };
            for (auto& m : j["messages"]) {
                std::string role = m.value("role", std::string("user"));
                std::string content = m.value("content", std::string(""));
                if (!role.empty()) role[0] = (char)std::toupper((unsigned char)role[0]);
                auto ids = srv_encode(role + ": " + content);
                req_prompt.insert(req_prompt.end(), ids.begin(), ids.end());
                req_prompt.push_back(EOT);
            }
            auto pre = srv_encode("Assistant: ");
            req_prompt.insert(req_prompt.end(), pre.begin(), pre.end());

            int   req_max = j.value("max_tokens", 256);
            bool  req_stream = j.value("stream", false);
            std::string model_name = j.value("model", std::string("bitnet-b1.58-2b-4t"));

            // Clamp to the KV cache bound. max_len is the fixed allocation;
            // prompt + generated must fit. Reject cleanly if prompt alone
            // overflows so the caller gets a 400 rather than a SEGV.
            const int ctx_room = max_len - (int)req_prompt.size();
            if (ctx_room <= 0) {
                res.status = 400;
                nlohmann::json err = {{"error", {
                    {"message", "prompt exceeds context window (" +
                                std::to_string(req_prompt.size()) + " >= " +
                                std::to_string(max_len) + ")"},
                    {"type", "invalid_request_error"},
                    {"code", "context_length_exceeded"}
                }}};
                res.set_content(err.dump(), "application/json");
                return;
            }
            if (req_max > ctx_room) req_max = ctx_room;

            // Per-request sampler overrides (reset to session defaults after).
            float save_temp = temperature, save_top_p = top_p_val, save_rep = rep_penalty;
            int   save_top_k = top_k_val;
            temperature = j.value("temperature", temperature);
            top_p_val   = j.value("top_p",       top_p_val);
            rep_penalty = j.value("frequency_penalty", 0.0f) + 1.0f;
            if (rep_penalty < 1.0f) rep_penalty = 1.0f;

            cache_pos = 0;
            sampler_history.clear();
            sampler_history = req_prompt;

            const std::string chat_id =
                "chatcmpl-" + std::to_string((long)std::chrono::steady_clock::now().time_since_epoch().count());
            const long created = (long)std::time(nullptr);

            // ─── Streaming path (SSE chunked /v1/chat/completions) ───
            if (req_stream) {
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [this_dump = chat_id, created, model_name,
                     prompt_sz = (int)req_prompt.size(),
                     &run_turn, &temperature, &top_p_val, &rep_penalty, &top_k_val,
                     save_temp, save_top_p, save_rep, save_top_k,
                     req_prompt, req_max,
                     lk](size_t, httplib::DataSink& sink) {
                        auto fire = [&](const std::string& delta) {
                            nlohmann::json chunk = {
                                {"id",      this_dump},
                                {"object",  "chat.completion.chunk"},
                                {"created", created},
                                {"model",   model_name},
                                {"choices", nlohmann::json::array({
                                    {{"index", 0},
                                     {"delta", {{"content", delta}}},
                                     {"finish_reason", nullptr}}
                                })}
                            };
                            std::string line = "data: " + chunk.dump() + "\n\n";
                            sink.write(line.data(), line.size());
                        };
                        // First event carries just the role (OpenAI convention).
                        {
                            nlohmann::json first = {
                                {"id", this_dump}, {"object", "chat.completion.chunk"},
                                {"created", created}, {"model", model_name},
                                {"choices", nlohmann::json::array({
                                    {{"index", 0},
                                     {"delta", {{"role", "assistant"}}},
                                     {"finish_reason", nullptr}}
                                })}
                            };
                            std::string l = "data: " + first.dump() + "\n\n";
                            sink.write(l.data(), l.size());
                        }
                        (void)run_turn(req_prompt, req_max, nullptr, fire);
                        // Final chunk — finish_reason stop, then [DONE].
                        nlohmann::json fin = {
                            {"id", this_dump}, {"object", "chat.completion.chunk"},
                            {"created", created}, {"model", model_name},
                            {"choices", nlohmann::json::array({
                                {{"index", 0},
                                 {"delta", nlohmann::json::object()},
                                 {"finish_reason", "stop"}}
                            })}
                        };
                        std::string l = "data: " + fin.dump() + "\n\n";
                        sink.write(l.data(), l.size());
                        std::string done = "data: [DONE]\n\n";
                        sink.write(done.data(), done.size());
                        sink.done();
                        // Restore session sampler after the stream ends.
                        temperature = save_temp; top_p_val = save_top_p;
                        rep_penalty = save_rep;  top_k_val = save_top_k;
                        return true;
                    });
                return;
            }

            // ─── Non-streaming path (single JSON response) ───
            std::string text;
            auto t0 = std::chrono::steady_clock::now();
            (void)run_turn(req_prompt, req_max, &text);
            auto t1 = std::chrono::steady_clock::now();
            double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            temperature = save_temp; top_p_val = save_top_p;
            rep_penalty = save_rep;  top_k_val = save_top_k;

            const std::string eot = "<|eot_id|>";
            if (text.size() >= eot.size() &&
                text.compare(text.size() - eot.size(), eot.size(), eot) == 0) {
                text.erase(text.size() - eot.size());
            }

            nlohmann::json resp = {
                {"id",      chat_id},
                {"object",  "chat.completion"},
                {"created", created},
                {"model",   model_name},
                {"choices", nlohmann::json::array({
                    {{"index", 0},
                     {"message", {{"role", "assistant"}, {"content", text}}},
                     {"finish_reason", "stop"}}
                })},
                {"usage", {
                    {"prompt_tokens",     (int)req_prompt.size()},
                    {"completion_tokens", (int)cache_pos - (int)req_prompt.size()},
                    {"total_tokens",      (int)cache_pos},
                    {"latency_ms",        dt_ms}
                }}
            };
            res.set_content(resp.dump(), "application/json");
        });

        fprintf(stderr, "[server] listening on %s:%d\n",
                server_bind.c_str(), server_port);
        svr.listen(server_bind.c_str(), server_port);

        rcpp_tokenizer_free(stok);
    }

    if (dec_tok) rcpp_tokenizer_free(dec_tok);

    // Cleanup
    for (int l = 0; l < L; ++l) {
        if (kv_int8) {
            hipFree(K_caches_i8[l]); hipFree(V_caches_i8[l]);
            hipFree(K_scales[l]);    hipFree(V_scales[l]);
        } else if (kv_rotor) {
            hipFree(K_caches_pq3[l]); hipFree(V_caches_pq3[l]);
        } else {
            hipFree(K_caches[l]); hipFree(V_caches[l]);
        }
    }
    rcpp_bitnet_free(&m);
    return 0;
}
