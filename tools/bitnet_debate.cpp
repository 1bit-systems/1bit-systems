// bitnet_debate — two BitNet instances debating a topic.
//
// Same model weights loaded once, two independent KV caches. One side
// argues FOR the topic, the other argues AGAINST. The moderator kicks
// off with the topic; each side responds to the other's previous turn
// for N rounds. No agent framework deps — this is the single-process
// proof-of-concept that agent-cpp specialists will later compose.
//
// Usage:
//   bitnet_debate <model.h1b> "<topic>" [rounds=3] [max_tok_per_turn=120]

#include "rocm_cpp/ck_gemm.h"
#include "rocm_cpp/bitnet_model.h"
#include "rocm_cpp/tokenizer.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP %d %s:%d\n",_s,__FILE__,__LINE__); return 1;}} while(0)
#define RC_OK(e)  do { auto _s=(e); if(_s!=RCPP_OK){fprintf(stderr,"rcpp err %d at %s:%d\n",(int)_s,__FILE__,__LINE__); return 1;}} while(0)

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: bitnet_debate <model.h1b> \"<topic>\" [rounds=3] [max_tok=120]\n");
        return 1;
    }
    const char* model_path = argv[1];
    const char* topic      = argv[2];
    const int   rounds     = argc > 3 ? std::atoi(argv[3]) : 3;
    const int   max_tok    = argc > 4 ? std::atoi(argv[4]) : 120;
    // Derive tokenizer path from model path (.h1b -> .htok, same dir).
    // Keeps the binary relocatable instead of baking in a build-time $HOME.
    std::string derived_tok_path = model_path;
    if (derived_tok_path.size() > 4 &&
        derived_tok_path.compare(derived_tok_path.size() - 4, 4, ".h1b") == 0) {
        derived_tok_path.replace(derived_tok_path.size() - 4, 4, ".htok");
    } else {
        derived_tok_path += ".htok";
    }
    const char* tok_path   = derived_tok_path.c_str();

    rcpp_bitnet_model_t m;
    if (rcpp_bitnet_load_h1b(model_path, &m) != RCPP_OK) {
        fprintf(stderr, "failed to load %s\n", model_path); return 1;
    }
    rcpp_tokenizer_t* tok = nullptr;
    if (rcpp_tokenizer_load(tok_path, &tok) != RCPP_OK) {
        fprintf(stderr, "failed to load tokenizer %s\n", tok_path); return 1;
    }

    const int hs = m.hidden_size, is = m.intermediate_size;
    const int nh = m.num_heads, nkv = m.num_kv_heads;
    const int hd = hs / nh, L = m.num_layers, V = m.vocab_size;
    const int max_ctx = 4096;
    const float scale = 1.0f / std::sqrt((float)hd);
    const int BOS = 128000, EOT = 128009;
    const int STOP_A = 128001, STOP_B = 128009;

    auto encode_chunk = [&](const char* s) {
        std::vector<int> buf(4096); size_t n = 0;
        rcpp_tokenizer_encode(tok, s, std::strlen(s), 0, buf.data(), buf.size(), &n);
        buf.resize(std::min(n, buf.size())); return buf;
    };

    // --- Shared scratch buffers (one set; reused across sides per forward pass) ---
    float    *x_fp32;  _Float16 *x_fp16, *normed;
    int8_t   *x_i8;    float    *x_scale_dev;
    float    *q_raw, *k_raw, *v_raw, *o_raw, *gate_raw, *up_raw, *down_raw;
    _Float16 *q_fp16, *k_fp16, *v_fp16, *o_fp16, *gate_fp16, *up_fp16, *down_fp16;
    _Float16 *silu_out;  int8_t *silu_i8;  float *silu_scale_dev;
    float    *logits;    int    *next_tok_dev;
    HIP_OK(hipMalloc(&x_fp32, hs * 4));
    HIP_OK(hipMalloc(&x_fp16, hs * 2));
    HIP_OK(hipMalloc(&normed, hs * 2));
    HIP_OK(hipMalloc(&x_i8,   hs));
    HIP_OK(hipMalloc(&x_scale_dev, 4));
    HIP_OK(hipMalloc(&q_raw,  nh * hd * 4));
    HIP_OK(hipMalloc(&k_raw, nkv * hd * 4));
    HIP_OK(hipMalloc(&v_raw, nkv * hd * 4));
    HIP_OK(hipMalloc(&q_fp16, nh * hd * 2));
    HIP_OK(hipMalloc(&k_fp16, nkv * hd * 2));
    HIP_OK(hipMalloc(&v_fp16, nkv * hd * 2));
    HIP_OK(hipMalloc(&o_raw, hs * 4));  HIP_OK(hipMalloc(&o_fp16, hs * 2));
    HIP_OK(hipMalloc(&gate_raw, is * 4)); HIP_OK(hipMalloc(&up_raw, is * 4));
    HIP_OK(hipMalloc(&gate_fp16, is * 2)); HIP_OK(hipMalloc(&up_fp16, is * 2));
    HIP_OK(hipMalloc(&down_raw, hs * 4)); HIP_OK(hipMalloc(&down_fp16, hs * 2));
    HIP_OK(hipMalloc(&silu_out, is * 2)); HIP_OK(hipMalloc(&silu_i8, is));
    HIP_OK(hipMalloc(&silu_scale_dev, 4));
    HIP_OK(hipMalloc(&logits, V * 4));
    HIP_OK(hipMalloc(&next_tok_dev, 4));

    // --- Per-side KV caches: two independent slabs ---
    struct Side {
        std::vector<_Float16*> K, V;
        int pos = 0;
        std::string name;
        std::string stance;
    };
    Side pro, con;
    pro.K.resize(L); pro.V.resize(L); pro.name = "PRO"; pro.stance = std::string("You are arguing FOR this statement: ") + topic + ". Be concise, direct, fact-based. One paragraph.";
    con.K.resize(L); con.V.resize(L); con.name = "CON"; con.stance = std::string("You are arguing AGAINST this statement: ") + topic + ". Be concise, direct, fact-based. One paragraph.";
    const size_t kv_size = (size_t)max_ctx * nkv * hd * sizeof(_Float16);
    for (int l = 0; l < L; ++l) {
        HIP_OK(hipMalloc(&pro.K[l], kv_size)); HIP_OK(hipMalloc(&pro.V[l], kv_size));
        HIP_OK(hipMalloc(&con.K[l], kv_size)); HIP_OK(hipMalloc(&con.V[l], kv_size));
    }

    // --- One forward_token call, parameterized by which side's cache to touch ---
    auto forward = [&](Side& s, int token_id, int pos) -> int {
        RC_OK(rcpp_embedding_lookup_fp16(m.embedding_dev, token_id, x_fp16, hs, nullptr));
        HIP_OK(hipMemsetAsync(x_fp32, 0, hs * 4, nullptr));
        RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, x_fp16, hs, nullptr));
        for (int l = 0; l < L; ++l) {
            rcpp_bitnet_layer_t& ly = m.layers[l];
            RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, ly.input_norm_dev, normed, m.rms_norm_eps, hs, nullptr));
            RC_OK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs, nullptr));
            float xs; HIP_OK(hipMemcpy(&xs, x_scale_dev, 4, hipMemcpyDeviceToHost));
            RC_OK(rcpp_ternary_gemv_halo(ly.q_packed_dev, x_i8, xs, ly.q_scales_dev, q_raw, nh*hd,  hs, nullptr));
            RC_OK(rcpp_ternary_gemv_halo(ly.k_packed_dev, x_i8, xs, ly.k_scales_dev, k_raw, nkv*hd, hs, nullptr));
            RC_OK(rcpp_ternary_gemv_halo(ly.v_packed_dev, x_i8, xs, ly.v_scales_dev, v_raw, nkv*hd, hs, nullptr));
            RC_OK(rcpp_fp32_to_fp16(q_raw, q_fp16, nh*hd,  nullptr));
            RC_OK(rcpp_fp32_to_fp16(k_raw, k_fp16, nkv*hd, nullptr));
            RC_OK(rcpp_fp32_to_fp16(v_raw, v_fp16, nkv*hd, nullptr));
            RC_OK(rcpp_rope_fp16(q_fp16, pos, m.rope_theta, nh,  hd, nullptr));
            RC_OK(rcpp_rope_fp16(k_fp16, pos, m.rope_theta, nkv, hd, nullptr));
            HIP_OK(hipMemcpy(s.K[l] + (size_t)pos * nkv * hd, k_fp16, nkv*hd*2, hipMemcpyDeviceToDevice));
            HIP_OK(hipMemcpy(s.V[l] + (size_t)pos * nkv * hd, v_fp16, nkv*hd*2, hipMemcpyDeviceToDevice));
            RC_OK(rcpp_kv_cache_attn_decode(q_fp16, s.K[l], s.V[l], o_fp16, nh, nkv, hd, pos+1, scale, nullptr));
            RC_OK(rcpp_rmsnorm_fp16(o_fp16, ly.attn_sub_norm_dev, normed, m.rms_norm_eps, hs, nullptr));
            RC_OK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs, nullptr));
            HIP_OK(hipMemcpy(&xs, x_scale_dev, 4, hipMemcpyDeviceToHost));
            RC_OK(rcpp_ternary_gemv_halo(ly.o_packed_dev, x_i8, xs, ly.o_scales_dev, o_raw, hs, nh*hd, nullptr));
            RC_OK(rcpp_fp32_to_fp16(o_raw, o_fp16, hs, nullptr));
            RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, o_fp16, hs, nullptr));
            RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, ly.post_attn_norm_dev, normed, m.rms_norm_eps, hs, nullptr));
            RC_OK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs, nullptr));
            HIP_OK(hipMemcpy(&xs, x_scale_dev, 4, hipMemcpyDeviceToHost));
            RC_OK(rcpp_ternary_gemv_halo(ly.gate_packed_dev, x_i8, xs, ly.gate_scales_dev, gate_raw, is, hs, nullptr));
            RC_OK(rcpp_ternary_gemv_halo(ly.up_packed_dev,   x_i8, xs, ly.up_scales_dev,   up_raw,   is, hs, nullptr));
            RC_OK(rcpp_fp32_to_fp16(gate_raw, gate_fp16, is, nullptr));
            RC_OK(rcpp_fp32_to_fp16(up_raw,   up_fp16,   is, nullptr));
            RC_OK(rcpp_relu2_glu_rmsnorm_fp16(gate_fp16, up_fp16, ly.ffn_sub_norm_dev, silu_out, m.rms_norm_eps, is, nullptr));
            RC_OK(rcpp_quantize_fp16_to_i8(silu_out, silu_i8, silu_scale_dev, is, nullptr));
            float ss; HIP_OK(hipMemcpy(&ss, silu_scale_dev, 4, hipMemcpyDeviceToHost));
            RC_OK(rcpp_ternary_gemv_halo(ly.down_packed_dev, silu_i8, ss, ly.down_scales_dev, down_raw, hs, is, nullptr));
            RC_OK(rcpp_fp32_to_fp16(down_raw, down_fp16, hs, nullptr));
            RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, down_fp16, hs, nullptr));
        }
        RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, m.final_norm_weight_dev, normed, m.rms_norm_eps, hs, nullptr));
        RC_OK(rcpp_fp16_gemv(m.embedding_dev, normed, logits, V, hs, nullptr));
        RC_OK(rcpp_argmax_fp32(logits, next_tok_dev, V, nullptr));
        int next_tok; HIP_OK(hipMemcpy(&next_tok, next_tok_dev, 4, hipMemcpyDeviceToHost));
        return next_tok;
    };

    // --- Prime each side with its system prompt + BOS ---
    auto prime = [&](Side& s) -> int {
        std::vector<int> ids = { BOS };
        auto sys = encode_chunk((std::string("System: ") + s.stance).c_str());
        ids.insert(ids.end(), sys.begin(), sys.end());
        ids.push_back(EOT);
        for (int t : ids) { forward(s, t, s.pos++); (void)hipDeviceSynchronize(); }
        return 0;
    };
    (void)prime(pro); (void)prime(con);

    // --- Generate one turn on a side given the opposing last statement ---
    std::vector<char> stream_buf(8192);
    auto take_turn = [&](Side& s, const std::string& opponent_text) -> std::string {
        // Feed the opponent's text as a user turn + "Assistant: " prefix.
        auto user_ids = encode_chunk((std::string("User: ") + opponent_text).c_str());
        user_ids.push_back(EOT);
        auto pre     = encode_chunk("Assistant: ");
        std::vector<int> turn;
        turn.insert(turn.end(), user_ids.begin(), user_ids.end());
        turn.insert(turn.end(), pre.begin(),      pre.end());
        int last = 0;
        for (int t : turn) { last = forward(s, t, s.pos++); (void)hipDeviceSynchronize(); }

        // Greedy decode until EOS or max_tok.
        printf("[%s] ", s.name.c_str()); fflush(stdout);
        std::vector<int> gen;
        size_t printed = 0;
        int cur = last;
        for (int i = 0; i < max_tok; ++i) {
            int nt = forward(s, cur, s.pos++);
            (void)hipDeviceSynchronize();
            gen.push_back(nt);
            size_t tlen = 0;
            rcpp_tokenizer_decode(tok, gen.data(), gen.size(),
                                  stream_buf.data(), stream_buf.size(), &tlen);
            tlen = std::min(tlen, stream_buf.size());
            if (tlen > printed) {
                fwrite(stream_buf.data() + printed, 1, tlen - printed, stdout);
                fflush(stdout);
                printed = tlen;
            }
            if (nt == STOP_A || nt == STOP_B) break;
            cur = nt;
        }
        printf("\n\n"); fflush(stdout);
        std::string out(stream_buf.data(), printed);
        // Strip trailing <|eot_id|> glyphs if present.
        while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
        return out;
    };

    // --- The debate ---
    printf("═════════════════════════════════════════════════════════════════\n");
    printf("  TOPIC: %s\n", topic);
    printf("═════════════════════════════════════════════════════════════════\n\n");
    std::string last_pro = std::string("The proposition stands: ") + topic + ". Defend it.";
    std::string last_con;
    for (int r = 0; r < rounds; ++r) {
        printf("--- Round %d ---\n", r + 1);
        last_con = take_turn(con, last_pro);
        last_pro = take_turn(pro, last_con);
    }

    // --- Cleanup ---
    for (int l = 0; l < L; ++l) {
        hipFree(pro.K[l]); hipFree(pro.V[l]);
        hipFree(con.K[l]); hipFree(con.V[l]);
    }
    rcpp_tokenizer_free(tok);
    rcpp_bitnet_free(&m);
    return 0;
}
