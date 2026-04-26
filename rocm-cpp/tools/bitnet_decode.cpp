// bitnet_decode — thin CLI / HTTP wrapper around rocm_cpp::Engine.
//
// All decode logic (KV cache, sampler, Sherry/Bonsai/Medusa dispatch, KV
// compression knobs) lives in the Engine class. This file only handles
// argv parsing, prompt assembly (text / chat / server / repl / ppl / bench
// modes), and routing to either stdout or an httplib server.
//
// Pre-refactor (2026-04-25) this was 1470 LOC. Hot-path forward / sampler /
// KV state was extracted into src/engine.cpp + include/rocm_cpp/engine.h
// so the same primitives can be embedded inside lemond's C++ server with
// no code duplication. Every CLI flag from the prior monolith is preserved
// here — new logic should land in Engine, not in this wrapper.
//
// Modes:
//   bitnet_decode <model.h1b>                                # smoke
//   bitnet_decode <model.h1b> "<prompt-text>" <num_new>      # legacy text
//   bitnet_decode <model.h1b> --text "<prompt>" <num_new>    # explicit text
//   bitnet_decode <model.h1b> --chat "<msg>" <num_new> ["<sys>"]
//   bitnet_decode <model.h1b> --server <port> [<default_max>]
//   bitnet_decode <model.h1b> --repl [max_per_turn]
//   bitnet_decode <model.h1b> --ppl <file.txt> [max_tokens]
//   bitnet_decode --model <m> --ctx <N> --iters <N>          # bench mode
// Trailing flags: --temp / --top-k / --top-p / --rep-penalty / --rep-last-n
//   / --seed / --stop / --bind / --tokenizer / --kv-int8 / --kv-rotor
//   / --max-tokens / --prompt
//
// Bench mode (--model + --ctx + --iters) prints ONE LINE of "<tok/s>" to
// stdout; sherry-bench.sh parses that. All other output goes to stderr.
//
// PPL mode prints a JSON line with mean_nll / perplexity to stdout.

#include "rocm_cpp/engine.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace {

struct ParsedArgs {
    // Positional / mode
    std::string model_path;
    std::string mode;             // "", "text", "chat", "server", "repl", "ppl", "bench"
    std::string prompt_text;      // for text/chat/ppl
    std::string system_msg;       // for chat
    int         num_new = 16;

    // Tokenizer override
    std::string tokenizer_path;

    // Sampler
    rocm_cpp::GenerateOptions opts;

    // Server
    int         server_port = 0;
    std::string server_bind = "127.0.0.1";

    // KV cfg
    bool kv_int8  = false;
    bool kv_rotor = false;

    // Bench-mode
    int  bench_ctx   = 0;
    int  bench_iters = 0;
};

// Extract trailing named flags from argv. Returns the index of the first
// flag arg seen (or argc if none) so the caller knows where positional args
// end. Mutates `out` in place. Mirrors the dual-pass scan bitnet_decode used
// pre-refactor (positional args MUST come before any --flag).
int parse_flags(int argc, char** argv, ParsedArgs& out) {
    int positional_end = argc;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        bool consumed = true;
        if      (a == "--model"       && i + 1 < argc) { out.model_path   = argv[++i]; }
        else if (a == "--ctx"         && i + 1 < argc) { out.bench_ctx    = std::atoi(argv[++i]); }
        else if (a == "--iters"       && i + 1 < argc) { out.bench_iters  = std::atoi(argv[++i]); }
        else if (a == "--prompt"      && i + 1 < argc) { out.prompt_text  = argv[++i]; out.mode = "text"; }
        else if (a == "--max-tokens"  && i + 1 < argc) { out.num_new      = std::atoi(argv[++i]); }
        else if (a == "--max-ppl-tokens" && i + 1 < argc) { out.num_new   = std::atoi(argv[++i]); }
        else if (a == "--stop"        && i + 1 < argc) { out.opts.stop_sequences.emplace_back(argv[++i]); }
        else if (a == "--temp"        && i + 1 < argc) { out.opts.temperature = (float)std::atof(argv[++i]); }
        else if (a == "--top-k"       && i + 1 < argc) { out.opts.top_k       = std::atoi(argv[++i]); }
        else if (a == "--top-p"       && i + 1 < argc) { out.opts.top_p       = (float)std::atof(argv[++i]); }
        else if (a == "--rep-penalty" && i + 1 < argc) { out.opts.rep_penalty = (float)std::atof(argv[++i]); }
        else if (a == "--rep-last-n"  && i + 1 < argc) { out.opts.rep_last_n  = std::atoi(argv[++i]); }
        else if (a == "--seed"        && i + 1 < argc) { out.opts.seed        = (uint64_t)std::atoll(argv[++i]); }
        else if (a == "--bind"        && i + 1 < argc) { out.server_bind  = argv[++i]; }
        else if (a == "--tokenizer"   && i + 1 < argc) { out.tokenizer_path = argv[++i]; }
        else if (a == "--kv-int8")                     { out.kv_int8     = true; }
        else if (a == "--kv-rotor")                    { out.kv_rotor    = true; }
        else { consumed = false; }
        if (consumed && positional_end == argc) {
            // First --flag we see caps the positional scan.
            positional_end = i;
        }
    }
    return positional_end;
}

std::string apply_chat_template(const std::string& user_msg,
                                const std::string& system_msg) {
    // Best-effort BitNet-2B-4T chat template, encoded as text (literal
    // <|eot_id|>). The byte-level BPE tokenizer emits the bytes of this
    // string verbatim — perfect specials would need a regex pre-tokenizer
    // we don't have. Smoke-test-equivalent on hello-world prompts.
    std::string out;
    if (!system_msg.empty()) {
        out += "System: " + system_msg + "<|eot_id|>";
    }
    out += "User: " + user_msg + "<|eot_id|>Assistant: ";
    return out;
}

// Llama-3 special-token IDs. The byte-level BPE encoder in rcpp_tokenizer
// does NOT recognize the textual <|...|> markers, so we inject them as raw
// IDs around the content tokens. BitNet-b1.58-2B-4T inherits the Llama-3
// vocabulary so these constants are bit-exact for that model family.
constexpr int LLAMA3_BOS              = 128000;
constexpr int LLAMA3_EOT              = 128009;
constexpr int LLAMA3_START_HEADER     = 128006;
constexpr int LLAMA3_END_HEADER       = 128007;

// Build the Llama-3-shape chat-template token stream for an OpenAI-format
// `messages` array.
//
//   <|begin_of_text|>
//     for each message:
//       <|start_header_id|> <role-bytes> <|end_header_id|> "\n\n"
//       <content-bytes> <|eot_id|>
//     <|start_header_id|> "assistant" <|end_header_id|> "\n\n"
//
// Special tokens are pushed as IDs; role / content / "\n\n" go through the
// engine's byte-level encoder.
std::vector<int> build_llama3_chat_tokens(rocm_cpp::Engine& engine,
                                          const nlohmann::json& messages) {
    std::vector<int> ids;
    ids.reserve(64);
    const int bos = engine.bos_id();
    ids.push_back(bos > 0 ? bos : LLAMA3_BOS);

    auto push_encoded = [&](const std::string& s) {
        if (s.empty()) return;
        auto e = engine.tokenize(s, /*add_bos=*/false);
        ids.insert(ids.end(), e.begin(), e.end());
    };

    if (messages.is_array()) {
        for (const auto& m : messages) {
            std::string role    = m.value("role",    std::string("user"));
            std::string content = m.value("content", std::string(""));
            ids.push_back(LLAMA3_START_HEADER);
            push_encoded(role);
            ids.push_back(LLAMA3_END_HEADER);
            push_encoded("\n\n");
            push_encoded(content);
            ids.push_back(LLAMA3_EOT);
        }
    }

    // Assistant priming header — model resumes at "\n\n" boundary.
    ids.push_back(LLAMA3_START_HEADER);
    push_encoded("assistant");
    ids.push_back(LLAMA3_END_HEADER);
    push_encoded("\n\n");
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    ParsedArgs args;
    int positional_end = parse_flags(argc, argv, args);

    // Resolve model path: --model flag wins, else argv[1].
    if (args.model_path.empty()) {
        if (argc > 1 && std::string(argv[1]).rfind("--", 0) != 0) {
            args.model_path = argv[1];
        } else {
            args.model_path = "/home/bcloud/halo-1bit/models/halo-1bit-2b.h1b";
        }
    }

    const bool bench_mode =
        (args.bench_ctx > 0 && args.bench_iters > 0 && !args.model_path.empty());
    if (bench_mode) args.mode = "bench";

    // Resolve mode + positional prompt args. argv[1] = model, argv[2] = mode/prompt.
    // Skip if --prompt (already in args.prompt_text) or bench mode.
    int pos = 2;
    if (args.mode.empty() && pos < positional_end && pos < argc) {
        std::string a2 = argv[pos];
        if (a2 == "--text") {
            args.mode = "text";
            ++pos;
            if (pos < positional_end) { args.prompt_text = argv[pos++]; }
            if (pos < positional_end) { args.num_new     = std::atoi(argv[pos++]); }
            if (pos < positional_end && argv[pos][0] != '-') {
                args.tokenizer_path = argv[pos++];
            }
        } else if (a2 == "--chat") {
            args.mode = "chat";
            ++pos;
            if (pos < positional_end) { args.prompt_text = argv[pos++]; }
            else { std::fprintf(stderr, "usage: --chat \"<msg>\" <num_new> [\"<sys>\"]\n"); return 1; }
            if (pos < positional_end) { args.num_new     = std::atoi(argv[pos++]); }
            if (pos < positional_end) { args.system_msg  = argv[pos++]; }
        } else if (a2 == "--server") {
            args.mode = "server";
            ++pos;
            if (pos < positional_end) { args.server_port = std::atoi(argv[pos++]); }
            else                       { args.server_port = 8080; }
            if (pos < positional_end) { args.num_new     = std::atoi(argv[pos++]); }
            else if (args.num_new == 16) { args.num_new = 256; }
        } else if (a2 == "--repl") {
            args.mode = "repl";
            ++pos;
            if (pos < positional_end) { args.num_new = std::atoi(argv[pos++]); }
            else                       { args.num_new = 256; }
            if (pos < positional_end && argv[pos][0] != '-') {
                args.tokenizer_path = argv[pos++];
            }
        } else if (a2 == "--ppl") {
            args.mode = "ppl";
            ++pos;
            if (pos < positional_end) { args.prompt_text = argv[pos++]; }
            else { std::fprintf(stderr, "usage: --ppl <file> [max_tokens]\n"); return 1; }
            if (pos < positional_end) { args.num_new = std::atoi(argv[pos++]); }
            else                       { args.num_new = 4095; }
        } else {
            // Legacy positional: <model> <int_or_text> <num_new>
            // Treat anything that's not a pure integer as raw text input.
            args.mode        = "text";
            args.prompt_text = a2;
            ++pos;
            if (pos < positional_end) { args.num_new = std::atoi(argv[pos++]); }
        }
    }

    if (args.mode.empty()) {
        // Bare `bitnet_decode <model>` — short greedy smoke test.
        args.mode        = "text";
        args.prompt_text = " ";
        args.num_new     = 16;
    }
    // num_new wins over the GenerateOptions default; --max-tokens already
    // wrote into num_new during parse_flags.
    if (args.num_new > 0) args.opts.max_tokens = args.num_new;

    // ---- Build Engine ----
    rocm_cpp::Config cfg;
    cfg.kv_int8  = args.kv_int8;
    cfg.kv_rotor = args.kv_rotor;
    cfg.max_context = (args.mode == "repl" || args.mode == "server") ? 4096
                                                                     : 4096;
    rocm_cpp::Engine engine(cfg);
    try {
        engine.load(args.model_path, args.tokenizer_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load failed: %s\n", e.what());
        return 1;
    }

    if (args.opts.temperature > 0.0f) {
        std::fprintf(stderr,
                     "[sampler] temp=%.3f top_k=%d top_p=%.3f rep=%.3f/%d "
                     "seed=%llu\n",
                     args.opts.temperature, args.opts.top_k, args.opts.top_p,
                     args.opts.rep_penalty, args.opts.rep_last_n,
                     (unsigned long long)args.opts.seed);
    }

    // ─── Bench mode ───────────────────────────────────────────────────────
    if (args.mode == "bench") {
        // Synthesize a long prompt of single-byte chars so the BPE tokenizer
        // produces ~bench_ctx tokens. " " repeats are mapped to space tokens
        // so this is roughly token-stable across tokenizers.
        std::string synth(args.bench_ctx, ' ');
        rocm_cpp::GenerateOptions opts = args.opts;
        opts.max_tokens = args.bench_iters;
        try {
            auto r = engine.generate(synth, opts, {});
            std::printf("%.2f\n", r.decode_tok_per_sec);
            std::fflush(stdout);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "bench failed: %s\n", e.what());
            return 1;
        }
    }

    // ─── PPL mode ─────────────────────────────────────────────────────────
    // Wikitext-style perplexity. Engine::compute_nll returns per-position NLL
    // for a single contiguous prefill (cap = max_context). For longer corpora
    // we run a fixed-stride sliding window: window=ctx, stride=ctx/2, scoring
    // only the second half of each window after the first (standard
    // wikitext-103 protocol from llama.cpp / Hugging Face evaluate).
    if (args.mode == "ppl") {
        std::ifstream f(args.prompt_text);
        if (!f) {
            std::fprintf(stderr, "cannot open --ppl file: %s\n",
                         args.prompt_text.c_str());
            return 1;
        }
        std::stringstream ss; ss << f.rdbuf();
        std::string text = ss.str();

        std::vector<int> all_ids = engine.tokenize(text, /*add_bos=*/true);
        if ((int)all_ids.size() < 2) {
            std::fprintf(stderr, "[ppl] corpus too small (%zu tokens)\n",
                         all_ids.size());
            return 1;
        }

        // --max-ppl-tokens cap (re-uses --max-tokens / num_new). 0 / negative
        // => no cap. Default 4095 from the legacy positional default.
        int cap = args.num_new;
        if (cap > 0 && (int)all_ids.size() > cap) {
            all_ids.resize(cap);
        }

        const int win = 2048;     // sliding window length
        const int stride = 1024;  // step
        const int total = (int)all_ids.size();

        double sum_nll = 0.0;
        long   count   = 0;
        try {
            if (total <= win) {
                auto nll = engine.compute_nll(all_ids);
                for (float v : nll) { sum_nll += v; ++count; }
            } else {
                // First window: score every position.
                {
                    std::vector<int> chunk(all_ids.begin(),
                                           all_ids.begin() + win);
                    auto nll = engine.compute_nll(chunk);
                    for (float v : nll) { sum_nll += v; ++count; }
                }
                // Subsequent windows: only score the new (non-overlapping)
                // suffix to avoid double-counting.
                int start = stride;
                while (start + win <= total) {
                    std::vector<int> chunk(all_ids.begin() + start,
                                           all_ids.begin() + start + win);
                    auto nll = engine.compute_nll(chunk);
                    // nll[t] corresponds to predicting chunk[t+1] from
                    // chunk[0..t]. Only take the second half (positions
                    // win-stride .. win-1 in 0-indexed nll → indices
                    // (win - stride - 1) .. (win - 2) inclusive).
                    int score_start = win - stride - 1;
                    if (score_start < 0) score_start = 0;
                    for (int i = score_start; i < (int)nll.size(); ++i) {
                        sum_nll += nll[i];
                        ++count;
                    }
                    start += stride;
                }
                // Tail: final partial window so we score every token once.
                if (start < total - 1) {
                    int tail_len = total - start;
                    std::vector<int> chunk(all_ids.begin() + start,
                                           all_ids.end());
                    auto nll = engine.compute_nll(chunk);
                    int score_start = win - stride - 1;
                    if (score_start < 0) score_start = 0;
                    if (score_start >= (int)nll.size())
                        score_start = (int)nll.size();
                    for (int i = score_start; i < (int)nll.size(); ++i) {
                        sum_nll += nll[i];
                        ++count;
                    }
                    (void)tail_len;
                }
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "ppl failed: %s\n", e.what());
            return 1;
        }

        if (count <= 0) {
            std::fprintf(stderr, "[ppl] no tokens scored\n");
            return 1;
        }
        const double mean_nll = sum_nll / (double)count;
        const double ppl = std::exp(mean_nll);
        std::fprintf(stderr,
                     "[ppl] file=%s tokens=%ld nll=%.4f ppl=%.4f\n",
                     args.prompt_text.c_str(), count, mean_nll, ppl);
        std::printf("{\"file\":\"%s\",\"tokens\":%ld,\"nll\":%.6f,\"ppl\":%.6f}\n",
                    args.prompt_text.c_str(), count, mean_nll, ppl);
        std::fflush(stdout);
        return (ppl > 100.0) ? 2 : 0;
    }

    // ─── Text mode (default + --text + bare-prompt) ───────────────────────
    if (args.mode == "text") {
        try {
            auto on_token = [](const std::string& delta) -> bool {
                std::fwrite(delta.data(), 1, delta.size(), stdout);
                std::fflush(stdout);
                return true;
            };
            auto t0 = std::chrono::steady_clock::now();
            auto r = engine.generate(args.prompt_text, args.opts, on_token);
            auto t1 = std::chrono::steady_clock::now();
            std::printf("\n");
            std::fflush(stdout);
            double dt =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::fprintf(stderr,
                         "[bitnet_decode] %d prompt + %d new tok in %.2f ms "
                         "(prefill %.1f tok/s, decode %.1f tok/s) [%s]\n",
                         r.prompt_tokens, r.completion_tokens, dt,
                         r.prefill_tok_per_sec, r.decode_tok_per_sec,
                         r.finish_reason.c_str());
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "generate failed: %s\n", e.what());
            return 1;
        }
    }

    // ─── Chat mode ────────────────────────────────────────────────────────
    if (args.mode == "chat") {
        std::string prompt =
            apply_chat_template(args.prompt_text, args.system_msg);
        // Default stop for chat — model often emits <|eot_id|> as text.
        if (args.opts.stop_sequences.empty()) {
            args.opts.stop_sequences.push_back("<|eot_id|>");
        }
        try {
            auto on_token = [](const std::string& delta) -> bool {
                std::fwrite(delta.data(), 1, delta.size(), stdout);
                std::fflush(stdout);
                return true;
            };
            auto r = engine.generate(prompt, args.opts, on_token);
            std::printf("\n");
            std::fflush(stdout);
            std::fprintf(stderr,
                         "[chat] %d new tok @ %.1f tok/s [%s]\n",
                         r.completion_tokens, r.decode_tok_per_sec,
                         r.finish_reason.c_str());
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "chat failed: %s\n", e.what());
            return 1;
        }
    }

    // ─── REPL mode ────────────────────────────────────────────────────────
    if (args.mode == "repl") {
        std::fprintf(stderr,
                     "[repl] %d max tokens/turn. Ctrl-D or 'quit' to exit.\n"
                     "       (KV cache resets between turns in this build — "
                     "session-persistent KV is a follow-up.)\n",
                     args.num_new);
        std::string line;
        while (true) {
            std::fprintf(stderr, "\n> "); std::fflush(stderr);
            if (!std::getline(std::cin, line)) break;
            if (line == "quit" || line == "exit") break;
            if (line.empty()) continue;
            std::string prompt = apply_chat_template(line, /*system=*/"");
            rocm_cpp::GenerateOptions opts = args.opts;
            opts.max_tokens = args.num_new;
            if (opts.stop_sequences.empty()) {
                opts.stop_sequences.push_back("<|eot_id|>");
            }
            try {
                auto on_token = [](const std::string& delta) -> bool {
                    std::fwrite(delta.data(), 1, delta.size(), stdout);
                    std::fflush(stdout);
                    return true;
                };
                (void)engine.generate(prompt, opts, on_token);
                std::printf("\n"); std::fflush(stdout);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[repl] generate failed: %s\n", e.what());
            }
        }
        return 0;
    }

    // ─── HTTP server mode ─────────────────────────────────────────────────
    if (args.mode == "server") {
        std::fprintf(stderr,
                     "[server] OpenAI-compat API on %s:%d (default max_tokens=%d)\n",
                     args.server_bind.c_str(), args.server_port, args.num_new);
        if (args.server_bind == "0.0.0.0") {
            std::fprintf(stderr,
                         "[server] WARNING: bound to 0.0.0.0 — publicly reachable.\n"
                         "[server] WARNING: no auth, no TLS, no rate limit.\n");
        }

        // Engine itself is not thread-safe across generate() calls — every
        // request grabs this mutex. Lemond's WrappedServer uses the same
        // pattern (per-instance is_loading_ mutex).
        std::mutex gen_mu;

        httplib::Server svr;

        svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("OK\n", "text/plain");
        });

        svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
            nlohmann::json j = {
                {"object", "list"},
                {"data", nlohmann::json::array({{
                    {"id", "bitnet-b1.58-2b-4t"},
                    {"object", "model"},
                    {"owned_by", "halo-ai"}}})}};
            res.set_content(j.dump(), "application/json");
        });

        svr.Post("/v1/chat/completions",
                 [&engine, &gen_mu, default_max = args.num_new,
                  default_opts = args.opts]
                 (const httplib::Request& req, httplib::Response& res) {
            auto lk = std::make_shared<std::unique_lock<std::mutex>>(gen_mu);
            nlohmann::json j;
            try { j = nlohmann::json::parse(req.body); }
            catch (const std::exception& e) {
                res.status = 400;
                res.set_content(std::string("{\"error\":\"bad json: ") +
                                e.what() + "\"}",
                                "application/json");
                return;
            }

            // Build prompt as a Llama-3 token-id sequence: BOS + per-message
            // <|start_header_id|> role <|end_header_id|>\n\n content <|eot_id|>
            // + assistant priming header. Special tokens are pushed as IDs
            // (128000/128006/128007/128009) — the byte-level BPE encoder does
            // NOT recognize the textual <|...|> markers, so without this the
            // tokenizer would emit ~10 raw bytes per marker instead of the
            // single special-id, causing some models (sherry-cpp) to produce
            // empty content because they were never trained to predict the
            // byte-pattern.
            std::vector<int> prompt_ids =
                build_llama3_chat_tokens(engine, j["messages"]);

            int   req_max     = j.value("max_tokens", default_max);
            bool  req_stream  = j.value("stream", false);
            std::string model_name =
                j.value("model", std::string("bitnet-b1.58-2b-4t"));

            rocm_cpp::GenerateOptions opts = default_opts;
            opts.max_tokens = req_max;
            opts.temperature = j.value("temperature", default_opts.temperature);
            opts.top_p       = j.value("top_p",       default_opts.top_p);
            opts.rep_penalty =
                j.value("frequency_penalty", 0.0f) + 1.0f;
            if (opts.rep_penalty < 1.0f) opts.rep_penalty = 1.0f;
            // No literal-text stop needed — Engine treats id 128009 as a
            // hard stop token already. Leaving stop_sequences empty avoids
            // a pointless suffix-match pass per decode step.

            const std::string chat_id =
                "chatcmpl-" +
                std::to_string((long)std::chrono::steady_clock::now()
                                   .time_since_epoch().count());
            const long created = (long)std::time(nullptr);

            // ─── Streaming ───
            if (req_stream) {
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [chat_id, created, model_name, opts, prompt_ids,
                     &engine, lk](size_t, httplib::DataSink& sink) {
                        // Initial role chunk
                        nlohmann::json first = {
                            {"id", chat_id},
                            {"object", "chat.completion.chunk"},
                            {"created", created},
                            {"model", model_name},
                            {"choices", nlohmann::json::array({{
                                {"index", 0},
                                {"delta", {{"role", "assistant"}}},
                                {"finish_reason", nullptr}}})}};
                        std::string l =
                            "data: " + first.dump() + "\n\n";
                        sink.write(l.data(), l.size());

                        auto fire = [&](const std::string& delta) -> bool {
                            nlohmann::json chunk = {
                                {"id", chat_id},
                                {"object", "chat.completion.chunk"},
                                {"created", created},
                                {"model", model_name},
                                {"choices", nlohmann::json::array({{
                                    {"index", 0},
                                    {"delta", {{"content", delta}}},
                                    {"finish_reason", nullptr}}})}};
                            std::string line =
                                "data: " + chunk.dump() + "\n\n";
                            sink.write(line.data(), line.size());
                            return true;
                        };
                        try {
                            (void)engine.generate_from_tokens(
                                prompt_ids, opts, fire);
                        } catch (const std::exception& e) {
                            std::fprintf(stderr,
                                         "[server] generate failed: %s\n",
                                         e.what());
                        }
                        nlohmann::json fin = {
                            {"id", chat_id},
                            {"object", "chat.completion.chunk"},
                            {"created", created},
                            {"model", model_name},
                            {"choices", nlohmann::json::array({{
                                {"index", 0},
                                {"delta", nlohmann::json::object()},
                                {"finish_reason", "stop"}}})}};
                        std::string lf =
                            "data: " + fin.dump() + "\n\n";
                        sink.write(lf.data(), lf.size());
                        std::string done = "data: [DONE]\n\n";
                        sink.write(done.data(), done.size());
                        sink.done();
                        return true;
                    });
                return;
            }

            // ─── Non-streaming ───
            rocm_cpp::GenerateResult r;
            auto t0 = std::chrono::steady_clock::now();
            try {
                r = engine.generate_from_tokens(prompt_ids, opts, {});
            } catch (const std::exception& e) {
                res.status = 500;
                nlohmann::json err = {{"error", {
                    {"message", e.what()},
                    {"type", "internal_error"}}}};
                res.set_content(err.dump(), "application/json");
                return;
            }
            auto t1 = std::chrono::steady_clock::now();
            double dt_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Final <|eot_id|> stripping happens inside Engine already.
            nlohmann::json resp = {
                {"id", chat_id},
                {"object", "chat.completion"},
                {"created", created},
                {"model", model_name},
                {"choices", nlohmann::json::array({{
                    {"index", 0},
                    {"message",
                     {{"role", "assistant"}, {"content", r.text}}},
                    {"finish_reason", r.finish_reason}}})},
                {"usage", {
                    {"prompt_tokens",     r.prompt_tokens},
                    {"completion_tokens", r.completion_tokens},
                    {"total_tokens",
                     r.prompt_tokens + r.completion_tokens},
                    {"latency_ms",        dt_ms}}}};
            res.set_content(resp.dump(), "application/json");
        });

        std::fprintf(stderr, "[server] listening on %s:%d\n",
                     args.server_bind.c_str(), args.server_port);
        svr.listen(args.server_bind.c_str(), args.server_port);
        return 0;
    }

    std::fprintf(stderr, "unknown mode: %s\n", args.mode.c_str());
    return 1;
}
