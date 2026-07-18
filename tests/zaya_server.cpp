// zaya_server.cpp — Pure C++ multi-model, multi-backend inference server.
//
// ONE BINARY. Zero Python. Zero Rust at runtime.
//
// Auto-detects available hardware (ROCm HIP > Vulkan > NPU > CPU fallback).
// Auto-detects model architecture from .h1b header or model manifest.
// TokenRouter dispatches to the best backend per request.
// 6 routing strategies: auto, cascade, spec_decode, content, parallel_moe, passthrough.
// OpenAI-compatible API: POST /v1/chat/completions
//
// Build: cmake --build . --target zaya_server -j8
// Run:   ./build/zaya_server --model model.h1b --port 8088

#include "backends/backend.h"
#include "backends/token_router.h"
#include "rocm_cpp/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

#include <httplib.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// ─── .h1b header auto-detection (no external deps) ────────────────
static bool detect_from_h1b(const std::string& path, ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4];
    f.read(magic, 4);
    if (std::strncmp(magic, "H1B", 3) != 0) return false;
    int32_t version;
    f.read(reinterpret_cast<char*>(&version), 4);
    if (version < 1 || version > 5) return false;
    int32_t hdr[9];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    cfg.hidden_size       = hdr[0];
    cfg.intermediate_size = hdr[1];
    cfg.num_layers        = hdr[2];
    cfg.num_heads         = hdr[3];
    cfg.num_kv_heads      = hdr[4];
    cfg.vocab_size        = hdr[5];
    cfg.max_seq_len       = hdr[6];
    cfg.head_dim          = cfg.hidden_size / cfg.num_heads;
    cfg.num_experts       = 16;
    cfg.num_experts_top   = 2;
    cfg.router_hidden     = 256;
    if (version >= 2) {
        float extras[2];
        f.read(reinterpret_cast<char*>(extras), sizeof(extras));
        cfg.rope_theta   = extras[0] > 0 ? extras[0] : 500000.0f;
        cfg.rms_norm_eps = extras[1] > 0 ? extras[1] : 1e-5f;
    }
    auto slash = path.find_last_of('/');
    cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
    cfg.model_path = path;
    fprintf(stderr, "  Auto-detected from .h1b: %s\n", cfg.model_name.c_str());
    fprintf(stderr, "    hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d\n",
            cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads,
            cfg.head_dim, cfg.vocab_size);
    return true;
}

static bool detect_from_manifest(const std::string& path, ModelConfig& cfg) {
    std::ifstream f(path);
    if (!f) return false;
    try {
        json j = json::parse(f);
        cfg.hidden_size       = j.value("hidden_size", 2048);
        cfg.num_heads         = j.value("num_heads", 16);
        cfg.num_kv_heads      = j.value("num_kv_heads", 2);
        cfg.head_dim          = j.value("head_dim", 128);
        cfg.num_layers        = j.value("num_layers", 40);
        cfg.vocab_size        = j.value("vocab_size", 262272);
        cfg.intermediate_size = j.value("intermediate_size", 2048);
        cfg.max_seq_len       = j.value("max_seq_len", 2048);
        cfg.num_experts       = j.value("num_experts", 16);
        cfg.router_hidden     = j.value("router_hidden", 256);
        cfg.num_experts_top   = j.value("num_experts_top", 2);
        cfg.rope_theta        = j.value("rope_theta", 500000.0f);
        cfg.rms_norm_eps      = j.value("rms_norm_eps", 1e-5f);
        cfg.model_name        = j.value("name", std::string());
        cfg.weights_dir       = j.value("weights_dir", std::string());
        if (cfg.model_name.empty()) {
            auto slash = path.find_last_of('/');
            cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
        }
        if (cfg.weights_dir.empty()) cfg.weights_dir = "/tmp/zaya_weights/";
        fprintf(stderr, "  Loaded manifest: %s\n", cfg.model_name.c_str());
        fprintf(stderr, "    hidden=%d layers=%d heads=%d vocab=%d\n",
                cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.vocab_size);
        return true;
    } catch (const json::exception& e) {
        fprintf(stderr, "  Manifest parse error: %s\n", e.what());
        return false;
    }
}

static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 32) { char buf[8]; snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c); out += buf; }
                else out += c;
        }
    }
    return out;
}

static std::string build_chatml(const std::string& body) {
    try {
        json j = json::parse(body);

        // Check messages array
        if (j.contains("messages") && j["messages"].is_array()) {
            std::string result;
            for (auto& msg : j["messages"]) {
                std::string role = msg.value("role", std::string());
                std::string content = msg.value("content", std::string());
                if (!role.empty() && !content.empty())
                    result += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
            }
            if (!result.empty())
                result += "<|im_start|>assistant\n";
            return result;
        }

        // Fallback to prompt field
        if (j.contains("prompt") && j["prompt"].is_string()) {
            std::string prompt = j["prompt"];
            if (!prompt.empty())
                return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        }
    } catch (const json::exception& e) {
        fprintf(stderr, "  ChatML parse error: %s\n", e.what());
    }
    return "";
}

struct SimpleTokenizer {
    int bos_id = 2;
    int eos_id = 106;
    bool use_bpe = false;
    rcpp_tokenizer_t* bpe_tok = nullptr;

    ~SimpleTokenizer() { if (bpe_tok) rcpp_tokenizer_free(bpe_tok); }

    bool load_htok(const std::string& path) {
        rcpp_tokenizer_t* tok = nullptr;
        rcpp_status_t st = rcpp_tokenizer_load(path.c_str(), &tok);
        if (st == RCPP_OK && tok) {
            bpe_tok = tok;
            use_bpe = true;
            bos_id = rcpp_tokenizer_bos_id(bpe_tok);
            eos_id = rcpp_tokenizer_eos_id(bpe_tok);
            fprintf(stderr, "  BPE tokenizer: BOS=%d EOS=%d\n", bos_id, eos_id);
            return true;
        }
        return false;
    }

    std::vector<int> encode(const std::string& text) {
        if (use_bpe && bpe_tok) {
            std::vector<int> r(4096);
            size_t out_n = 0;
            rcpp_status_t st = rcpp_tokenizer_encode(bpe_tok, text.c_str(), text.size(),
                                                      1, r.data(), r.size(), &out_n);
            if (st == RCPP_OK && out_n > 0) {
                r.resize(out_n);
                return r;
            }
            return {bos_id};
        }
        std::vector<int> r = {bos_id};
        for (unsigned char c : text) {
            if (c >= 32 && c <= 126) r.push_back((int)c + 100);
            else if (c != 0) r.push_back((int)c + 200);
        }
        return r;
    }

    std::string decode(const std::vector<int>& tokens) {
        if (use_bpe && bpe_tok) {
            std::string r(4096, '\0');
            size_t out_len = 0;
            rcpp_status_t st = rcpp_tokenizer_decode(bpe_tok, tokens.data(), tokens.size(),
                                                      r.data(), r.size(), &out_len);
            if (st == RCPP_OK && out_len > 0) {
                r.resize(out_len);
                return r;
            }
            return "";
        }
        std::string r;
        for (int v : tokens) {
            if (v == bos_id || v == eos_id) continue;
            if (v > 100 && v < 200) r += (char)(v - 100);
            else if (v > 200 && v < 456) r += (char)(v - 200);
            else { r += '['; r += std::to_string(v); r += ']'; }
        }
        return r;
    }
};

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int port = 8088;
    std::string model_arg, manifest_arg, weights_dir = "/tmp/zaya_weights/", lora_path;
    RouteStrategy strategy = RouteStrategy::AUTO;

    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        if (a == "--port" && i+1 < argc) port = atoi(argv[++i]);
        else if (a == "--model" && i+1 < argc) model_arg = argv[++i];
        else if (a == "--manifest" && i+1 < argc) manifest_arg = argv[++i];
        else if (a == "--weights-dir" && i+1 < argc) weights_dir = argv[++i];
        else if (a == "--strategy" && i+1 < argc) {
            std::string s(argv[++i]);
            if (s == "auto") strategy = RouteStrategy::AUTO;
            else if (s == "cascade") strategy = RouteStrategy::CASCADE;
            else if (s == "spec_decode") strategy = RouteStrategy::SPEC_DECODE;
            else if (s == "content") strategy = RouteStrategy::CONTENT;
            else if (s == "parallel_moe") strategy = RouteStrategy::PARALLEL_MOE;
            else if (s == "passthrough") strategy = RouteStrategy::PASSTHROUGH;
        } else if (a == "--help" || a == "-h") {
            printf("zaya_server — Pure C++ multi-model, multi-backend inference server\n\n");
            printf("Usage: %s [flags]\n\n", argv[0]);
            printf("Model detection:\n");
            printf("  --model PATH.h1b    Auto-detect architecture from .h1b header\n");
            printf("  --manifest PATH     Load model config from JSON manifest\n");
            printf("  --weights-dir DIR   Directory for weight .bin files\n\n");
            printf("Routing:\n");
            printf("  --strategy auto|cascade|spec_decode|content|parallel_moe|passthrough\n\n");
            printf("Server:\n");
            printf("  --port N            Listen port (default: 8088)\n\n");
            printf("Endpoints:\n");
            printf("  GET  /v1/models                      List loaded models\n");
            printf("  POST /v1/chat/completions            OpenAI-compatible chat\n");
            printf("  GET  /                               Server health\n");
            return 0;
        } else if (a[0] != '-' && model_arg.empty()) port = atoi(argv[i]);
    }

    TokenRouter router;
    router.strategy = strategy;
    if (!router.init()) { fprintf(stderr, "FATAL: TokenRouter init failed\n"); return 1; }

    ModelConfig cfg;
    bool detected = false;
    if (!manifest_arg.empty()) detected = detect_from_manifest(manifest_arg, cfg);
    if (!detected && !model_arg.empty()) {
        detected = detect_from_h1b(model_arg, cfg);
        if (detected && cfg.weights_dir.empty()) {
            auto slash = model_arg.find_last_of('/');
            cfg.weights_dir = (slash != std::string::npos) ? model_arg.substr(0, slash + 1) : "./";
        }
    }
    if (!detected) {
        // No model path was given. The backends will still start (the CPU
        // fallback always "loads"), but with no real weights the server only
        // ever produces empty/garbage output — which is exactly the silent
        // failure mode issue #232 reported. Warn loudly and remember the
        // state so /v1/chat/completions can return an actionable 503 instead
        // of an empty 200, and / health can report model_loaded=false.
        fprintf(stderr,
            "\n  *** No model specified — running WITHOUT weights. ***\n"
            "  The server will start, but /v1/chat/completions will return an\n"
            "  error until you pass a real model, e.g.:\n"
            "      %s --model /path/to/model.h1b\n"
            "      %s --manifest model.json\n\n",
            argv[0], argv[0]);
        cfg.model_name = "Zaya1-8B";
        cfg.weights_dir = weights_dir;
    }
    const bool model_loaded = detected;
    if (cfg.weights_dir.empty()) cfg.weights_dir = weights_dir;
    cfg.lora_path = lora_path;
    fprintf(stderr, "Weights directory: %s\n", cfg.weights_dir.c_str());
    if (!cfg.lora_path.empty()) fprintf(stderr, "LoRA adapter: %s\n", cfg.lora_path.c_str());
    fprintf(stderr, "\n");

    if (model_loaded && !router.load_model(cfg)) { fprintf(stderr, "FATAL: Failed to load model\n"); return 1; }

    httplib::Server svr;

    // Maximum request body size: 1MB (fixes #197: stack buffer + incomplete read + no limit)
    const size_t MAX_BODY_BYTES = 1 * 1024 * 1024;
    svr.set_payload_max_length(MAX_BODY_BYTES);

    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        if (req.method == "OPTIONS") {
            res.status = 200;
            res.set_content("{\"ok\":true}", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    fprintf(stderr, "\nListening on http://127.0.0.1:%d\n", port);
    fprintf(stderr, "   GET  /                  — health\n");
    fprintf(stderr, "   GET  /v1/models          — model list\n");
    fprintf(stderr, "   POST /v1/chat/completions — OpenAI-compatible\n");
    fprintf(stderr, "   Strategy: %s\n\n",
        strategy == RouteStrategy::AUTO ? "auto (fastest available)" :
        strategy == RouteStrategy::CASCADE ? "cascade (per-token fallback)" :
        strategy == RouteStrategy::SPEC_DECODE ? "spec_decode (draft+verify)" :
        strategy == RouteStrategy::CONTENT ? "content (keyword-based)" :
        strategy == RouteStrategy::PARALLEL_MOE ? "parallel_moe (GPU+NPU)" : "passthrough");

    SimpleTokenizer tok;

    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        std::string resp = "{\"status\":\"" + std::string(model_loaded ? "ok" : "no_model") + "\",\"model_loaded\":" + (model_loaded ? "true" : "false") + ",\"model\":\"" + json_escape(cfg.model_name) + "\","
            "\"backend\":\"" + std::string(router.primary ? router.primary->name() : "none") + "\","
            "\"hidden_size\":" + std::to_string(cfg.hidden_size) + ","
            "\"layers\":" + std::to_string(cfg.num_layers) + ","
            "\"vocab\":" + std::to_string(cfg.vocab_size) + ","
            "\"strategy\":\"" +
            (strategy == RouteStrategy::AUTO ? "auto" :
             strategy == RouteStrategy::CASCADE ? "cascade" :
             strategy == RouteStrategy::SPEC_DECODE ? "spec_decode" :
             strategy == RouteStrategy::CONTENT ? "content" :
             strategy == RouteStrategy::PARALLEL_MOE ? "parallel_moe" : "passthrough") +
            "\","
            "\"moe_pipeline\":" + std::string(router.moe_pipeline_.enabled_ ? "true" : "false") + ","
            "\"version\":\"2026.07\"}";
        res.set_content(resp, "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        std::string resp = "{\"object\":\"list\",\"data\":[";
        for (size_t i = 0; i < router.loaded_models.size(); i++) {
            if (i) resp += ",";
            resp += "{\"id\":\"" + json_escape(router.loaded_models[i].model_name) + "\",\"object\":\"model\",\"owned_by\":\"1bit-systems\"}";
        }
        resp += "]}";
        res.set_content(resp, "application/json");
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        const std::string& body = req.body;
        // No real model loaded → return an actionable error instead of an
        // empty 200 (issue #232). The CPU fallback always "loads", so we
        // gate on whether a model path/manifest was actually provided.
        if (!model_loaded) {
            res.status = 503;
            res.set_content(
                "{\"error\":{\"message\":\"No model loaded. Restart with --model "
                "<path.h1b> or --manifest <model.json> (the README quick-start runs "
                "without weights by default).\",\"type\":\"no_model\","
                "\"code\":\"model_not_loaded\"}}", "application/json");
            return;
        }
        int max_tokens = 256;
        try {
            json jbody = json::parse(body);
            max_tokens = jbody.value("max_tokens", 256);
        } catch (...) {}

        RouteStrategy use_strat = strategy;
        if (use_strat == RouteStrategy::CONTENT) {
            std::string user_msg;
            try {
                json jbody = json::parse(body);
                if (jbody.contains("messages") && jbody["messages"].is_array() && !jbody["messages"].empty())
                    user_msg = jbody["messages"][0].value("content", std::string());
                else
                    user_msg = jbody.value("content", std::string());
            } catch (...) {}
            fprintf(stderr, "  [content] routing: %s\n", should_use_large_model(user_msg) ? "large model" : "small model (NPU)");
            use_strat = RouteStrategy::AUTO;
        }

        std::string prompt = build_chatml(body);
        if (prompt.empty()) {
            try {
                json jbody = json::parse(body);
                prompt = jbody.value("prompt", std::string());
            } catch (...) {}
            if (prompt.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"No messages or prompt\"}", "application/json");
                return;
            }
            prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        }

        std::vector<int> tokens = tok.encode(prompt);
        fprintf(stderr, "  → %d prompt tokens, max %d new\n", (int)tokens.size(), max_tokens);

        InferenceResult result = router.infer(tokens, max_tokens, use_strat);
        std::string text = tok.decode(result.tokens);
        std::string finish_reason = "stop";
        if (!result.tokens.empty() && result.tokens.back() != tok.eos_id && (int)result.tokens.size() >= max_tokens)
            finish_reason = "length";

        // Dynamic buffer -- no fixed-size limit (fixes #194: truncation of long output)
        std::string resp_body =
            std::string("{\"id\":\"chatcmpl-") + std::to_string((long long)time(nullptr)) +
            "\",\"object\":\"chat.completion\",\"created\":" + std::to_string((long long)time(nullptr)) +
            ",\"model\":\"" + json_escape(cfg.model_name) +
            "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
            json_escape(text) +
            "\"},\"finish_reason\":\"" + finish_reason +
            "\"}],\"usage\":{\"prompt_tokens\":" + std::to_string((int)tokens.size()) +
            ",\"completion_tokens\":" + std::to_string((int)result.tokens.size()) +
            ",\"total_tokens\":" + std::to_string((int)(tokens.size() + result.tokens.size())) +
            "},\"x-backend\":\"" + (router.primary ? router.primary->name() : "none") +
            "\",\"x-ms\":" + std::to_string((long long)result.gen_ms) +
            ",\"x-tok-s\":" + std::to_string((double)result.tok_s) + "}";
        res.set_content(resp_body, "application/json");
        fprintf(stderr, "  ← %d tokens in %.0fms (%.1f tok/s) [%s]\n",
                (int)result.tokens.size(), result.gen_ms, result.tok_s,
                router.primary ? router.primary->name() : "none");
    });

    svr.Post("/completion", [&](const httplib::Request& req, httplib::Response& res) {
        const std::string& body = req.body;
        std::vector<int> input;
        int np = 16;
        try {
            json jbody = json::parse(body);
            if (jbody.contains("tokens") && jbody["tokens"].is_array()) {
                for (auto& t : jbody["tokens"])
                    input.push_back(t.get<int>());
            }
            if (input.empty() && jbody.contains("prompt") && jbody["prompt"].is_string()) {
                input = tok.encode(jbody["prompt"].get<std::string>());
            }
            np = jbody.value("n_predict", 16);
        } catch (...) {}
        if (input.empty()) {
            std::string prompt;
            try {
                json jbody = json::parse(body);
                prompt = jbody.value("prompt", std::string());
            } catch (...) {}
            if (prompt.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"need prompt or tokens\"}", "application/json");
                return;
            }
            input = tok.encode(prompt);
        }
        InferenceResult result = router.infer(input, np, RouteStrategy::AUTO);
        std::string text = tok.decode(result.tokens);
        std::string rsp = "{\"tokens\":[";
        for (size_t i = 0; i < result.tokens.size(); i++) {
            if (i) rsp += ",";
            rsp += std::to_string(result.tokens[i]);
        }
        rsp += "],\"text\":\"" + json_escape(text) + "\",\"gen_ms\":" +
               std::to_string(result.gen_ms) + ",\"tok_s\":" +
               std::to_string(result.tok_s) + "}";
        res.set_content(rsp, "application/json");
    });

    svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (res.status == 404)
            res.set_content("{\"error\":\"not found\"}", "application/json");
    });

    if (!svr.listen("0.0.0.0", port)) {
        fprintf(stderr, "FATAL: failed to bind/listen on port %d\n", port);
        return 1;
    }
    return 0;
}
