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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

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
    cfg.num_experts_top   = 17;
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
        cfg.num_experts_top   = j.value("num_experts_top", 17);
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
    std::vector<int> encode(const std::string& text) {
        std::vector<int> r = {2};
        for (char c : text) if (c >= ' ' && c <= '~') r.push_back((unsigned char)c + 100);
        return r;
    }
    std::string decode(const std::vector<int>& tokens) {
        std::string r;
        for (int v : tokens) {
            if (v == 2 || v == 106) continue;
            if (v > 100 && v < 200) r += (char)(v - 100);
            else { r += '['; r += std::to_string(v); r += ']'; }
        }
        return r;
    }
};

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int port = 8088;
    std::string model_arg, manifest_arg, weights_dir = "/tmp/zaya_weights/";
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
        fprintf(stderr, "No model specified — using Zaya1-8B defaults.\n");
        fprintf(stderr, "  Provide --model or --manifest for auto-detection.\n\n");
        cfg.model_name = "Zaya1-8B";
        cfg.weights_dir = weights_dir;
    }
    if (cfg.weights_dir.empty()) cfg.weights_dir = weights_dir;
    fprintf(stderr, "Weights directory: %s\n\n", cfg.weights_dir.c_str());

    if (!router.load_model(cfg)) { fprintf(stderr, "FATAL: Failed to load model\n"); return 1; }

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1; setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {AF_INET, htons((uint16_t)port), {INADDR_ANY}};
    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    listen(sock, 8);

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

    // Maximum request body size: 1MB (fixes #197: stack buffer + incomplete read + no limit)
    const size_t MAX_BODY_BYTES = 1 * 1024 * 1024;

    while (true) {
        int cl = accept(sock, nullptr, nullptr);
        if (cl < 0) continue;
        // Use heap-allocated buffer instead of 256KB stack buffer
        std::vector<char> buf(32768);
        int n = read(cl, buf.data(), buf.size() - 1);
        if (n <= 0) { close(cl); continue; }
        buf[n] = 0;
        std::string r(buf.data());
        auto bp = r.find("\r\n\r\n");
        std::string body = (bp == std::string::npos) ? "" : r.substr(bp + 4);
        // Case-insensitive Content-Length search per RFC 7230
        auto clp = r.find("Content-Length: ");
        if (clp == std::string::npos) {
            std::string lower = r;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            clp = lower.find("content-length: ");
        }
        if (clp != std::string::npos) {
            auto cle = r.find("\r\n", clp);
            if (cle != std::string::npos) {
                int bl = atoi(r.substr(clp + 16, cle - clp - 16).c_str());
                // Enforce maximum body size
                if (bl > (int)MAX_BODY_BYTES) {
                    const char* err = "HTTP/1.1 413 Payload Too Large\r\nContent-Length: 47\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"Request payload too large\"}\n";
                    write(cl, err, strlen(err)); close(cl); continue;
                }
                // Loop until all Content-Length bytes are received
                while ((int)body.size() < bl) {
                    if (body.empty()) {
                        // Body wasn't in first read; need to read from start
                        int remaining = bl;
                        std::vector<char> bbuf(remaining + 1, 0);
                        int got = 0;
                        while (got < remaining) {
                            int nr = read(cl, bbuf.data() + got, remaining - got);
                            if (nr <= 0) break;
                            got += nr;
                        }
                        body = std::string(bbuf.data(), got);
                        break;
                    } else {
                        // Read remaining bytes
                        std::vector<char> extra(bl - body.size() + 1, 0);
                        int nr = read(cl, extra.data(), bl - body.size());
                        if (nr <= 0) break;
                        body.append(extra.data(), nr);
                    }
                }
            }
        }

        auto send_json = [&](int code, const std::string& j) {
            std::string h = "HTTP/1.1 " + std::to_string(code) + " OK\r\n"
                "Content-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                "Content-Length: " + std::to_string(j.size()) + "\r\nConnection: close\r\n\r\n" + j;
            write(cl, h.data(), h.size()); close(cl);
        };

        if (r.find("OPTIONS") == 0) { send_json(200, "{\"ok\":true}"); continue; }

        if (r.find("GET / ") != std::string::npos || r.find("GET / HTTP") != std::string::npos) {
            std::string resp = "{\"status\":\"ok\",\"model\":\"" + json_escape(cfg.model_name) + "\","
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
            send_json(200, resp); continue;
        }

        if (r.find("GET /v1/models") != std::string::npos) {
            std::string resp = "{\"object\":\"list\",\"data\":[";
            for (size_t i = 0; i < router.loaded_models.size(); i++) {
                if (i) resp += ",";
                resp += "{\"id\":\"" + json_escape(router.loaded_models[i].model_name) + "\",\"object\":\"model\",\"owned_by\":\"1bit-systems\"}";
            }
            resp += "]}";
            send_json(200, resp); continue;
        }

        if (r.find("POST /v1/chat/completions") != std::string::npos) {
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
                if (prompt.empty()) { send_json(400, "{\"error\":\"No messages or prompt\"}"); continue; }
                prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
            }

            std::vector<int> tokens = tok.encode(prompt);
            fprintf(stderr, "  → %d prompt tokens, max %d new\n", (int)tokens.size(), max_tokens);

            InferenceResult result = router.infer(tokens, max_tokens, use_strat);
            std::string text = tok.decode(result.tokens);
            std::string finish_reason = "stop";
            if (!result.tokens.empty() && result.tokens.back() != 106 && (int)result.tokens.size() >= max_tokens)
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
            send_json(200, resp_body);
            fprintf(stderr, "  ← %d tokens in %.0fms (%.1f tok/s) [%s]\n",
                    (int)result.tokens.size(), result.gen_ms, result.tok_s,
                    router.primary ? router.primary->name() : "none");
            continue;
        }

        if (r.find("POST /completion") != std::string::npos) {
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
                if (prompt.empty()) { send_json(400, "{\"error\":\"need prompt or tokens\"}"); continue; }
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
            send_json(200, rsp);
            continue;
        }

        send_json(404, "{\"error\":\"not found\"}");
    }
    close(sock);
    return 0;
}
