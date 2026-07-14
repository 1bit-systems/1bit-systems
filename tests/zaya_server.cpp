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
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    using namespace json_helpers;
    cfg.hidden_size       = get_int(json, "hidden_size", 2048);
    cfg.num_heads         = get_int(json, "num_heads", 16);
    cfg.num_kv_heads      = get_int(json, "num_kv_heads", 2);
    cfg.head_dim          = get_int(json, "head_dim", 128);
    cfg.num_layers        = get_int(json, "num_layers", 40);
    cfg.vocab_size        = get_int(json, "vocab_size", 262272);
    cfg.intermediate_size = get_int(json, "intermediate_size", 2048);
    cfg.max_seq_len       = get_int(json, "max_seq_len", 2048);
    cfg.num_experts       = get_int(json, "num_experts", 16);
    cfg.router_hidden     = get_int(json, "router_hidden", 256);
    cfg.num_experts_top   = get_int(json, "num_experts_top", 17);
    cfg.rope_theta        = get_float(json, "rope_theta", 500000.0f);
    cfg.rms_norm_eps      = get_float(json, "rms_norm_eps", 1e-5f);
    cfg.model_name        = get_str(json, "name");
    cfg.weights_dir       = get_str(json, "weights_dir");
    if (cfg.model_name.empty()) {
        auto slash = path.find_last_of('/');
        cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
    }
    if (cfg.weights_dir.empty()) cfg.weights_dir = "/tmp/zaya_weights/";
    fprintf(stderr, "  Loaded manifest: %s\n", cfg.model_name.c_str());
    fprintf(stderr, "    hidden=%d layers=%d heads=%d vocab=%d\n",
            cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.vocab_size);
    return true;
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
    size_t msgs_start = body.find("\"messages\"");
    if (msgs_start == std::string::npos) {
        std::string prompt = json_helpers::get_str(body, "prompt");
        if (!prompt.empty()) return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        return "";
    }
    msgs_start = body.find('[', msgs_start);
    if (msgs_start == std::string::npos) return "";
    std::string result;
    size_t pos = msgs_start;
    while (pos < body.size()) {
        size_t obj_start = body.find('{', pos);
        if (obj_start == std::string::npos) break;
        int depth = 1;
        size_t obj_end = obj_start + 1;
        while (obj_end < body.size() && depth > 0) {
            if (body[obj_end] == '{') depth++;
            else if (body[obj_end] == '}') depth--;
            obj_end++;
        }
        if (depth != 0) break;
        obj_end--;
        std::string msg = body.substr(obj_start, obj_end - obj_start + 1);
        std::string role = json_helpers::get_str(msg, "role");
        std::string content = json_helpers::get_str(msg, "content");
        if (!role.empty() && !content.empty())
            result += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        pos = obj_end + 1;
        if (pos >= body.size()) break;
        if (body.find(']', pos) < body.find('{', pos)) break;
    }
    if (result.empty()) return result;
    result += "<|im_start|>assistant\n";
    return result;
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

    while (true) {
        int cl = accept(sock, nullptr, nullptr);
        if (cl < 0) continue;
        char buf[262144];
        int n = read(cl, buf, sizeof(buf) - 1);
        if (n <= 0) { close(cl); continue; }
        buf[n] = 0;
        std::string r(buf);
        auto bp = r.find("\r\n\r\n");
        std::string body = (bp == std::string::npos) ? "" : r.substr(bp + 4);
        auto clp = r.find("Content-Length: ");
        if (clp != std::string::npos) {
            auto cle = r.find("\r\n", clp);
            if (cle != std::string::npos) {
                int bl = atoi(r.substr(clp + 16, cle - clp - 16).c_str());
                if ((int)body.size() < bl) {
                    int more = read(cl, buf + bp + 4 + body.size(), bl - (int)body.size());
                    if (more > 0) body.append(buf + bp + 4 + body.size(), more);
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
            int max_tokens = json_helpers::get_int(body, "max_tokens", 256);
            RouteStrategy use_strat = strategy;
            if (use_strat == RouteStrategy::CONTENT) {
                std::string user_msg = json_helpers::get_str(body, "content");
                if (user_msg.empty()) {
                    auto ms = body.find("\"messages\"");
                    if (ms != std::string::npos) {
                        auto cs = body.find("\"content\"", ms);
                        if (cs != std::string::npos) user_msg = json_helpers::get_str(body.substr(cs), "content");
                    }
                }
                fprintf(stderr, "  [content] routing: %s\n", should_use_large_model(user_msg) ? "large model" : "small model (NPU)");
                use_strat = RouteStrategy::AUTO;
            }

            std::string prompt = build_chatml(body);
            if (prompt.empty()) {
                prompt = json_helpers::get_str(body, "prompt");
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

            char resp_buf[65536];
            snprintf(resp_buf, sizeof(resp_buf),
                "{\"id\":\"chatcmpl-%lld\",\"object\":\"chat.completion\",\"created\":%lld,"
                "\"model\":\"%s\","
                "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                "\"finish_reason\":\"%s\"}],"
                "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},"
                "\"x-backend\":\"%s\",\"x-ms\":%.0f,\"x-tok-s\":%.1f}",
                (long long)time(nullptr), (long long)time(nullptr),
                json_escape(cfg.model_name).c_str(),
                json_escape(text).c_str(), finish_reason.c_str(),
                (int)tokens.size(), (int)result.tokens.size(), (int)(tokens.size() + result.tokens.size()),
                router.primary ? router.primary->name() : "none",
                result.gen_ms, result.tok_s);
            send_json(200, resp_buf);
            fprintf(stderr, "  ← %d tokens in %.0fms (%.1f tok/s) [%s]\n",
                    (int)result.tokens.size(), result.gen_ms, result.tok_s,
                    router.primary ? router.primary->name() : "none");
            continue;
        }

        if (r.find("POST /completion") != std::string::npos) {
            auto jtokens = [](const std::string& b) -> std::vector<int> {
                std::vector<int> r;
                auto p = b.find("\"tokens\"");
                if (p == std::string::npos) return r;
                p = b.find('[', p); if (p == std::string::npos) return r;
                p++;
                while (p < b.size() && b[p] != ']') {
                    while (p < b.size() && (b[p]==' '||b[p]==','||b[p]=='\"')) p++;
                    if (p >= b.size() || b[p]==']') break;
                    char* e; r.push_back((int)strtol(&b[p],&e,10)); p=e-b.data();
                }
                return r;
            };
            std::vector<int> input = jtokens(body);
            if (input.empty()) {
                std::string prompt = json_helpers::get_str(body, "prompt");
                if (prompt.empty()) { send_json(400, "{\"error\":\"need prompt or tokens\"}"); continue; }
                input = tok.encode(prompt);
            }
            int np = json_helpers::get_int(body, "n_predict", 16);
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
