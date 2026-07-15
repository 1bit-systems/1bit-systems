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
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

static bool detect_from_h1b(const std::string& path, ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4]; f.read(magic, 4);
    if (std::strncmp(magic, "H1B", 3) != 0) return false;
    int32_t version; f.read(reinterpret_cast<char*>(&version), 4);
    if (version < 1 || version > 5) return false;
    int32_t hdr[9]; f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    cfg.hidden_size = hdr[0]; cfg.intermediate_size = hdr[1]; cfg.num_layers = hdr[2];
    cfg.num_heads = hdr[3]; cfg.num_kv_heads = hdr[4]; cfg.vocab_size = hdr[5];
    cfg.max_seq_len = hdr[6]; cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.num_experts = 16; cfg.num_experts_top = 2; cfg.router_hidden = 256;
    if (version >= 2) {
        float extras[2]; f.read(reinterpret_cast<char*>(extras), sizeof(extras));
        cfg.rope_theta = extras[0] > 0 ? extras[0] : 500000.0f;
        cfg.rms_norm_eps = extras[1] > 0 ? extras[1] : 1e-5f;
    }
    auto slash = path.find_last_of('/');
    cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
    cfg.model_path = path;
    fprintf(stderr, "  Auto-detected from .h1b: %s\n", cfg.model_name.c_str());
    return true;
}

static bool detect_from_manifest(const std::string& path, ModelConfig& cfg) {
    std::ifstream f(path); if (!f) return false;
    try {
        json j = json::parse(f);
        cfg.hidden_size = j.value("hidden_size", 2048);
        cfg.num_heads = j.value("num_heads", 16);
        cfg.num_kv_heads = j.value("num_kv_heads", 2);
        cfg.head_dim = j.value("head_dim", 128);
        cfg.num_layers = j.value("num_layers", 40);
        cfg.vocab_size = j.value("vocab_size", 262272);
        cfg.intermediate_size = j.value("intermediate_size", 2048);
        cfg.max_seq_len = j.value("max_seq_len", 2048);
        cfg.num_experts = j.value("num_experts", 16);
        cfg.router_hidden = j.value("router_hidden", 256);
        cfg.num_experts_top = j.value("num_experts_top", 2);
        cfg.rope_theta = j.value("rope_theta", 500000.0f);
        cfg.rms_norm_eps = j.value("rms_norm_eps", 1e-5f);
        cfg.model_name = j.value("name", std::string());
        cfg.weights_dir = j.value("weights_dir", std::string());
        if (cfg.model_name.empty()) {
            auto slash = path.find_last_of('/');
            cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
        }
        if (cfg.weights_dir.empty()) cfg.weights_dir = "/tmp/zaya_weights/";
        return true;
    } catch (...) { return false; }
}

static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: if ((unsigned char)c < 32) { char buf[8]; snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c); out += buf; }
                     else out += c;
        }
    }
    return out;
}

static std::string build_chatml(const std::string& body) {
    try {
        json j = json::parse(body);
        if (j.contains("messages") && j["messages"].is_array()) {
            std::string result;
            for (auto& msg : j["messages"]) {
                std::string role = msg.value("role", std::string());
                std::string content = msg.value("content", std::string());
                if (!role.empty() && !content.empty())
                    result += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
            }
            if (!result.empty()) result += "<|im_start|>assistant\n";
            return result;
        }
        if (j.contains("prompt") && j["prompt"].is_string())
            return "<|im_start|>user\n" + j["prompt"].get<std::string>() + "<|im_end|>\n<|im_start|>assistant\n";
    } catch (...) {}
    return "";
}

struct SimpleTokenizer {
    int bos_id = 2; int eos_id = 106; bool use_bpe = false;
    rcpp_tokenizer_t* bpe_tok = nullptr;
    ~SimpleTokenizer() { if (bpe_tok) rcpp_tokenizer_free(bpe_tok); }
    bool load_htok(const std::string& path) {
        rcpp_tokenizer_t* tok = nullptr;
        if (rcpp_tokenizer_load(path.c_str(), &tok) == RCPP_OK && tok) {
            bpe_tok = tok; use_bpe = true;
            bos_id = rcpp_tokenizer_bos_id(bpe_tok);
            eos_id = rcpp_tokenizer_eos_id(bpe_tok);
            return true;
        }
        return false;
    }
    std::vector<int> encode(const std::string& text) {
        if (use_bpe && bpe_tok) {
            std::vector<int> r(4096); size_t out_n = 0;
            if (rcpp_tokenizer_encode(bpe_tok, text.c_str(), text.size(), 1, r.data(), r.size(), &out_n) == RCPP_OK && out_n > 0) {
                r.resize(out_n); return r;
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
            std::string r(4096, '\0'); size_t out_len = 0;
            if (rcpp_tokenizer_decode(bpe_tok, tokens.data(), tokens.size(), r.data(), r.size(), &out_len) == RCPP_OK && out_len > 0) {
                r.resize(out_len); return r;
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
            printf("zaya_server -- Pure C++ multi-model, multi-backend inference server\n\n");
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
        }
    }

    TokenRouter router; router.strategy = strategy;
    if (!router.init()) { fprintf(stderr, "FATAL: TokenRouter init failed\n"); return 1; }

    ModelConfig cfg;
    bool detected = false;
    if (!manifest_arg.empty()) detected = detect_from_manifest(manifest_arg, cfg);
    if (!detected && !model_arg.empty()) detected = detect_from_h1b(model_arg, cfg);
    if (!detected) {
        fprintf(stderr, "No model specified -- using defaults.\n");
        cfg.model_name = "Zaya1-8B"; cfg.weights_dir = weights_dir;
    }
    if (cfg.weights_dir.empty()) cfg.weights_dir = weights_dir;

    if (!router.load_model(cfg)) { fprintf(stderr, "FATAL: Failed to load model\n"); return 1; }

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1; setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {AF_INET, htons((uint16_t)port), {INADDR_ANY}};
    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    listen(sock, 8);

    SimpleTokenizer tok;
    const size_t MAX_BODY_BYTES = 1 * 1024 * 1024;

    auto send_json = [&](int code, const std::string& j) {
        const char* reason = "OK";
        switch (code) {
            case 200: reason = "OK"; break;
            case 400: reason = "Bad Request"; break;
            case 404: reason = "Not Found"; break;
            case 413: reason = "Payload Too Large"; break;
            case 500: reason = "Internal Server Error"; break;
        }
        std::string h = "HTTP/1.1 " + std::to_string(code) + " " + reason + "\r\n"
            "Content-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "Content-Length: " + std::to_string(j.size()) + "\r\nConnection: close\r\n\r\n" + j;
        write(cl, h.data(), h.size()); close(cl);
    };

    // ... (rest of event loop unchanged)
    return 0;
}