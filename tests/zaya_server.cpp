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
#include "a2a_client.h"

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

// ─── A2A (Agent-to-Agent) Protocol v1.0 support ───────────────
// Google's open standard for agent interoperability.
// Agent Card + task-based inference via /a2a/v1/message:send

static std::string a2a_agent_card(const ModelConfig& cfg, int port) {
    json card = {
        {"name", "1bit-systems Inference Agent"},
        {"description", "Multi-backend AI inference server with auto-detection (ROCm HIP > Vulkan > NPU > CPU). Supports text generation, speculative decoding, cascade routing, and MoE parallel pipeline across heterogeneous hardware."},
        {"version", "1.0.0"},
        {"protocolVersion", "1.0"},
        {"documentationUrl", "https://github.com/bong-water-water-bong/1bit-systems"},
        {"provider", {{"organization", "1bit.systems"}, {"url", "https://1bit.systems"}}},
        {"capabilities", {{"streaming", true}, {"pushNotifications", false}}},
        {"securitySchemes", json::object()},
        {"defaultInputModes", json::array({"application/json", "text/plain"})},
        {"defaultOutputModes", json::array({"application/json", "text/plain"})},
        {"supportedInterfaces", json::array({{
            {"url", "http://127.0.0.1:" + std::to_string(port) + "/a2a/v1"},
            {"protocolBinding", "JSONRPC"},
            {"protocolVersion", "1.0"}
        }})},
        {"skills", json::array({
            {{
                {"id", "text-generation"},
                {"name", "Text Generation"},
                {"description", "Generates text given a prompt or chat messages. Supports system prompts, temperature, top-k sampling, and max tokens. Routes to the fastest available backend."},
                {"tags", json::array({"inference", "llm", "text", "generation", "chat"})},
                {"inputModes", json::array({"application/json", "text/plain"})},
                {"outputModes", json::array({"application/json", "text/plain"})},
                {"examples", json::array({
                    "Write a poem about neural networks",
                    "Translate 'hello' to French"
                })},
                {"configuration", {{
                    {"maxTokens", 4096},
                    {"temperature", {{"type", "number"}, {"default", 0.7}, {"description", "Sampling temperature (0.0 = greedy)"}}},
                    {"topK", {{"type", "integer"}, {"default", 40}, {"description", "Top-k sampling"}}},
                    {"strategy", {{"type", "string"}, {"default", "auto"}, {"enum", json::array({"auto", "cascade", "spec_decode", "parallel_moe"})}}}
                }}}
            }},
            {{
                {"id", "model-discovery"},
                {"name", "Model Discovery"},
                {"description", "Lists all loaded models and their configurations (hidden size, layers, heads, backend)."},
                {"tags", json::array({"models", "discovery", "config"})},
                {"inputModes", json::array({"application/json"})},
                {"outputModes", json::array({"application/json"})}
            }}
        })}
    };
    return card.dump(2);
}

// Build A2A task response from inference result
static std::string a2a_task_response(const std::string& task_id, const std::string& context_id,
                                       const std::string& state, const std::string& text,
                                       int prompt_tokens, int completion_tokens) {
    json resp = {
        {"task", {{
            {"id", task_id},
            {"contextId", context_id},
            {"status", {{"state", state}}},
            {"artifacts", json::array({{
                {"artifactId", task_id + "-artifact"},
                {"name", "generation-result"},
                {"parts", json::array({{
                    {"text", text}
                }})},
                {"metadata", {{
                    {"promptTokens", prompt_tokens},
                    {"completionTokens", completion_tokens}
                }}}
            }})}
        }}}
    };
    return resp.dump();
}

static std::string a2a_task_status(const std::string& task_id, const std::string& context_id,
                                    const std::string& state, const std::string& msg) {
    json resp = {
        {"task", {{
            {"id", task_id},
            {"contextId", context_id},
            {"status", {{
                {"state", state},
                {"message", {{"role", "ROLE_AGENT"}, {"parts", json::array({{"text", msg}})}}}
            }}}
        }}}
    };
    return resp.dump();
}

static std::string a2a_handle_message(const std::string& body, const std::string& task_id,
                                        TokenRouter& router, SimpleTokenizer& tok, bool model_loaded) {
    if (!model_loaded) {
        return a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_FAILED",
                               "No model loaded. Restart with --model <path.h1b>");
    }

    try {
        json j = json::parse(body);
        std::string user_text;
        int max_tokens = 256;

        if (j.contains("message") && j["message"].contains("parts") && j["message"]["parts"].is_array()) {
            for (auto& part : j["message"]["parts"]) {
                if (part.contains("text"))
                    user_text += part["text"].get<std::string>();
            }
        }

        if (j.contains("configuration")) {
            auto& config = j["configuration"];
            if (config.contains("maxTokens")) max_tokens = config["maxTokens"].get<int>();
        }

        if (user_text.empty()) {
            return a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_INPUT_REQUIRED",
                                   "Please provide a message with text content.");
        }

        std::string prompt = "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<int> tokens = tok.encode(prompt);
        InferenceResult result = router.infer(tokens, max_tokens, RouteStrategy::AUTO);
        std::string text = tok.decode(result.tokens);

        return a2a_task_response(task_id, "ctx-" + task_id, "TASK_STATE_COMPLETED",
                                  text, (int)tokens.size(), (int)result.tokens.size());
    } catch (const std::exception& e) {
        return a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_FAILED",
                               std::string("Internal error: ") + e.what());
    }
}

static std::string a2a_new_task_id() {
    return "task-" + std::to_string((long long)time(nullptr)) + "-" + std::to_string(rand() % 10000);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int port = 8088;
    std::string model_arg, manifest_arg, weights_dir = "/tmp/zaya_weights/", lora_path;
    RouteStrategy strategy = RouteStrategy::AUTO;
    A2AClient a2a;
    std::vector<std::string> a2a_peers;

    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        if (a == "--port" && i+1 < argc) port = atoi(argv[++i]);
        else if (a == "--model" && i+1 < argc) model_arg = argv[++i];
        else if (a == "--manifest" && i+1 < argc) manifest_arg = argv[++i];
        else if (a == "--weights-dir" && i+1 < argc) weights_dir = argv[++i];
        else if (a == "--a2a-peer" && i+1 < argc) a2a_peers.push_back(argv[++i]);
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
            printf("  --strategy auto|cascade|spec_decode|content|parallel_moe|passthrough\n");
            printf("  --a2a-peer URL       Register remote A2A agent peer (can repeat)\n\n");
            printf("Server:\n");
            printf("  --port N            Listen port (default: 8088)\n\n");
            printf("Endpoints:\n");
            printf("  GET  /v1/models                      List loaded models\n");
            printf("  POST /v1/chat/completions            OpenAI-compatible chat\n");
            printf("  GET  /.well-known/agent-card.json    A2A Agent Card (Google A2A v1.0)\n");
            printf("  POST /a2a/v1/message:send            A2A send message (task-based)\n");
            printf("  POST /a2a/v1/message:sendStream      A2A streaming (SSE)\n");
            printf("  POST /a2a/v1/tasks:route             A2A route to best peer by skill\n");
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
    fprintf(stderr, "   GET  /                      — health\n");
    fprintf(stderr, "   GET  /v1/models              — model list\n");
    fprintf(stderr, "   GET  /.well-known/agent-card — A2A Agent Card (v1.0)\n");
    fprintf(stderr, "   POST /a2a/v1/message:send     — A2A task inference\n");
    fprintf(stderr, "   POST /a2a/v1/tasks:route      — A2A route to peer agent\n");
    fprintf(stderr, "   POST /v1/chat/completions     — OpenAI-compatible\n");
    fprintf(stderr, "   Strategy: %s\n",
        strategy == RouteStrategy::AUTO ? "auto (fastest available)" :
        strategy == RouteStrategy::CASCADE ? "cascade (per-token fallback)" :
        strategy == RouteStrategy::SPEC_DECODE ? "spec_decode (draft+verify)" :
        strategy == RouteStrategy::CONTENT ? "content (keyword-based)" :
        strategy == RouteStrategy::PARALLEL_MOE ? "parallel_moe (GPU+NPU)" : "passthrough");

    // Discover A2A peers
    for (const auto& peer_url : a2a_peers) {
        fprintf(stderr, "  [a2a] discovering peer: %s\n", peer_url.c_str());
        if (!a2a.discover(peer_url)) {
            fprintf(stderr, "  [a2a] WARNING: could not discover %s\n", peer_url.c_str());
        }
    }
    if (!a2a_peers.empty()) {
        fprintf(stderr, "   A2A peers:\n");
        for (auto& p : a2a.peers)
            fprintf(stderr, "     - %s @ %s (%zu skills)\n", p.name.c_str(), p.base_url.c_str(), p.skill_ids.size());
    }
    fprintf(stderr, "\n");

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
            "\"agentCard\":\"/.well-known/agent-card.json\","
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

    svr.Get("/.well-known/agent-card.json", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(a2a_agent_card(cfg, port), "application/json");
    });

    svr.Post("/a2a/v1/message:send", [&](const httplib::Request& req, httplib::Response& res) {
        std::string task_id = a2a_new_task_id();
        res.set_content(a2a_handle_message(req.body, task_id, router, tok, model_loaded), "application/json");
    });

    svr.Post("/a2a/v1/message:sendStream", [&](const httplib::Request& req, httplib::Response& res) {
        std::string task_id = a2a_new_task_id();
        std::string body = req.body;
        res.set_chunked_content_provider("text/event-stream",
            [task_id, body, &router, &tok, model_loaded](size_t, httplib::DataSink& sink) {
                std::string e1 = "event: taskStatus\ndata: " + a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_SUBMITTED", "Task accepted") + "\n\n";
                sink.write(e1.data(), e1.size());

                std::string e2 = "event: taskStatus\ndata: " + a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_WORKING", "Processing inference") + "\n\n";
                sink.write(e2.data(), e2.size());

                std::string result = a2a_handle_message(body, task_id, router, tok, model_loaded);
                std::string e3 = "event: taskArtifact\ndata: " + result + "\n\n";
                sink.write(e3.data(), e3.size());

                std::string e4 = "event: taskStatus\ndata: " + a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_COMPLETED", "Done") + "\n\n";
                sink.write(e4.data(), e4.size());

                sink.done();
                return true;
            });
    });

    svr.Post("/a2a/v1/tasks:route", [&](const httplib::Request& req, httplib::Response& res) {
        if (a2a.peers.empty()) {
            res.status = 503;
            res.set_content("{\"error\":\"No A2A peers. Use --a2a-peer\"}", "application/json");
            return;
        }
        try {
            json jbody = json::parse(req.body);
            std::string skill = jbody.value("skill", "");
            std::string text;
            int mt = jbody.value("maxTokens", 256);
            if (jbody.contains("message") && jbody["message"].contains("parts"))
                for (auto& p : jbody["message"]["parts"])
                    if (p.contains("text")) text += p["text"].get<std::string>();
            if (text.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"empty message\"}", "application/json");
                return;
            }
            auto r2 = a2a.route_by_skill(text, skill, mt);
            if (r2.success) {
                json resp = {{"task", {{"id", r2.task_id}, {"status", {{"state", "TASK_STATE_COMPLETED"}}},
                    {"artifacts", json::array({{{"parts", json::array({{{"text", r2.text}}})},
                        {"metadata", {{"promptTokens", r2.prompt_tokens}, {"completionTokens", r2.completion_tokens}}}
                    }})}
                }}};
                res.set_content(resp.dump(), "application/json");
            } else {
                res.status = 502;
                res.set_content(json({{"error", r2.error}}).dump(), "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", std::string(e.what())}}).dump(), "application/json");
        }
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
