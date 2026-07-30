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
#include "gguf_reader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <mutex>

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
    if (!f.good() || version < 1 || version > 5) return false;
    int32_t hdr[9];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!f.good()) return false;
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
        if (f.good()) {
            cfg.rope_theta   = extras[0] > 0 ? extras[0] : 500000.0f;
            cfg.rms_norm_eps = extras[1] > 0 ? extras[1] : 1e-5f;
        }
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

// ─── .1bp header auto-detection ─────────────────────────────────
// 1BP format: 256-byte OnebpHeader. Magic = 0x00504231 = "1BP\0".
// Header layout (all uint32/int32):
//   [0]:  magic  [1]: version  [2]: arch  [3]: quant
//   [4]:  scale_type  [5..17]: hidden_size..max_seq_len (13 int32)
//   [18]: tile_rows  [19]: tile_cols  [20]: group_size
//   [21]: has_q_norm  [22]: has_k_norm  [23]: has_bias
//   [19]: rope_theta_f  [20]: bos_token_id  [21]: eos_token_id
//   [22]: tensor_count
//   [28]: num_experts  [29]: n_expert_used  [30..35]: expert config
//   [36]: rope_freq_base_swa_f  [37]: n_rot_swa  [38]: n_rot_full
//   [39..50]: reserved[12]  [51..63]: reserved[13]
//   [64..79]: model_tag[64] as chars (offset 192)
static bool detect_from_1bp(const std::string& path, ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t header[64]; // 256 bytes / 4
    f.read(reinterpret_cast<char*>(header), 256);
    if (!f.good()) return false;
    if (header[0] != 0x00504231) return false; // magic "1BP\0"
    // Read dimensions from header[5..17]
    cfg.hidden_size       = (int32_t)header[5];
    cfg.num_layers        = (int32_t)header[6];
    cfg.num_heads         = (int32_t)header[7];
    cfg.num_kv_heads      = (int32_t)header[8];
    cfg.head_dim          = (int32_t)header[9];
    cfg.intermediate_size = (int32_t)header[10];
    cfg.vocab_size        = (int32_t)header[11];
    cfg.max_seq_len       = (int32_t)header[12];
    cfg.num_experts       = (int32_t)header[28];
    cfg.num_experts_top   = (int32_t)header[29];
    uint32_t eos_u        = header[21];  // eos_token_id at index 21 per OnebpHeader
    cfg.eos_token_id      = (int)eos_u;
    // router_hidden default (not in 1BP header for older models)
    cfg.router_hidden     = 256;
    f.close();
    auto slash = path.find_last_of('/');
    cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
    cfg.model_path = path;
    cfg.weights_dir = (slash != std::string::npos) ? path.substr(0, slash + 1) : "./";
    fprintf(stderr, "  Auto-detected from .1bp: %s\n", cfg.model_name.c_str());
    fprintf(stderr, "    hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d eos=%d\n",
            cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads,
            cfg.head_dim, cfg.vocab_size, cfg.eos_token_id);
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
        cfg.model_path        = j.value("model_path", std::string());
        cfg.weights_dir       = j.value("weights_dir", std::string());
        if (cfg.model_name.empty()) {
            auto slash = path.find_last_of('/');
            cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
        }
        if (cfg.weights_dir.empty()) {
            const char* home = getenv("HOME");
            cfg.weights_dir = (home && home[0]) ? std::string(home) + "/.local/share/1bit-systems/weights/" : "/tmp/zaya_weights/";
        }
        fprintf(stderr, "  Loaded manifest: %s\n", cfg.model_name.c_str());
        fprintf(stderr, "    hidden=%d layers=%d heads=%d vocab=%d\n",
                cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.vocab_size);
        return true;
    } catch (const json::exception& e) {
        fprintf(stderr, "  Manifest parse error: %s\n", e.what());
        return false;
    }
}

static bool detect_from_gguf(const std::string& path, ModelConfig& cfg) {
    // Read model dimensions from GGUF KV metadata so the backends get
    // the correct H, L, NH, NKV, V upfront (fixes models whose
    // dimensions differ from the default 2048/40/8/2/262272).
    cfg.model_path = path;
    cfg.weights_dir = path.substr(0, path.find_last_of('/') + 1);
    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1, dot - slash - 1) : "gguf-model";
    cfg.hidden_size = 0;
    // Read dimensions from GGUF metadata using the shared reader.
    // Keys are architecture-prefixed (e.g. llama.embedding_length,
    // qwen2.attention.head_count, zr1.block_count).
    GgufReader reader;
    if (reader.open(path)) {
        std::string arch = reader.architecture();
        if (arch.empty()) arch = "llm";
        auto gu = [&](const std::string& k, int def) -> int {
            // Try with architecture prefix first, then bare key
            uint32_t v;
            if (reader.get_u32(arch + "." + k, v)) return (int)v;
            if (reader.get_u32(k, v)) return (int)v;
            return def;
        };
        cfg.hidden_size       = gu("embedding_length", 0);
        cfg.num_layers        = gu("block_count", 40);
        cfg.num_heads         = gu("attention.head_count", 0);
        cfg.num_kv_heads      = gu("attention.head_count_kv", 0);
        cfg.intermediate_size = gu("feed_forward_length", 0);
        cfg.vocab_size        = gu("vocab_size", 0);
        // If vocab_size wasn't in KV metadata, derive it from token_embd.weight
        // If vocab_size wasn't in KV metadata, derive it from output.weight
        // or token_embd.weight shape: numel = V * H, so V = numel / H.
        if (cfg.vocab_size == 0 && cfg.hidden_size > 0) {
            int H = cfg.hidden_size;
            // output.weight has shape [H, V] in GGUF (fastest-first), numel = V*H
            const GgufTensorInfo* out = reader.tensor_info("output.weight");
            if (out && out->numel > 0)
                cfg.vocab_size = (int)(out->numel / H);
            if (cfg.vocab_size == 0) {
                const GgufTensorInfo* emb = reader.tensor_info("token_embd.weight");
                if (!emb) emb = reader.tensor_info("model.embed_tokens.weight");
                if (emb && emb->numel > 0)
                    cfg.vocab_size = (int)(emb->numel / H);
            }
        }
        uint32_t max_seq = 0;
        if (reader.get_u32(arch + ".context_length", max_seq) ||
            reader.get_u32("context_length", max_seq))
            cfg.max_seq_len = (int)max_seq;
        // Cap max_seq_len to prevent excessive KV cache allocation.
        // The backend can dynamically allocate more if needed.
        if (cfg.max_seq_len > 32768) cfg.max_seq_len = 32768;
        if (cfg.hidden_size > 0) {
            // Set numeric architecture enum for backend routing
    // Map GGUF arch string to rcpp_arch_t
    if (arch == "zamba2")      cfg.arch = RCPP_ARCH_ZAMBA2;
    else if (arch == "zamba")  cfg.arch = RCPP_ARCH_ZAMBA;
    else if (arch == "mamba")  cfg.arch = RCPP_ARCH_MAMBA;
    else if (arch == "llama")  cfg.arch = RCPP_ARCH_LLAMA;
    else if (arch == "qwen3") cfg.arch = RCPP_ARCH_QWEN3;
    else if (arch == "qwen2") cfg.arch = RCPP_ARCH_QWEN2;
    else if (arch == "mistral") cfg.arch = RCPP_ARCH_MISTRAL;
    else if (arch == "gemma" || arch == "gemma2" || arch == "gemma3") cfg.arch = RCPP_ARCH_GEMMA;
    else if (arch == "zr1")    cfg.arch = RCPP_ARCH_ZAMBA2;
    else cfg.arch = RCPP_ARCH_BITNET;  // default / unknown

    fprintf(stderr, "  GGUF dims: H=%d L=%d NH=%d NKV=%d FF=%d V=%d CTX=%d (arch=%s)\n",
                    cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads,
                    cfg.intermediate_size, cfg.vocab_size, cfg.max_seq_len, arch.c_str());
        }
    } else {
        fprintf(stderr, "  GGUF: could not open for dimension detection\n");
    }
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
    // Vocab lookup: maps token_id -> token string (loaded from GGUF)
    std::vector<std::string> id_to_token;

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

    /// Load BOS/EOS from GGUF metadata.
    bool load_from_gguf(GgufReader& reader) {
        uint32_t bos = 2, eos = 106;
        if (reader.get_u32("tokenizer.ggml.bos_token_id", bos)) bos_id = (int)bos;
        else { uint32_t alt=0; if(reader.get_u32("tokenizer.ggml.bos_id", alt)) bos_id=(int)alt; }
        if (reader.get_u32("tokenizer.ggml.eos_token_id", eos)) eos_id = (int)eos;
        else { uint32_t alt=0; if(reader.get_u32("tokenizer.ggml.eos_id", alt)) eos_id=(int)alt; }
        fprintf(stderr, "  GGUF tokenizer metadata: BOS=%d EOS=%d\n", bos_id, eos_id);
        if (use_bpe) return true;
        return false;
    }

    /// Load vocab table from GGUF's tokenizer.ggml.tokens array.
    /// This enables readable decode output without a .htok BPE file.
    bool load_vocab_from_gguf(GgufReader& reader) {
        std::vector<std::string> tokens;
        if (!reader.get_string_array("tokenizer.ggml.tokens", tokens)) {
            fprintf(stderr, "  Vocab: tokenizer.ggml.tokens not found\n");
            return false;
        }
        id_to_token = std::move(tokens);
        fprintf(stderr, "  Vocab loaded: %zu tokens from GGUF\n", id_to_token.size());
        return true;
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
        // Character-level fallback with correct BOS/EOS from GGUF metadata
        std::vector<int> r = {bos_id};
        for (unsigned char c : text) {
            if (c >= 32 && c <= 126) r.push_back((int)c + 100);
            else if (c != 0) r.push_back((int)c + 200);
        }
        return r;
    }

    // GPT-2 byte decoding: replaces byte-encoded Unicode chars (U+0100-U+017F)
    // which appear as UTF-8 sequences 0xC4 0x80..0xC5 0xBF back to raw bytes.
    // This converts "Ġ" (U+0120 = space) back to ' ' and "Ċ" (U+010A = \n) back.
    static std::string gpt2_byte_decode(const std::string& s) {
        std::string r;
        r.reserve(s.size());
        for (size_t i = 0; i < s.size();) {
            unsigned char c = (unsigned char)s[i];
            if ((c == 0xC4 || c == 0xC5) && i + 1 < s.size()) {
                unsigned char lo = (unsigned char)s[i+1];
                int cp = ((int)(c & 0x1F) << 6) | (int)(lo & 0x3F);
                r += (char)(cp - 256);
                i += 2;
            } else if (c < 128) {
                r += (char)c;
                i += 1;
            } else {
                int n = 1;
                if ((c & 0xE0) == 0xC0) n = 2;
                else if ((c & 0xF0) == 0xE0) n = 3;
                else if ((c & 0xF8) == 0xF0) n = 4;
                r.append(s.c_str() + i, n);
                i += n;
            }
        }
        return r;
    }

    std::string decode(const std::vector<int>& tokens) {
        if (use_bpe && bpe_tok) {
            std::string r(4096, '\0');
            size_t out_len = 0;
            rcpp_status_t st = rcpp_tokenizer_decode(bpe_tok, tokens.data(), tokens.size(),
                                                      r.data(), r.size(), &out_len);
            if (st == RCPP_OK && out_len > 0) { r.resize(out_len); return gpt2_byte_decode(r); }
            return "";
        }
        // Vocab-based decode (from GGUF tokenizer.ggml.tokens)
        if (!id_to_token.empty()) {
            std::string r;
            for (int v : tokens) {
                if (v == bos_id || v == eos_id) continue;
                if (v >= 0 && v < (int)id_to_token.size()) {
                    r += id_to_token[v];
                } else {
                    r += '<'; r += std::to_string(v); r += '>'; }
            }
            return gpt2_byte_decode(r);
        }
        // Character-level fallback
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
    static std::atomic<uint64_t> counter{0};
    return "task-" + std::to_string((long long)time(nullptr)) + "-" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int port = 8088;
    const char* home_default = getenv("HOME");
    std::string default_weights = (home_default && home_default[0]) ? std::string(home_default) + "/.local/share/1bit-systems/weights/" : "/tmp/zaya_weights/";
    std::string model_arg, manifest_arg, draft_model_arg, weights_dir = default_weights, lora_path;
    RouteStrategy strategy = RouteStrategy::AUTO;
    A2AClient a2a;
    std::vector<std::string> a2a_peers;

    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        if (a == "--port" && i+1 < argc) port = atoi(argv[++i]);
        else if (a == "--model" && i+1 < argc) model_arg = argv[++i];
        else if (a == "--draft-model" && i+1 < argc) draft_model_arg = argv[++i];
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
            printf("  --draft-model PATH  Load draft model for speculative decoding\n");
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
            printf("  POST /v1/batch/completions           Batch inference (multi-prompt)\n");
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

    // httplib serves each connection from a thread pool — every access to
    // `router` (a single stack object captured by reference in every
    // handler below) must be serialized, or concurrent requests race on its
    // KV cache position, active backend pointer, and loaded_models list
    // (AUDIT_ISSUES.md #2). router.infer() is the real hot path but the
    // metadata reads in "/" and "/v1/models" get the same guard for
    // correctness — they're cheap, and a torn/half-updated read of
    // router.primary while another thread is mid-infer() is still UB.
    std::mutex g_router_mutex;

    ModelConfig cfg;
    // Model detection
    bool detected = false;
    if (!manifest_arg.empty()) detected = detect_from_manifest(manifest_arg, cfg);
    if (!detected && !model_arg.empty()) {
        // Try GGUF detection first (most universal path)
        std::string ext = model_arg.size() > 5 ? model_arg.substr(model_arg.size() - 5) : "";
        if (ext == ".gguf") {
            detected = detect_from_gguf(model_arg, cfg);
            fprintf(stderr, "  GGUF detection: %s\n", detected ? "ok" : "failed");
        }
    }
    if (!detected && !model_arg.empty()) {
        std::string ext = model_arg.size() > 4 ? model_arg.substr(model_arg.size() - 4) : "";
        if (ext == ".1bp") {
            detected = detect_from_1bp(model_arg, cfg);
        }
    }
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

    // ── Load draft model for speculative decoding ────────────────
    bool draft_loaded = false;
    if (!draft_model_arg.empty()) {
        ModelConfig draft_cfg;
        bool draft_detected = false;
        std::string ext = draft_model_arg.size() > 5 ? draft_model_arg.substr(draft_model_arg.size() - 5) : "";
        if (ext == ".gguf") {
            draft_detected = detect_from_gguf(draft_model_arg, draft_cfg);
        }
        if (!draft_detected) {
            draft_detected = detect_from_h1b(draft_model_arg, draft_cfg);
        }
        if (draft_detected) {
            draft_cfg.weights_dir = cfg.weights_dir;
            draft_loaded = router.load_draft_model(draft_cfg);
        }
        if (!draft_loaded) {
            fprintf(stderr, "WARNING: Failed to load draft model — running without spec decode\n");
        } else {
            // Auto-enable spec_decode strategy when draft model is loaded
            if (strategy == RouteStrategy::AUTO) {
                router.strategy = RouteStrategy::SPEC_DECODE;
                strategy = RouteStrategy::SPEC_DECODE;
            }
        }
    }

    SimpleTokenizer tok;

    // Try to load tokenizer from GGUF metadata if available
    {
        std::string gguf_path;
        if (!model_arg.empty()) {
            std::string ext = model_arg.size() > 5 ? model_arg.substr(model_arg.size() - 5) : "";
            if (ext == ".gguf") gguf_path = model_arg;
        }
        if (gguf_path.empty() && cfg.model_path.size() > 5) {
            std::string ext = cfg.model_path.substr(cfg.model_path.size() - 5);
            if (ext == ".gguf") gguf_path = cfg.model_path;
        }
        if (!gguf_path.empty()) {
            GgufReader reader;
            if (reader.open(gguf_path)) {
                // Try .htok file alongside the GGUF for full BPE tokenizer
                std::string htok_path = gguf_path.substr(0, gguf_path.size() - 5) + ".htok";
                FILE* htok_test = fopen(htok_path.c_str(), "rb");
                if (htok_test) {
                    fclose(htok_test);
                    tok.load_htok(htok_path);
                }
                tok.load_from_gguf(reader);
                tok.load_vocab_from_gguf(reader);
                fprintf(stderr, "  Tokenizer: BOS=%d EOS=%d %s(vocab=%zu)\n",
                        tok.bos_id, tok.eos_id,
                        tok.use_bpe ? "+ BPE " : "",
                        tok.id_to_token.size());
            }
        } else {
            // Non-GGUF model: search for .htok tokenizer alongside model or in weights_dir
            std::vector<std::string> htok_candidates;
            // Priority 1: <model_path>.htok (same basename, .htok extension)
            std::string model_base = cfg.model_path;
            auto dot = model_base.find_last_of('.');
            if (dot != std::string::npos) {
                htok_candidates.push_back(model_base.substr(0, dot) + ".htok");
            }
            // Priority 2: <model_dir>/tokenizer.htok
            auto slash = cfg.model_path.find_last_of('/');
            std::string model_dir = (slash != std::string::npos) ? cfg.model_path.substr(0, slash) : ".";
            htok_candidates.push_back(model_dir + "/tokenizer.htok");
            // Priority 3: weights_dir/tokenizer.htok
            if (!cfg.weights_dir.empty()) {
                htok_candidates.push_back(cfg.weights_dir + "tokenizer.htok");
            }
            // Priority 4: XDG/HOME fallback
            const char* xdg = getenv("XDG_DATA_HOME");
            if (xdg && xdg[0]) htok_candidates.push_back(std::string(xdg) + "/1bit-systems/weights/tokenizer.htok");
            const char* home = getenv("HOME");
            if (home && home[0]) htok_candidates.push_back(std::string(home) + "/.local/share/1bit-systems/weights/tokenizer.htok");

            bool found = false;
            for (const auto& htok_path : htok_candidates) {
                FILE* htok_test = fopen(htok_path.c_str(), "rb");
                if (htok_test) {
                    fclose(htok_test);
                    fprintf(stderr, "  Tokenizer: found .htok at %s\n", htok_path.c_str());
                    if (tok.load_htok(htok_path)) {
                        found = true;
                        break;
                    }
                }
            }
            if (found) {
                fprintf(stderr, "  Tokenizer: BOS=%d EOS=%d + BPE(vocab loaded from .htok)\n",
                        tok.bos_id, tok.eos_id);
            } else {
                fprintf(stderr, "  Tokenizer: using default BOS=%d EOS=%d (no .htok/GGUF found)\n",
                        tok.bos_id, tok.eos_id);
            }
        }
    }

    httplib::Server svr;

    // Maximum request body size: 1MB (fixes #197: stack buffer + incomplete read + no limit)
    const size_t MAX_BODY_BYTES = 1 * 1024 * 1024;
    svr.set_payload_max_length(MAX_BODY_BYTES);

    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        // Restrictive CORS by default — only allow same-origin (localhost) access.
        // Set ZAYA_CORS_ORIGIN env var to "*" or a specific origin if needed.
        const char* cors_origin = getenv("ZAYA_CORS_ORIGIN");
        res.set_header("Access-Control-Allow-Origin", cors_origin ? cors_origin : "http://127.0.0.1");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        if (req.method == "OPTIONS") {
            res.status = 200;
            res.set_content("{\"ok\":true}", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    fprintf(stderr, "   Strategy: %s\n",
        strategy == RouteStrategy::AUTO ? "auto (fastest available)" :
        strategy == RouteStrategy::CASCADE ? "cascade (per-token fallback)" :
        strategy == RouteStrategy::SPEC_DECODE ?
            (draft_loaded ? "spec_decode (ZR1 draft + Zaya verify)" : "spec_decode (draft+verify)") :
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

    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(g_router_mutex);
        std::string resp = "{\"status\":\"" + std::string(model_loaded ? "ok" : "no_model") + "\",\"model_loaded\":" + (model_loaded ? "true" : "false") + ",\"model\":\"" + json_escape(cfg.model_name) + "\","
            "\"draft_model_loaded\":" + (draft_loaded ? "true" : "false") + ","
            "\"draft_model\":\"" + (draft_loaded && !router.draft_loaded_models.empty() ? json_escape(router.draft_loaded_models[0].model_name) : "") + "\","
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
        std::lock_guard<std::mutex> lock(g_router_mutex);
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
        std::string result;
        {
            std::lock_guard<std::mutex> lock(g_router_mutex);
            result = a2a_handle_message(req.body, task_id, router, tok, model_loaded);
        }
        res.set_content(result, "application/json");
    });

    svr.Post("/a2a/v1/message:sendStream", [&](const httplib::Request& req, httplib::Response& res) {
        std::string task_id = a2a_new_task_id();
        std::string body = req.body;
        res.set_chunked_content_provider("text/event-stream",
            [task_id, body, &router, &tok, model_loaded, &g_router_mutex](size_t, httplib::DataSink& sink) {
                std::string e1 = "event: taskStatus\ndata: " + a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_SUBMITTED", "Task accepted") + "\n\n";
                sink.write(e1.data(), e1.size());

                std::string e2 = "event: taskStatus\ndata: " + a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_WORKING", "Processing inference") + "\n\n";
                sink.write(e2.data(), e2.size());

                std::string result;
                {
                    std::lock_guard<std::mutex> lock(g_router_mutex);
                    result = a2a_handle_message(body, task_id, router, tok, model_loaded);
                }
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
            if (max_tokens < 1) max_tokens = 1;
            if (max_tokens > 32768) max_tokens = 32768;
        } catch (...) { fprintf(stderr, "[zaya] JSON parse error in max_tokens\n"); }

        RouteStrategy use_strat = strategy;
        if (use_strat == RouteStrategy::CONTENT) {
            std::string user_msg;
            try {
                json jbody = json::parse(body);
                if (jbody.contains("messages") && jbody["messages"].is_array() && !jbody["messages"].empty())
                    user_msg = jbody["messages"][0].value("content", std::string());
                else
                    user_msg = jbody.value("content", std::string());
            } catch (...) { fprintf(stderr, "[zaya] JSON parse error in content routing\n"); }
            fprintf(stderr, "  [content] routing: %s\n", should_use_large_model(user_msg) ? "large model" : "small model (NPU)");
            use_strat = RouteStrategy::AUTO;
        }

        std::string prompt = build_chatml(body);
        if (prompt.empty()) {
            try {
                json jbody = json::parse(body);
                prompt = jbody.value("prompt", std::string());
            } catch (...) { fprintf(stderr, "[zaya] JSON parse error in prompt fallback\n"); }
            if (prompt.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"No messages or prompt\"}", "application/json");
                return;
            }
            prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        }

        std::vector<int> tokens = tok.encode(prompt);
        fprintf(stderr, "  → %d prompt tokens, max %d new\n", (int)tokens.size(), max_tokens);

        std::string resp_body;
        {
            std::lock_guard<std::mutex> lock(g_router_mutex);
            InferenceResult result = router.infer(tokens, max_tokens, use_strat);
            std::string text = tok.decode(result.tokens);
            std::string finish_reason = "stop";
            if (!result.tokens.empty() && result.tokens.back() != tok.eos_id && (int)result.tokens.size() >= max_tokens)
                finish_reason = "length";

            // Dynamic buffer -- no fixed-size limit (fixes #194: truncation of long output)
            resp_body =
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
            fprintf(stderr, "  ← %d tokens in %.0fms (%.1f tok/s) [%s]\n",
                    (int)result.tokens.size(), result.gen_ms, result.tok_s,
                    router.primary ? router.primary->name() : "none");
        }
        res.set_content(resp_body, "application/json");
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
        } catch (...) { fprintf(stderr, "[zaya] JSON parse error in /v1/completions\n"); }
        if (input.empty()) {
            std::string prompt;
            try {
                json jbody = json::parse(body);
                prompt = jbody.value("prompt", std::string());
            } catch (...) { fprintf(stderr, "[zaya] JSON parse error in /v1/completions prompt fallback\n"); }
            if (prompt.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"need prompt or tokens\"}", "application/json");
                return;
            }
            input = tok.encode(prompt);
        }
        std::string rsp;
        {
            std::lock_guard<std::mutex> lock(g_router_mutex);
            InferenceResult result = router.infer(input, np, RouteStrategy::AUTO);
            std::string text = tok.decode(result.tokens);
            rsp = "{\"tokens\":[";
            for (size_t i = 0; i < result.tokens.size(); i++) {
                if (i) rsp += ",";
                rsp += std::to_string(result.tokens[i]);
            }
            rsp += "],\"text\":\"" + json_escape(text) + "\",\"gen_ms\":" +
                   std::to_string(result.gen_ms) + ",\"tok_s\":" +
                   std::to_string(result.tok_s) + "}";
        }
        res.set_content(rsp, "application/json");
    });

    // ── Batch endpoint: POST /v1/batch/completions ───────────────
    // Accepts an array of prompts, returns an array of completions.
    // Uses TokenRouter::infer_batch() for efficient multi-prompt processing.
    svr.Post("/v1/batch/completions", [&](const httplib::Request& req, httplib::Response& res) {
        if (!model_loaded) {
            res.status = 503;
            res.set_content("{\"error\":\"no model loaded\"}", "application/json");
            return;
        }
        try {
            json jbody = json::parse(req.body);
            int max_tokens = jbody.value("max_tokens", 256);
            if (max_tokens < 1) max_tokens = 1;
            if (max_tokens > 32768) max_tokens = 32768;

            std::vector<std::vector<int>> prompts;
            if (jbody.contains("prompts") && jbody["prompts"].is_array()) {
                for (auto& p : jbody["prompts"]) {
                    std::string text = p.is_string() ? p.get<std::string>() : p.dump();
                    prompts.push_back(tok.encode(text));
                }
            }

            if (prompts.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"need 'prompts' array\"}", "application/json");
                return;
            }

            std::string resp_body;
            {
                std::lock_guard<std::mutex> lock(g_router_mutex);
                auto results = router.infer_batch(prompts, max_tokens);
                json arr = json::array();
                for (size_t i = 0; i < results.size(); i++) {
                    std::string text = tok.decode(results[i].tokens);
                    arr.push_back({
                        {"index", i},
                        {"text", text},
                        {"tokens", results[i].tokens},
                        {"prompt_tokens", results[i].prompt_tokens},
                        {"completion_tokens", (int)results[i].tokens.size()},
                        {"gen_ms", results[i].gen_ms},
                        {"tok_s", results[i].tok_s}
                    });
                }
                json r = {{"object", "list"}, {"data", arr}};
                resp_body = r.dump();
                fprintf(stderr, "  [batch] %zu prompts, %d max_tokens\n", prompts.size(), max_tokens);
            }
            res.set_content(resp_body, "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (res.status == 404)
            res.set_content("{\"error\":\"not found\"}", "application/json");
    });

    // Bind to localhost by default — use --bind 0.0.0.0 to expose publicly.
    // Binding to all interfaces without auth or TLS is a security risk (AUDIT #7).
    const char* bind_addr = getenv("ZAYA_BIND_ADDR");
    if (!bind_addr || !bind_addr[0]) bind_addr = "127.0.0.1";
    fprintf(stderr, "\nListening on http://%s:%d\n", bind_addr, port);
    if (draft_loaded) {
        fprintf(stderr, "   Speculative decode ACTIVE\n");
        fprintf(stderr, "     Draft: %s\n", router.draft_loaded_models[0].model_name.c_str());
        fprintf(stderr, "     Target: %s\n", cfg.model_name.c_str());
    }
    fprintf(stderr, "   GET  /                      — health\n");
    fprintf(stderr, "   GET  /v1/models              — model list\n");
    fprintf(stderr, "   GET  /.well-known/agent-card — A2A Agent Card (v1.0)\n");
    fprintf(stderr, "   POST /a2a/v1/message:send     — A2A task inference\n");
    fprintf(stderr, "   POST /a2a/v1/tasks:route      — A2A route to peer agent\n");
    fprintf(stderr, "   POST /v1/chat/completions     — OpenAI-compatible\n");
    fprintf(stderr, "   POST /v1/batch/completions    — Batch inference (multi-prompt)\n");
    if (strcmp(bind_addr, "0.0.0.0") == 0) {
        fprintf(stderr,
            "\n  *** WARNING: binding to 0.0.0.0 — server is publicly reachable. ***\n"
            "  No authentication, no TLS, no rate limiting is enabled.\n"
            "  Use a reverse proxy or set ZAYA_BIND_ADDR=127.0.0.1 for local-only access.\n\n");
    }
    if (!svr.listen(bind_addr, port)) {
        fprintf(stderr, "FATAL: failed to bind/listen on %s:%d\n", bind_addr, port);
        return 1;
    }
    return 0;
}
