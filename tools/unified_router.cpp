// unified_router.cpp — Content-aware routing proxy for 1bit systems
//
// Routes requests between NPU (small model) and GPU (large model) backends
// based on content complexity analysis. Pure C++ replacement for unified-router.py.
//
// Usage:
//   unified_router --port 18181 --npu-backend http://127.0.0.1:18101 --gpu-backend http://127.0.0.1:18102

#include <algorithm>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Configuration ────────────────────────────────────────────────────────
static constexpr size_t kMaxBodySize = 16 * 1024 * 1024;  // 16 MB

static std::string g_npu_url;
static std::string g_gpu_url;
static std::string g_small_model = "zaya-small";
static std::string g_big_model   = "zaya-large";
static std::string g_router_name = "unified-router";

// Keyword-based routing: if a message contains any of these keywords, route to GPU
static const std::unordered_set<std::string> kGpuKeywords = {
    "explain", "summarize", "write", "create", "design", "review", "analyze",
    "code", "function", "class", "implement", "refactor", "debug", "architecture",
    "compare", "evaluate", "recommend", "translate",
};

// ── Helpers ─────────────────────────────────────────────────────────────
static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

static std::vector<std::string> split_words(std::string_view text) {
    std::vector<std::string> words;
    std::string current;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else {
            if (!current.empty()) {
                words.push_back(std::move(current));
                current.clear();
            }
        }
    }
    if (!current.empty()) words.push_back(std::move(current));
    return words;
}

static bool should_route_to_gpu(const json &body) {
    // Check for explicit model routing
    auto model_it = body.find("model");
    if (model_it != body.end() && model_it->is_string()) {
        std::string model = to_lower(model_it->get<std::string>());
        if (model == g_big_model || model == "gpu") return true;
        if (model == g_small_model || model == "npu") return false;
    }

    // Check for content-based routing via messages
    const json *messages = nullptr;
    auto msg_it = body.find("messages");
    if (msg_it != body.end() && msg_it->is_array()) {
        messages = &(*msg_it);
    } else {
        auto prompt_it = body.find("prompt");
        if (prompt_it != body.end() && prompt_it->is_string()) {
            static json fake_msgs = json::array();
            fake_msgs = json::array({json::object({{"content", prompt_it->get<std::string>()}})});
            messages = &fake_msgs;
        }
    }

    if (messages) {
        for (const auto &msg : *messages) {
            auto content_it = msg.find("content");
            if (content_it == msg.end() || !content_it->is_string()) continue;
            std::string content = content_it->get<std::string>();
            auto words = split_words(content);
            for (const auto &word : words) {
                if (kGpuKeywords.count(word)) return true;
                // Check for keyword as prefix
                for (const auto &kw : kGpuKeywords) {
                    if (word.size() >= kw.size() &&
                        word.compare(0, kw.size(), kw) == 0) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// ── Proxy helpers ───────────────────────────────────────────────────────
static std::string build_target_url(const std::string &backend, const std::string &path) {
    std::string url = backend;
    if (!url.empty() && url.back() == '/') url.pop_back();
    return url + path;
}

static std::pair<int, std::string> proxy_request(
    const std::string &backend_url,
    const std::string &path,
    const std::string &method,
    const std::string &body,
    const std::string &content_type = "application/json")
{
    httplib::Client cli(backend_url);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(300);
    cli.set_write_timeout(300);

    httplib::Headers headers;
    if (!content_type.empty()) {
        headers.emplace("Content-Type", content_type);
    }

    httplib::Result res;
    if (method == "GET") {
        res = cli.Get(path.c_str(), headers);
    } else if (method == "POST") {
        res = cli.Post(path.c_str(), headers, body, content_type.c_str());
    } else {
        return {405, R"({"error":"method not allowed"})"};
    }

    if (res) {
        return {res->status, res->body};
    } else {
        auto err = httplib::to_string(res.error());
        return {502, R"({"error":"Backend unreachable: )" + err + R"("})"};
    }
}

// ── Streaming SSE proxy for chat completions ────────────────────────────
static void proxy_sse_stream(httplib::Response &resp,
                             const std::string &gpu_url,
                             const std::string &path,
                             const std::string &body)
{
    std::string target = build_target_url(gpu_url, path);
    httplib::Client cli(gpu_url);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(300);
    cli.set_write_timeout(300);

    auto res = cli.Post(path.c_str(),
                        httplib::Headers{{"Content-Type", "application/json"}},
                        body, "application/json");

    if (!res) {
        resp.status = 502;
        resp.set_content(R"({"error":"Backend unreachable"})", "application/json");
        return;
    }

    // Stream back as SSE
    resp.status = res->status;
    resp.set_header("Content-Type", "text/event-stream");
    resp.set_header("Cache-Control", "no-cache");
    resp.set_header("Connection", "keep-alive");
    resp.set_header("Access-Control-Allow-Origin", "127.0.0.1");

    // httplib will handle the body; for true streaming we'd need chunked transfer,
    // but for the initial port we pass the complete response through.
    resp.set_content(res->body, "text/event-stream");
}

// ── Main ────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    // Parse CLI arguments
    int port = 18180;
    std::string bind_addr = "127.0.0.1";

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--bind" && i + 1 < argc) {
            bind_addr = argv[++i];
        } else if (arg == "--npu-backend" && i + 1 < argc) {
            g_npu_url = argv[++i];
        } else if (arg == "--gpu-backend" && i + 1 < argc) {
            g_gpu_url = argv[++i];
        } else if (arg == "--small-model" && i + 1 < argc) {
            g_small_model = argv[++i];
        } else if (arg == "--large-model" && i + 1 < argc) {
            g_big_model = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << R"(unified_router — Content-aware NPU+GPU routing proxy

Usage: unified_router [OPTIONS]

Options:
  --port PORT           Listen port (default: 18180)
  --bind ADDR           Bind address (default: 127.0.0.1; use 0.0.0.0 to expose)
  --npu-backend URL     NPU backend URL (default: http://127.0.0.1:18101)
  --gpu-backend URL     GPU backend URL (default: http://127.0.0.1:18102)
  --small-model NAME    Small model name (default: zaya-small)
  --large-model NAME    Large model name (default: zaya-large)
  --help, -h            Show this help
)";
            return 0;
        }
    }

    // Apply defaults from environment
    if (g_npu_url.empty()) {
        const char *env = std::getenv("NPU_BACKEND_URL");
        g_npu_url = env ? env : "http://127.0.0.1:18101";
    }
    if (g_gpu_url.empty()) {
        const char *env = std::getenv("GPU_BACKEND_URL");
        g_gpu_url = env ? env : "http://127.0.0.1:18102";
    }
    {
        const char *env = std::getenv("SMALL_MODEL");
        if (env) g_small_model = env;
    }
    {
        const char *env = std::getenv("BIG_MODEL");
        if (env) g_big_model = env;
    }

    if (bind_addr != "127.0.0.1") {
        std::cerr << "WARNING: Binding to non-localhost address. Ensure firewall rules are in place.\n";
    }

    std::cout << "==========================================================\n";
    std::cout << "  Unified NPU+GPU Router\n";
    std::cout << "==========================================================\n";
    std::cout << "  Listen:  http://" << bind_addr << ":" << port << "\n";
    std::cout << "  NPU:     " << g_npu_url << " (" << g_small_model << ")\n";
    std::cout << "  GPU:     " << g_gpu_url << " (" << g_big_model << ")\n";
    std::cout << "  Routing: keyword-based → GPU; default → NPU\n";
    std::cout << "    npu                  → " << g_small_model << " (NPU)\n";
    std::cout << "    gpu                  → " << g_big_model << " (GPU)\n";
    std::cout << "    <any other>          → pass-through to inference backend\n";
    std::cout << std::endl;

    // Build server
    httplib::Server svr;

    // GET /health, GET /
    svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
        json j;
        j["status"] = "ok";
        j["router"] = g_router_name;
        res.set_content(j.dump(), "application/json");
    });
    svr.Get("/", [](const httplib::Request &, httplib::Response &res) {
        json j;
        j["status"] = "ok";
        j["router"] = g_router_name;
        res.set_content(j.dump(), "application/json");
    });

    // GET /v1/models
    svr.Get("/v1/models", [](const httplib::Request &, httplib::Response &res) {
        auto [status, body] = proxy_request(g_gpu_url, "/v1/models", "GET", "");
        res.status = status;
        res.set_content(body, "application/json");
    });

    // POST /v1/completions
    svr.Post("/v1/completions", [](const httplib::Request &req, httplib::Response &res) {
        if (req.body.size() > kMaxBodySize) {
            res.status = 413;
            res.set_content(R"({"error":"Payload too large"})", "application/json");
            return;
        }

        json body;
        try {
            body = json::parse(req.body.empty() ? "{}" : req.body);
        } catch (const json::parse_error &e) {
            res.status = 400;
            res.set_content(R"({"error":"Invalid JSON: )" + std::string(e.what()) + R"("})", "application/json");
            return;
        }

        std::string model;
        auto model_it = body.find("model");
        if (model_it != body.end() && model_it->is_string()) {
            model = to_lower(model_it->get<std::string>());
        }

        if (model == g_router_name || model == "auto") {
            body["model"] = should_route_to_gpu(body) ? g_big_model : g_small_model;
        } else if (model == "npu") {
            body["model"] = g_small_model;
        } else if (model == "gpu") {
            body["model"] = g_big_model;
        }

        std::string target_url = g_gpu_url;  // default passthrough
        if (body["model"] == g_small_model) target_url = g_npu_url;

        auto [status, resp_body] = proxy_request(target_url, "/v1/completions", "POST", body.dump());
        res.status = status;
        res.set_content(resp_body, "application/json");
    });

    // POST /v1/chat/completions
    svr.Post("/v1/chat/completions", [](const httplib::Request &req, httplib::Response &res) {
        if (req.body.size() > kMaxBodySize) {
            res.status = 413;
            res.set_content(R"({"error":"Payload too large"})", "application/json");
            return;
        }

        json body;
        try {
            body = json::parse(req.body.empty() ? "{}" : req.body);
        } catch (const json::parse_error &) {
            body = json::object();
        }

        std::string model;
        auto model_it = body.find("model");
        if (model_it != body.end() && model_it->is_string()) {
            model = to_lower(model_it->get<std::string>());
        }

        bool stream = false;
        auto stream_it = body.find("stream");
        if (stream_it != body.end() && stream_it->is_boolean()) {
            stream = stream_it->get<bool>();
        }

        // Determine target backend
        std::string target;
        if (model == g_small_model || model == "npu") {
            target = g_npu_url;
        } else if (model == g_big_model || model == "gpu") {
            target = g_gpu_url;
        } else if (model == g_router_name || model == "auto" || model.empty()) {
            if (should_route_to_gpu(body)) {
                target = g_gpu_url;
                body["model"] = g_big_model;
            } else {
                target = g_npu_url;
                body["model"] = g_small_model;
            }
        } else {
            target = g_gpu_url;  // passthrough
        }

        if (stream) {
            proxy_sse_stream(res, target, "/v1/chat/completions", body.dump());
        } else {
            auto [status, resp_body] = proxy_request(target, "/v1/chat/completions", "POST", body.dump());
            res.status = status;
            res.set_header("Access-Control-Allow-Origin", "127.0.0.1");
            res.set_content(resp_body, "application/json");
        }
    });

    // OPTIONS handler for CORS
    svr.Options(R"(/.*)", [](const httplib::Request &, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "127.0.0.1");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.status = 200;
    });

    // Signal handling
    std::signal(SIGINT, [](int) { std::exit(0); });
    std::signal(SIGTERM, [](int) { std::exit(0); });

    std::cout << "Router listening on " << bind_addr << ":" << port << std::endl;
    if (!svr.listen(bind_addr.c_str(), port)) {
        std::cerr << "Failed to bind to " << bind_addr << ":" << port << std::endl;
        return 1;
    }

    return 0;
}
