// onebit.cpp — 1bit terminal coding agent (pure C++ replacement for rust/onebit-cli)
//
// A single static binary that replaces the Rust CLI and the old TypeScript wrapper.
// Handles NPU stack management, interactive chat, config, and engine builds.
//
// Architecture:
//   1bit chat        → interactive REPL → NPU API chat completions
//   1bit up          → spawn onebitd + zaya_server daemon
//   1bit down        → kill NPU processes
//   1bit status      → check NPU stack health
//   1bit build       → compile NPU engine from source
//   1bit config      → view / set settings
//   1bit serve       → agent runtime HTTP/SSE server
//   1bit auth        → manage API keys
//   1bit update      → check for updates
//
// Usage:
//   1bit "explain this code"     → non-interactive chat
//   1bit chat                    → interactive REPL
//   1bit up                      → start NPU stack

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

static constexpr const char *kVersion = "2026.07.22";
static constexpr const char *kDefaultEndpoint = "http://127.0.0.1:9090/v1";

static const char *kBanner = R"(
  ╔══════════════════════════════════════════╗
  ║                                          ║
  ║    ██   ██████╗  ██╗  ████████╗         ║
  ║    ██   ██╔══██╗  ██║  ╚══██╔══╝        ║
  ║    ██   ██████╔╝  ██║     ██║           ║
  ║    ██   ██╔══██╗  ██║     ██║           ║
  ║   ██████ ██████╔╝  ██║     ██║          ║
  ║   ╚═════ ╚═════╝   ╚═╝     ╚═╝          ║
  ║                                          ║
  ║       NPU-native coding agent            ║
  ║    50 TOPS · 94 tok/s · 0 cloud          ║
  ║              v%s                          ║
  ╚══════════════════════════════════════════╝
)";

// ── Configuration ────────────────────────────────────────────────────────

static fs::path config_dir() {
    const char *home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    return fs::path(home ? home : ".") / ".1bit";
}

static fs::path settings_path() {
    return config_dir() / "agent" / "settings.json";
}

struct Settings {
    std::string theme = "1bit";
    std::string default_provider = "npu";
    std::string default_model = "qwen3-0.6b-FLM";
    std::string npu_endpoint = kDefaultEndpoint;
    std::string thinking_level = "medium";
    std::string bitnet_decode_path = "bitnet_decode";
    std::string daemon_path;
    bool tune_prefill = false;
    int prefill_variant = -1;
    bool fp16_weights = false;
    int api_port = 9090;
    int lemond_port = 13305;

    void load() {
        auto path = settings_path();
        if (!fs::exists(path)) return;
        std::ifstream f(path);
        if (!f) return;
        try {
            json j = json::parse(f);
            if (j.contains("theme")) theme = j["theme"];
            if (j.contains("default_model")) default_model = j["default_model"];
            if (j.contains("npu_endpoint")) npu_endpoint = j["npu_endpoint"];
            if (j.contains("thinking_level")) thinking_level = j["thinking_level"];
            if (j.contains("bitnet_decode_path")) bitnet_decode_path = j["bitnet_decode_path"];
            if (j.contains("tune_prefill")) tune_prefill = j["tune_prefill"];
            if (j.contains("prefill_variant")) prefill_variant = j["prefill_variant"];
            if (j.contains("fp16_weights")) fp16_weights = j["fp16_weights"];
            if (j.contains("api_port")) api_port = j["api_port"];
            if (j.contains("lemond_port")) lemond_port = j["lemond_port"];
        } catch (...) {}
    }

    void save() {
        auto path = settings_path();
        fs::create_directories(path.parent_path());
        json j;
        j["theme"] = theme;
        j["default_model"] = default_model;
        j["npu_endpoint"] = npu_endpoint;
        j["thinking_level"] = thinking_level;
        j["bitnet_decode_path"] = bitnet_decode_path;
        j["tune_prefill"] = tune_prefill;
        j["prefill_variant"] = prefill_variant;
        j["fp16_weights"] = fp16_weights;
        j["api_port"] = api_port;
        j["lemond_port"] = lemond_port;
        std::ofstream f(path);
        f << j.dump(2) << "\n";
    }

    std::string get(const std::string &key) {
        if (key == "theme") return theme;
        if (key == "default_model") return default_model;
        if (key == "npu_endpoint") return npu_endpoint;
        if (key == "thinking_level") return thinking_level;
        if (key == "bitnet_decode_path") return bitnet_decode_path;
        if (key == "tune_prefill") return tune_prefill ? "true" : "false";
        if (key == "fp16_weights") return fp16_weights ? "true" : "false";
        if (key == "api_port") return std::to_string(api_port);
        if (key == "lemond_port") return std::to_string(lemond_port);
        return "";
    }

    void set(const std::string &key, const std::string &value) {
        if (key == "theme") theme = value;
        else if (key == "default_model") default_model = value;
        else if (key == "npu_endpoint") npu_endpoint = value;
        else if (key == "thinking_level") thinking_level = value;
        else if (key == "bitnet_decode_path") bitnet_decode_path = value;
        else if (key == "tune_prefill") tune_prefill = (value == "true" || value == "1");
        else if (key == "fp16_weights") fp16_weights = (value == "true" || value == "1");
        else if (key == "api_port") api_port = std::stoi(value);
        else if (key == "lemond_port") lemond_port = std::stoi(value);
        save();
    }
};

// ── NPU API Client ──────────────────────────────────────────────────────

struct NpuClient {
    std::string base_url;

    explicit NpuClient(const std::string &url) : base_url(url) {
        while (!base_url.empty() && base_url.back() == '/') base_url.pop_back();
    }

    bool health_check() {
        httplib::Client cli(base_url);
        cli.set_connection_timeout(3);
        cli.set_read_timeout(3);

        // Try /health first
        std::string health_url = base_url;
        auto v1pos = health_url.rfind("/v1");
        if (v1pos != std::string::npos) health_url = health_url.substr(0, v1pos);
        {
            httplib::Client hcli(health_url);
            auto res = hcli.Get("/health");
            if (res && res->status == 200) return true;
        }

        // Try /v1/models
        auto res = cli.Get("/models");
        return res && res->status == 200;
    }

    std::vector<std::string> list_models() {
        std::vector<std::string> models;
        httplib::Client cli(base_url);
        cli.set_connection_timeout(5);
        auto res = cli.Get("/models");
        if (!res || res->status != 200) return models;
        try {
            json j = json::parse(res->body);
            if (j.contains("data") && j["data"].is_array()) {
                for (auto &m : j["data"]) {
                    if (m.contains("id")) models.push_back(m["id"]);
                }
            }
        } catch (...) {}
        return models;
    }

    std::string chat(const std::string &model, const std::string &prompt) {
        httplib::Client cli(base_url);
        cli.set_connection_timeout(10);
        cli.set_read_timeout(300);

        json body;
        body["model"] = model;
        body["messages"] = json::array({json::object({
            {"role", "user"},
            {"content", prompt}
        })});
        body["stream"] = false;

        auto res = cli.Post("/chat/completions",
                           httplib::Headers{{"Content-Type", "application/json"}},
                           body.dump(), "application/json");

        if (!res || res->status != 200) {
            return "Error: NPU API returned status " +
                   (res ? std::to_string(res->status) : "connection failed");
        }

        try {
            json j = json::parse(res->body);
            if (j.contains("choices") && j["choices"].is_array() &&
                !j["choices"].empty()) {
                auto &choice = j["choices"][0];
                if (choice.contains("message") && choice["message"].contains("content")) {
                    return choice["message"]["content"];
                }
            }
        } catch (...) {}

        return res->body;
    }
};

// ── Process management ──────────────────────────────────────────────────

static void cmd_status() {
    // Check if onebitd or zaya_server is running
    std::cout << "  🔍 Checking NPU stack...\n";

    // Check via ps
    FILE *fp = popen("ps aux | grep -E '(onebitd|zaya_server|bitnet_decode)' | grep -v grep", "r");
    if (!fp) {
        std::cout << "  ⚠️  Could not check processes\n";
        return;
    }

    char buf[1024];
    bool found = false;
    while (fgets(buf, sizeof(buf), fp)) {
        std::cout << "  📡 " << buf;
        found = true;
    }
    pclose(fp);

    if (!found) {
        std::cout << "  ℹ️  NPU stack is not running. Type '1bit up' to start.\n";
    }

    // Check API health
    NpuClient client(kDefaultEndpoint);
    if (client.health_check()) {
        std::cout << "  ✅ NPU API is responding at " << kDefaultEndpoint << "\n";
        auto models = client.list_models();
        if (!models.empty()) {
            std::cout << "  📦 Available models:\n";
            for (auto &m : models) std::cout << "     • " << m << "\n";
        }
    }
}

static void cmd_up() {
    std::cout << "  🚀 Starting NPU stack...\n";

    // Check if already running
    NpuClient client(kDefaultEndpoint);
    if (client.health_check()) {
        std::cout << "  ✅ NPU stack is already running.\n";
        return;
    }

    // Spawn onebitd in background
    pid_t pid = fork();
    if (pid == 0) {
        // Child: run onebitd
        setsid(); // Daemonize
        execlp("onebitd", "onebitd",
               "--port", std::to_string(Settings().lemond_port).c_str(),
               "--bitnet-decode", "zaya_server",
               nullptr);
        std::cerr << "  ⚠️  Failed to start onebitd\n";
        _exit(1);
    } else if (pid > 0) {
        std::cout << "  ✅ onebitd started (pid " << pid << ")\n";
        std::cout << "  ⏳ Waiting for API to come online...\n";

        // Wait for health
        for (int i = 0; i < 30; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (client.health_check()) {
                std::cout << "  ✅ NPU API is ready at " << kDefaultEndpoint << "\n";
                return;
            }
            if (i % 5 == 4) std::cout << "  ⏳ Still waiting... (" << (i + 1) << "s)\n";
        }
        std::cout << "  ⚠️  NPU API did not come online within 30s.\n";
    }
}

static void cmd_down() {
    std::cout << "  🛑 Stopping NPU stack...\n";
    system("pkill -f onebitd 2>/dev/null");
    system("pkill -f zaya_server 2>/dev/null");
    system("pkill -f bitnet_decode 2>/dev/null");
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "  ✅ NPU stack stopped.\n";
}

static void cmd_build(const std::string &dir) {
    std::string build_dir = dir.empty() ? "engine/npu" : dir;
    std::cout << "  🔨 Building NPU engine in " << build_dir << "...\n";

    std::string cmd = "cd " + build_dir + " && cmake -B build -G Ninja && ninja -C build";
    int ret = system(cmd.c_str());
    if (ret == 0) {
        std::cout << "  ✅ Build complete.\n";
    } else {
        std::cout << "  ⚠️  Build failed (exit code " << ret << ").\n";
    }
}

// ── Interactive chat ────────────────────────────────────────────────────

static void cmd_chat(const std::string &model_override) {
    Settings settings;
    settings.load();

    std::string model = model_override.empty() ? settings.default_model : model_override;

    NpuClient client(settings.npu_endpoint);
    bool npu_online = client.health_check();

    printf(kBanner, kVersion);
    if (npu_online) {
        std::cout << "  ✅ NPU stack is online — ask me anything.\n\n";
    } else {
        std::cout << "  ℹ️  NPU stack is not running. Type /up to start it.\n\n";
    }

    std::cout << "  Type /help for commands, /exit to quit.\n\n";

    std::string line;
    while (true) {
        std::cout << "1bit> " << std::flush;
        if (!std::getline(std::cin, line)) break;

        std::string trimmed = line;
        // Trim whitespace
        while (!trimmed.empty() && std::isspace(trimmed.front())) trimmed.erase(0, 1);
        while (!trimmed.empty() && std::isspace(trimmed.back())) trimmed.pop_back();

        if (trimmed.empty()) continue;

        // Handle slash commands
        if (trimmed[0] == '/') {
            if (trimmed == "/exit" || trimmed == "/quit") break;
            if (trimmed == "/help") {
                std::cout << R"(
  Commands:
    /help              Show this help
    /status            Check NPU stack health
    /up                Start NPU stack (onebitd + zaya_server)
    /down              Stop NPU stack
    /models            List available models
    /clear             Clear screen
    /exit, /quit       Exit 1bit chat
)";
                continue;
            }
            if (trimmed == "/status") { cmd_status(); continue; }
            if (trimmed == "/up") { cmd_up(); continue; }
            if (trimmed == "/down") { cmd_down(); continue; }
            if (trimmed == "/models") {
                auto models = client.list_models();
                if (models.empty()) {
                    std::cout << "  ℹ️  No models available\n";
                } else {
                    std::cout << "  Available models:\n";
                    for (auto &m : models) std::cout << "    • " << m << "\n";
                }
                continue;
            }
            if (trimmed == "/clear") {
                std::cout << "\033[2J\033[1;1H";
                printf(kBanner, kVersion);
                continue;
            }
            std::cout << "  Unknown command: " << trimmed << "\n";
            continue;
        }

        // Send to NPU
        if (!client.health_check()) {
            std::cout << "  ⚠️  NPU stack not running. Type /up to start.\n\n";
            continue;
        }

        std::cout << "  🤔 Thinking...\n";
        std::string response = client.chat(model, trimmed);
        std::cout << "\n  " << response << "\n\n";
    }

    std::cout << "\n  👋 Goodbye!\n";
}

// ── Config management ───────────────────────────────────────────────────

static void cmd_config(const std::string &key, const std::string &value) {
    Settings settings;
    settings.load();

    if (key.empty()) {
        // Show all settings
        std::cout << "  Current configuration:\n";
        std::cout << "    theme              = " << settings.theme << "\n";
        std::cout << "    default_model      = " << settings.default_model << "\n";
        std::cout << "    npu_endpoint       = " << settings.npu_endpoint << "\n";
        std::cout << "    thinking_level     = " << settings.thinking_level << "\n";
        std::cout << "    bitnet_decode_path = " << settings.bitnet_decode_path << "\n";
        std::cout << "    tune_prefill       = " << (settings.tune_prefill ? "true" : "false") << "\n";
        std::cout << "    fp16_weights       = " << (settings.fp16_weights ? "true" : "false") << "\n";
        std::cout << "    api_port           = " << settings.api_port << "\n";
        std::cout << "    lemond_port        = " << settings.lemond_port << "\n";
        return;
    }

    if (value.empty()) {
        // Get value
        std::string v = settings.get(key);
        if (v.empty()) {
            std::cout << "  ⚠️  Unknown config key: " << key << "\n";
        } else {
            std::cout << "  " << key << " = " << v << "\n";
        }
    } else {
        // Set value
        settings.set(key, value);
        std::cout << "  ✅ Set " << key << " = " << value << "\n";
    }
}

// ── Auth management ─────────────────────────────────────────────────────

static void cmd_auth(const std::string &provider, const std::string &apikey,
                     bool remove, bool list) {
    auto secrets_path = config_dir() / "secrets" / "secrets.json";

    if (list) {
        if (!fs::exists(secrets_path)) {
            std::cout << "  ℹ️  No stored API keys.\n";
            return;
        }
        std::ifstream f(secrets_path);
        json j = json::parse(f);
        std::cout << "  Stored API keys:\n";
        for (auto &[k, v] : j.items()) {
            std::string masked = v.get<std::string>();
            if (masked.size() > 8) {
                masked = masked.substr(0, 4) + "..." + masked.substr(masked.size() - 4);
            }
            std::cout << "    " << k << " = " << masked << "\n";
        }
        return;
    }

    fs::create_directories(secrets_path.parent_path());

    json secrets;
    if (fs::exists(secrets_path)) {
        std::ifstream f(secrets_path);
        secrets = json::parse(f, nullptr, false);
        if (secrets.is_discarded()) secrets = json::object();
    }

    if (remove) {
        secrets.erase(provider);
        std::cout << "  ✅ Removed API key for " << provider << "\n";
    } else if (!apikey.empty()) {
        secrets[provider] = apikey;
        std::cout << "  ✅ Stored API key for " << provider << "\n";
    } else if (!provider.empty()) {
        // Read from stdin
        std::cout << "  Enter API key for " << provider << ": " << std::flush;
        std::string key;
        std::getline(std::cin, key);
        if (!key.empty()) {
            secrets[provider] = key;
            std::cout << "  ✅ Stored API key for " << provider << "\n";
        }
    }

    std::ofstream f(secrets_path);
    f << secrets.dump(2) << "\n";
    chmod(secrets_path.c_str(), 0600);
}

// ── HTTP/SSE serve ──────────────────────────────────────────────────────

static void cmd_serve(int port, const std::string &host, const std::string &model) {
    Settings settings;
    settings.load();

    std::string use_model = model.empty() ? settings.default_model : model;
    NpuClient client(settings.npu_endpoint);

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Post("/v1/chat/completions",
             [&client, &use_model](const httplib::Request &req, httplib::Response &res) {
        json body;
        try {
            body = json::parse(req.body.empty() ? "{}" : req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"Invalid JSON"})", "application/json");
            return;
        }

        // Extract messages
        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(R"({"error":"messages required"})", "application/json");
            return;
        }

        // Get last user message
        std::string prompt;
        for (auto &msg : body["messages"]) {
            if (msg.contains("role") && msg["role"] == "user" && msg.contains("content")) {
                prompt = msg["content"];
            }
        }

        std::string result = client.chat(use_model, prompt);

        json resp;
        resp["choices"] = json::array({json::object({
            {"message", json::object({
                {"role", "assistant"},
                {"content", result}
            })}
        })});
        res.set_content(resp.dump(), "application/json");
    });

    std::cout << "  🌐 Agent server listening on http://" << host << ":" << port << "\n";
    std::cout << "  📡 Using model: " << use_model << "\n";
    svr.listen(host.c_str(), port);
}

// ── Update ───────────────────────────────────────────────────────────────

static void cmd_update(bool check_only) {
    std::cout << "  🔄 Checking for updates...\n";

    // Check GitHub releases
    httplib::Client cli("https://api.github.com");
    cli.set_connection_timeout(10);
    cli.set_read_timeout(10);

    httplib::Headers headers;
    headers.emplace("User-Agent", "1bit-cli/" + std::string(kVersion));
    headers.emplace("Accept", "application/vnd.github.v3+json");

    auto res = cli.Get("/repos/bong-water-water-bong/1bit-systems/releases/latest", headers);
    if (!res || res->status != 200) {
        std::cout << "  ⚠️  Could not check for updates.\n";
        return;
    }

    try {
        json j = json::parse(res->body);
        std::string latest = j.value("tag_name", "unknown");
        std::cout << "  Current: v" << kVersion << "\n";
        std::cout << "  Latest:  " << latest << "\n";

        if (latest != ("v" + std::string(kVersion))) {
            if (check_only) {
                std::cout << "  📦 Update available!\n";
            } else {
                std::cout << "  📦 Update available! Run: git pull && cmake --build build\n";
            }
        } else {
            std::cout << "  ✅ Already up to date.\n";
        }
    } catch (...) {
        std::cout << "  ⚠️  Could not parse release info.\n";
    }
}

// ── Main ─────────────────────────────────────────────────────────────────

static void print_usage() {
    std::cout << R"(1bit — NPU-native coding agent for 1bit.systems

Usage: 1bit [COMMAND] [ARGS]

Commands:
  chat                Start interactive agent session (default)
  up                  Start NPU stack (onebitd + zaya_server)
  down                Stop NPU stack
  status              Show NPU stack status
  build [DIR]         Build NPU engine from source (default: engine/npu)
  config [KEY [VAL]]  View or set configuration
  auth PROVIDER       Manage API keys
  serve               Serve agent runtime over HTTP/SSE
  update              Check for updates
  --help, -h          Show this help
  --version, -v       Show version

Without a command, runs as interactive chat.
With a prompt argument, runs a single non-interactive query.

Examples:
  1bit                        Interactive chat
  1bit "explain this code"    Single query
  1bit up                     Start NPU stack
  1bit status                 Check stack health
  1bit config                 Show all settings
  1bit config default_model   Get a setting
  1bit config default_model gpt-4  Set a setting
  1bit auth deepseek          Store DeepSeek API key
  1bit serve --port 7878      Start HTTP server
)";
}

int main(int argc, char *argv[]) {
    std::string command;
    std::vector<std::string> args;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        if (arg == "--version" || arg == "-v") {
            std::cout << "1bit v" << kVersion << "\n";
            return 0;
        }
        if (arg[0] != '-' && command.empty()) {
            // First non-flag arg could be a command
            if (arg == "chat" || arg == "up" || arg == "down" ||
                arg == "status" || arg == "build" || arg == "config" ||
                arg == "auth" || arg == "serve" || arg == "update") {
                command = arg;
                continue;
            }
        }
        args.push_back(std::string(arg));
    }

    // Collect remaining as prompt (for non-interactive mode)
    std::string prompt;
    for (auto &a : args) {
        if (!prompt.empty()) prompt += " ";
        prompt += a;
    }

    if (command.empty() && !prompt.empty()) {
        // Non-interactive chat mode
        Settings settings;
        settings.load();
        NpuClient client(settings.npu_endpoint);

        if (!client.health_check()) {
            std::cerr << "⚠️  NPU stack is not running. Run '1bit up' to start.\n";
            return 1;
        }

        std::string response = client.chat(settings.default_model, prompt);
        std::cout << response << std::endl;
        return 0;
    }

    if (command.empty()) {
        command = "chat";
    }

    if (command == "chat") {
        cmd_chat(args.empty() ? "" : args[0]);
    } else if (command == "up") {
        cmd_up();
    } else if (command == "down") {
        cmd_down();
    } else if (command == "status") {
        cmd_status();
    } else if (command == "build") {
        cmd_build(args.empty() ? "" : args[0]);
    } else if (command == "config") {
        std::string key = args.size() > 0 ? args[0] : "";
        std::string val = args.size() > 1 ? args[1] : "";
        cmd_config(key, val);
    } else if (command == "auth") {
        std::string provider = args.size() > 0 ? args[0] : "";
        std::string apikey;
        bool remove = false;
        bool list = false;
        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--set" && i + 1 < args.size()) apikey = args[++i];
            else if (args[i] == "--remove") remove = true;
            else if (args[i] == "--list") list = true;
        }
        cmd_auth(provider, apikey, remove, list);
    } else if (command == "serve") {
        int port = 7878;
        std::string host = "127.0.0.1";
        std::string model;
        for (size_t i = 0; i < args.size(); ++i) {
            if ((args[i] == "-p" || args[i] == "--port") && i + 1 < args.size())
                port = std::stoi(args[++i]);
            else if (args[i] == "--host" && i + 1 < args.size())
                host = args[++i];
            else if ((args[i] == "-m" || args[i] == "--model") && i + 1 < args.size())
                model = args[++i];
        }
        cmd_serve(port, host, model);
    } else if (command == "update") {
        cmd_update(args.size() > 0 && (args[0] == "--check" || args[0] == "-c"));
    } else {
        std::cerr << "Unknown command: " << command << "\n";
        print_usage();
        return 1;
    }

    return 0;
}
