// onebitd.cpp — 1bit inference daemon (pure C++ replacement for rust/onebitd)
//
// Spawns bitnet_decode (or zaya_server) as a subprocess and reverse-proxies
// OpenAI-compatible HTTP requests to it. No Rust, no Python — pure C++.
//
// Architecture:
//   httplib HTTP server (configurable port)
//     → spawns bitnet_decode --server as subprocess
//     → health-checks until ready
//     → proxies /v1/* requests with streaming passthrough
//
// Usage:
//   onebitd --model model.h1b --port 13305
//   onebitd --model model.q4nx --port 8080 --bitnet-decode ./build/zaya_server

#include <array>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <httplib.h>

// ── Configuration ────────────────────────────────────────────────────────
static constexpr int kDefaultPort        = 13305;
static constexpr int kHealthTimeoutSecs  = 120;

// ── Global state ─────────────────────────────────────────────────────────
static pid_t g_child_pid = -1;
static volatile sig_atomic_t g_shutdown = 0;

static void signal_handler(int) {
    g_shutdown = 1;
}

static void kill_child() {
    if (g_child_pid > 0) {
        kill(g_child_pid, SIGTERM);
        // Wait up to 5 seconds for graceful shutdown
        for (int i = 0; i < 50; ++i) {
            int status;
            if (waitpid(g_child_pid, &status, WNOHANG) == g_child_pid) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        // Force kill if still alive
        kill(g_child_pid, SIGKILL);
        waitpid(g_child_pid, nullptr, 0);
        g_child_pid = -1;
    }
}

// ── Find a free TCP port ────────────────────────────────────────────────
static int find_free_port() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    if (bind(sock, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
        close(sock);
        return -1;
    }

    socklen_t len = sizeof(addr);
    if (getsockname(sock, reinterpret_cast<struct sockaddr *>(&addr), &len) < 0) {
        close(sock);
        return -1;
    }

    int port = ntohs(addr.sin_port);
    close(sock);
    return port;
}

// ── Spawn the inference backend subprocess ───────────────────────────────
static pid_t spawn_backend(const std::string &bitnet_decode_path,
                           const std::string &model_path,
                           int backend_port,
                           bool tune_prefill,
                           int prefill_variant,
                           bool fp16_weights,
                           const std::string &extra_args) {
    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "onebitd: fork failed\n";
        return -1;
    }

    if (pid == 0) {
        // Child process — build argv and exec

        // Close all file descriptors except stdin/stdout/stderr
        int max_fd = static_cast<int>(sysconf(_SC_OPEN_MAX));
        for (int fd = 3; fd < max_fd; ++fd) close(fd);

        std::vector<std::string> arg_strings;
        arg_strings.push_back(bitnet_decode_path);
        arg_strings.push_back(model_path);
        arg_strings.push_back("--server");
        arg_strings.push_back(std::to_string(backend_port));
        arg_strings.push_back("--bind");
        arg_strings.push_back("127.0.0.1");

        if (tune_prefill) arg_strings.push_back("--tune-prefill");
        if (prefill_variant >= 0) {
            arg_strings.push_back("--prefill-variant");
            arg_strings.push_back(std::to_string(prefill_variant));
        }
        if (fp16_weights) arg_strings.push_back("--fp16-weights");

        // Parse extra args (whitespace-separated)
        if (!extra_args.empty()) {
            std::string args = extra_args;
            size_t pos = 0;
            while (pos < args.size()) {
                // Skip whitespace
                while (pos < args.size() && std::isspace(args[pos])) ++pos;
                if (pos >= args.size()) break;
                size_t end = pos;
                while (end < args.size() && !std::isspace(args[end])) ++end;
                arg_strings.push_back(args.substr(pos, end - pos));
                pos = end;
            }
        }

        // Build char* array
        std::vector<char *> argv;
        for (auto &s : arg_strings) argv.push_back(s.data());
        argv.push_back(nullptr);

        // Set ROCm environment for the child
        setenv("HSA_OVERRIDE_GFX_VERSION", "11.5.1", 1);
        setenv("HSA_ENABLE_SDMA", "0", 1);

        execvp(argv[0], argv.data());

        // execvp only returns on error
        std::cerr << "onebitd: failed to exec " << bitnet_decode_path << ": "
                  << std::strerror(errno) << "\n";
        _exit(1);
    }

    return pid;
}

// ── Wait for backend to become healthy ───────────────────────────────────
static bool wait_for_backend(const std::string &backend_url, int timeout_secs) {
    httplib::Client cli(backend_url);
    cli.set_connection_timeout(2);
    cli.set_read_timeout(2);

    std::cout << "onebitd: waiting for backend at " << backend_url << " ..." << std::endl;

    for (int i = 0; i < timeout_secs; ++i) {
        if (g_shutdown) return false;

        // Check if child exited early
        if (g_child_pid > 0) {
            int status;
            pid_t r = waitpid(g_child_pid, &status, WNOHANG);
            if (r == g_child_pid) {
                std::cerr << "onebitd: backend exited early with status "
                          << WEXITSTATUS(status) << "\n";
                g_child_pid = -1;
                return false;
            }
        }

        auto res = cli.Get("/health");
        if (res && res->status == 200) {
            std::cout << "onebitd: backend ready after " << i << "s" << std::endl;
            return true;
        }

        if (i > 0 && i % 10 == 0) {
            std::cout << "onebitd: still waiting... (" << i << "s)" << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cerr << "onebitd: backend health check timed out after " << timeout_secs << "s\n";
    return false;
}

// ── Path allowlist for proxy ─────────────────────────────────────────────
static bool is_allowed_path(const std::string &path) {
    static const std::array<std::string_view, 9> kAllowed = {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/models",
        "/v1/health",
        "/v1/backend/status",
        "/v1/embeddings",
        "/health",
        "/v1/models/",
        "/v1/backend",
    };

    for (auto &prefix : kAllowed) {
        if (path == prefix) return true;
        if (path.size() > prefix.size() && path.compare(0, prefix.size(), prefix) == 0) {
            // Allow sub-paths like /v1/models/name
            return true;
        }
    }
    return false;
}

// ── Main ─────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    std::string model_path = "./model.h1b";
    int port = kDefaultPort;
    std::string host = "127.0.0.1";
    std::string bitnet_decode = "bitnet_decode";
    int backend_port = 0;
    std::string extra_args;
    bool tune_prefill = false;
    int prefill_variant = -1;
    bool fp16_weights = false;

    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--bitnet-decode" && i + 1 < argc) {
            bitnet_decode = argv[++i];
        } else if (arg == "--backend-port" && i + 1 < argc) {
            backend_port = std::stoi(argv[++i]);
        } else if (arg == "--bitnet-args" && i + 1 < argc) {
            extra_args = argv[++i];
        } else if (arg == "--tune-prefill") {
            tune_prefill = true;
        } else if (arg == "--prefill-variant" && i + 1 < argc) {
            prefill_variant = std::stoi(argv[++i]);
        } else if (arg == "--fp16-weights") {
            fp16_weights = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << R"(onebitd — 1bit inference daemon for Strix Halo

Usage: onebitd [OPTIONS]

Options:
  -m, --model PATH        Path to model file (default: ./model.h1b)
  -p, --port PORT         Port for OpenAI-compatible API (default: 13305)
  --host ADDR             Host to bind to (default: 127.0.0.1)
  --bitnet-decode PATH    Path to bitnet_decode binary (default: bitnet_decode)
  --backend-port PORT     Internal port for backend (0=auto, default: 0)
  --bitnet-args ARGS      Extra args to pass to bitnet_decode
  --tune-prefill          Run prefill kernel auto-tuning at startup
  --prefill-variant N     Force a specific prefill variant
  --fp16-weights          Pre-decode weights to FP16 at load time
  -h, --help              Show this help
)";
            return 0;
        }
    }

    // Allocate a free port for the backend if not specified
    if (backend_port == 0) {
        backend_port = find_free_port();
        if (backend_port < 0) {
            std::cerr << "onebitd: failed to find a free port for backend\n";
            return 1;
        }
    }

    std::string backend_url = "http://127.0.0.1:" + std::to_string(backend_port);

    // Install signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Spawn the backend
    std::cout << "onebitd: starting " << bitnet_decode << " on port " << backend_port << "..." << std::endl;
    g_child_pid = spawn_backend(bitnet_decode, model_path, backend_port,
                                tune_prefill, prefill_variant, fp16_weights, extra_args);
    if (g_child_pid < 0) {
        std::cerr << "onebitd: failed to start backend\n";
        return 1;
    }

    // Wait for backend to be ready
    if (!wait_for_backend(backend_url, kHealthTimeoutSecs)) {
        kill_child();
        return 1;
    }

    // Build the HTTP proxy server
    httplib::Server svr;

    // Health endpoint
    svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
        res.set_content("ok", "text/plain");
    });

    // Proxy GET requests
    auto proxy_get = [&backend_url](const httplib::Request &req, httplib::Response &res) {
        if (!is_allowed_path(req.path)) {
            res.status = 404;
            res.set_content("unknown endpoint", "text/plain");
            return;
        }

        std::string target = backend_url + req.path;
        if (!req.target.empty() && req.target.find('?') != std::string::npos) {
            target += "?" + req.target.substr(req.target.find('?') + 1);
        }

        httplib::Client cli(backend_url);
        cli.set_connection_timeout(5);
        cli.set_read_timeout(300);

        auto backend_res = cli.Get(req.path.c_str());
        if (backend_res) {
            res.status = backend_res->status;
            res.set_content(backend_res->body,
                          backend_res->get_header_value("Content-Type"));
        } else {
            res.status = 502;
            res.set_content(R"({"error":"Backend unreachable"})", "application/json");
        }
    };

    svr.Get(R"(/v1/models.*)", proxy_get);
    svr.Get("/v1/health", proxy_get);
    svr.Get("/v1/backend/status", proxy_get);

    // Proxy POST requests
    auto proxy_post = [&backend_url](const httplib::Request &req, httplib::Response &res) {
        if (!is_allowed_path(req.path)) {
            res.status = 404;
            res.set_content("unknown endpoint", "text/plain");
            return;
        }

        httplib::Client cli(backend_url);
        cli.set_connection_timeout(5);
        cli.set_read_timeout(300);
        cli.set_write_timeout(300);

        httplib::Headers headers;
        // Pass through relevant headers
        if (req.has_header("Authorization")) {
            headers.emplace("Authorization", req.get_header_value("Authorization"));
        }
        headers.emplace("Content-Type", "application/json");

        auto backend_res = cli.Post(req.path.c_str(), headers, req.body, "application/json");
        if (backend_res) {
            res.status = backend_res->status;

            // Handle streaming responses
            auto ct = backend_res->get_header_value("Content-Type");
            if (ct.find("text/event-stream") != std::string::npos) {
                res.set_header("Content-Type", "text/event-stream");
                res.set_header("Cache-Control", "no-cache");
                res.set_header("Connection", "keep-alive");
            }
            res.set_content(backend_res->body, ct);
        } else {
            res.status = 502;
            res.set_content(R"({"error":"Backend unreachable"})", "application/json");
        }
    };

    svr.Post("/v1/chat/completions", proxy_post);
    svr.Post("/v1/completions", proxy_post);
    svr.Post("/v1/embeddings", proxy_post);

    // OPTIONS for CORS
    svr.Options(R"(/.*)", [](const httplib::Request &, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.status = 200;
    });

    auto addr = host + ":" + std::to_string(port);
    std::cout << "onebitd: listening on http://" << addr << std::endl;
    std::cout << "onebitd: backend at " << backend_url << std::endl;

    // Start server in a separate thread so we can monitor shutdown signal
    std::thread server_thread([&]() {
        svr.listen(host.c_str(), port);
    });

    // Wait for shutdown signal
    while (!g_shutdown) {
        // Check if child exited
        if (g_child_pid > 0) {
            int status;
            pid_t r = waitpid(g_child_pid, &status, WNOHANG);
            if (r == g_child_pid) {
                std::cerr << "onebitd: backend exited (status="
                          << WEXITSTATUS(status) << "), shutting down\n";
                g_shutdown = 1;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    std::cout << "onebitd: shutting down..." << std::endl;
    svr.stop();
    if (server_thread.joinable()) server_thread.join();

    kill_child();
    std::cout << "onebitd: stopped." << std::endl;

    return 0;
}
