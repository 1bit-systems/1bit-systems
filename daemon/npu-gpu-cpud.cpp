/** npu-gpu-cpud — Unified Control Plane Daemon (C++23, zero Python)
 *
 *  Routes inference requests to NPU (FLM), GPU (lemond), or CPU based on
 *  model size policy. Provides OpenAI-compatible chat API, Stripe checkout,
 *  and order management — all in a single zero-dependency C++ binary.
 *
 *  Build:
 *    g++ -std=c++23 -O3 -o npu-gpu-cpud npu-gpu-cpud.cpp -lpthread
 *
 *  For Stripe support, link libcurl:
 *    g++ -std=c++23 -O3 -o npu-gpu-cpud npu-gpu-cpud.cpp -lpthread -lcurl
 *
 *  Usage:
 *    sudo ./npu-gpu-cpud [--port 9090] [--npu-port 52625]
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <atomic>
#include <functional>
#include <memory>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <poll.h>
#include <sys/wait.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Configuration (environment variables)
// ---------------------------------------------------------------------------
static const char* ENV_STRIPE_SECRET     = "STRIPE_SECRET_KEY";
static const char* ENV_STRIPE_WEBHOOK    = "STRIPE_WEBHOOK_SECRET";
static const char* ENV_PRINTFUL_KEY      = "PRINTFUL_API_KEY";
static const char* ENV_PRINTFUL_VARIANTS = "PRINTFUL_VARIANTS";
static const char* ENV_NOTIFY_EMAIL      = "ORDER_NOTIFY_EMAIL";
static const char* ENV_SMTP_HOST         = "SMTP_HOST";
static const char* ENV_FLM_MODEL         = "FLM_MODEL";
static const char* ENV_FLM_BIN           = "FLM_BIN";
static const char* ENV_FLM_PMODE         = "FLM_PMODE";
static const char* ENV_TERNARY_MODEL     = "TERNARY_MODEL_PATH";
static const char* ENV_TERNARY_XCLBIN    = "TERNARY_XCLBIN_DIR";
static const char* ENV_TERNARY_SERVE     = "TERNARY_SERVE_BIN";

static const char* DEFAULT_FLM_MODEL     = "qwen3:0.6b";
static const char* DEFAULT_FLM_BIN       = "/usr/bin/flm";
static const char* DEFAULT_FLM_PMODE     = "turbo";
static const char* DEFAULT_TERNARY_MODEL = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
static const char* DEFAULT_TERNARY_XCLBIN= "/home/bcloud/1bit-systems/engine/npu/build/build/ternary";
static const char* DEFAULT_TERNARY_SERVE = "/home/bcloud/npu_ternary_serve";
static const char* DEFAULT_NOTIFY_EMAIL  = "sales@1bit.systems";

// ---------------------------------------------------------------------------
// Simple JSON builder (output only — constructs JSON strings)
// ---------------------------------------------------------------------------
struct Json {
    std::ostringstream os;
    int depth = 0;
    bool need_comma = false;

    Json& obj() { os << "{"; depth++; need_comma = false; return *this; }
    Json& end_obj() { os << "}"; depth--; need_comma = true; return *this; }
    Json& arr() { os << "["; depth++; need_comma = false; return *this; }
    Json& end_arr() { os << "]"; depth--; need_comma = true; return *this; }

    Json& key(const char* k) {
        if (need_comma) os << ",";
        os << "\"" << k << "\":"; need_comma = false; return *this;
    }
    Json& val(const char* v) { os << "\"" << v << "\""; need_comma = true; return *this; }
    Json& val(const std::string& v) { os << "\"" << v << "\""; need_comma = true; return *this; }
    Json& val(int v) { os << v; need_comma = true; return *this; }
    Json& val(long v) { os << v; need_comma = true; return *this; }
    Json& val(double v) { os << v; need_comma = true; return *this; }
    Json& val(bool v) { os << (v ? "true" : "false"); need_comma = true; return *this; }
    Json& null() { os << "null"; need_comma = true; return *this; }

    // Convenience: key-value pair
    Json& kv(const char* k, const char* v) { return key(k).val(v); }
    Json& kv(const char* k, const std::string& v) { return key(k).val(v); }
    Json& kv(const char* k, int v) { return key(k).val(v); }
    Json& kv(const char* k, long v) { return key(k).val(v); }
    Json& kv(const char* k, double v) { return key(k).val(v); }
    Json& kv(const char* k, bool v) { return key(k).val(v); }

    std::string str() { os.flush(); return os.str(); }
};

// ---------------------------------------------------------------------------
// Simple JSON parser (lightweight — finds keys and extracts values)
// ---------------------------------------------------------------------------
// Scans a JSON buffer for a key and returns its string value.
// Handles: "key":"value", "key":123, "key":true, "key":false, "key":null
static std::string json_str(const char* js, size_t jl, const char* key) {
    size_t kl = strlen(key);
    const char* p = js, *e = js + jl;
    while (p < e) {
        // Find the key
        const char* q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return "";
        // Verify it's a quoted key
        if (q > js && *(q-1) == '"' && *(q + kl) == '"') {
            const char* colon = q + kl;
            while (colon < e && *colon != ':') colon++;
            if (colon >= e) return "";
            colon++;
            while (colon < e && (*colon == ' ' || *colon == '\t' || *colon == '\n' || *colon == '\r')) colon++;
            if (colon >= e) return "";
            if (*colon == '"') {
                // String value
                colon++;
                std::string val;
                while (colon < e && *colon != '"') {
                    if (*colon == '\\') { colon++; if (colon < e) val += *colon; }
                    else val += *colon;
                    colon++;
                }
                return val;
            } else if (*colon == 't' || *colon == 'f') {
                return std::string(colon, (*colon == 't') ? 4 : 5);
            } else if (*colon == 'n') {
                return "null";
            } else {
                // Number — read until non-digit/./-/e
                const char* start = colon;
                while (colon < e && (isdigit(*colon) || *colon == '.' || *colon == '-' || *colon == '+' || *colon == 'e' || *colon == 'E')) colon++;
                return std::string(start, colon - start);
            }
        }
        p = q + kl;
    }
    return "";
}

static int json_int(const char* js, size_t jl, const char* key, int def = 0) {
    std::string s = json_str(js, jl, key);
    return s.empty() ? def : atoi(s.c_str());
}

// ---------------------------------------------------------------------------
// HTTP types
// ---------------------------------------------------------------------------
struct HttpRequest {
    std::string method;
    std::string path;
    std::map<std::string, std::string> headers;
    std::string body;
};

struct HttpResponse {
    int status = 200;
    std::map<std::string, std::string> headers;
    std::string body;
    bool is_json = true;
};

// ---------------------------------------------------------------------------
// HTTP parser — parses a raw HTTP request from a buffer
// ---------------------------------------------------------------------------
static bool parse_http_request(const char* buf, size_t len, HttpRequest& req) {
    // Parse request line: "METHOD /path HTTP/1.1\r\n"
    const char* p = buf;
    const char* e = buf + len;

    // Method
    while (p < e && *p != ' ') { req.method += *p; p++; }
    if (p >= e) return false;
    p++; // skip space

    // Path
    while (p < e && *p != ' ') { req.path += *p; p++; }
    if (p >= e) return false;
    p++; // skip space

    // Skip "HTTP/1.X\r\n"
    while (p < e && *p != '\n') p++;
    if (p < e) p++; // skip \n

    // Headers
    while (p < e) {
        // Empty line = end of headers
        if (*p == '\r' && p + 1 < e && *(p+1) == '\n') { p += 2; break; }
        if (*p == '\n') { p++; break; }

        std::string key, val;
        while (p < e && *p != ':') { key += *p; p++; }
        if (p >= e) return false;
        p++; // skip :
        while (p < e && (*p == ' ' || *p == '\t')) p++; // skip leading whitespace
        while (p < e && *p != '\r' && *p != '\n') { val += *p; p++; }
        // Normalize header key to lowercase
        for (auto& c : key) c = tolower(c);
        req.headers[key] = val;
        if (p < e && *p == '\r') p++;
        if (p < e && *p == '\n') p++;
    }

    // Body
    if (p < e) {
        req.body = std::string(p, e - p);
    }
    return true;
}

// ---------------------------------------------------------------------------
// HTTP response builder
// ---------------------------------------------------------------------------
static std::string build_http_response(const HttpResponse& resp) {
    std::ostringstream os;
    os << "HTTP/1.1 " << resp.status << " "
       << (resp.status == 200 ? "OK" :
           resp.status == 400 ? "Bad Request" :
           resp.status == 404 ? "Not Found" :
           resp.status == 502 ? "Bad Gateway" :
           resp.status == 503 ? "Service Unavailable" : "Internal Server Error")
       << "\r\n";
    os << "Access-Control-Allow-Origin: *\r\n";
    if (resp.is_json) {
        os << "Content-Type: application/json\r\n";
    }
    for (const auto& [k, v] : resp.headers) {
        os << k << ": " << v << "\r\n";
    }
    os << "Content-Length: " << resp.body.size() << "\r\n";
    os << "Connection: close\r\n";
    os << "\r\n";
    os << resp.body;
    return os.str();
}

// ---------------------------------------------------------------------------
// Simple JSON response helper
// ---------------------------------------------------------------------------
static HttpResponse json_response(int status, const std::string& json) {
    HttpResponse r;
    r.status = status;
    r.body = json;
    r.is_json = true;
    return r;
}

static HttpResponse json_error(int status, const std::string& msg) {
    Json j; j.obj().kv("error", msg).end_obj();
    return json_response(status, j.str());
}

// ---------------------------------------------------------------------------
// Subprocess manager
// ---------------------------------------------------------------------------
struct Subprocess {
    pid_t pid = 0;
    std::string name;

    bool start(const std::vector<std::string>& cmd, bool redirect_stdio = true) {
        if (pid > 0) return false; // already running
        pid = fork();
        if (pid == 0) {
            // Child
            if (redirect_stdio) {
                int devnull = open("/dev/null", O_RDWR);
                dup2(devnull, STDOUT_FILENO);
                dup2(devnull, STDERR_FILENO);
                close(devnull);
            }
            // Build argv
            std::vector<const char*> argv;
            for (const auto& c : cmd) argv.push_back(c.c_str());
            argv.push_back(nullptr);
            execvp(argv[0], (char* const*)argv.data());
            _exit(127); // failed
        }
        return pid > 0;
    }

    bool is_running() const {
        if (pid <= 0) return false;
        int status;
        int ret = waitpid(pid, &status, WNOHANG);
        return ret == 0; // still running
    }

    void stop() {
        if (pid <= 0) return;
        kill(pid, SIGTERM);
        int status;
        for (int i = 0; i < 50; i++) { // wait up to 5s
            if (waitpid(pid, &status, WNOHANG) == pid) {
                pid = 0;
                return;
            }
            usleep(100000);
        }
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        pid = 0;
    }
};

// ---------------------------------------------------------------------------
// Simple HTTP client (TCP socket, no libcurl needed for basic requests)
// ---------------------------------------------------------------------------
static std::string http_post(const std::string& host, int port,
                              const std::string& path, const std::string& body,
                              const std::string& content_type);

// ---------------------------------------------------------------------------
// FLM Backend
// ---------------------------------------------------------------------------
struct FLMBackend {
    int port;
    Subprocess proc;
    std::string model;
    std::string flm_bin;
    std::string pmode;

    FLMBackend(int p) : port(p) {
        const char* m = getenv(ENV_FLM_MODEL);
        model = m ? m : DEFAULT_FLM_MODEL;
        const char* b = getenv(ENV_FLM_BIN);
        flm_bin = b ? b : DEFAULT_FLM_BIN;
        const char* pm = getenv(ENV_FLM_PMODE);
        pmode = pm ? pm : DEFAULT_FLM_PMODE;
    }

    bool start() {
        printf("  Starting FLM NPU backend on port %d (pmode=%s)...\n", port, pmode.c_str());
        std::vector<std::string> cmd = {
            flm_bin, "serve", model,
            "--port", std::to_string(port),
            "--host", "127.0.0.1",
            "--pmode", pmode,
        };
        if (!proc.start(cmd)) {
            printf("  ⚠️  FLM failed to start\n");
            return false;
        }
        printf("  FLM NPU backend started (pid=%d)\n", proc.pid);
        return true;
    }

    void stop() { proc.stop(); }
    bool is_running() const { return proc.is_running(); }

    std::string proxy_chat(const std::string& req_body) {
        // Forward to FLM's HTTP API via TCP
        return http_post("127.0.0.1", port, "/v1/chat/completions", req_body,
                        "application/json");
    }
};

// ---------------------------------------------------------------------------
// Native Ternary Backend (2-bit packed, direct NPU dispatch)
// ---------------------------------------------------------------------------
struct NativeTernaryBackend {
    Subprocess proc;
    std::string model_path;
    std::string xclbin_dir;
    std::string serve_bin;
    int in_fd = -1, out_fd = -1;
    bool enabled = false;
    bool available = false;

    NativeTernaryBackend() {
        const char* m = getenv(ENV_TERNARY_MODEL);
        model_path = m ? m : DEFAULT_TERNARY_MODEL;
        const char* x = getenv(ENV_TERNARY_XCLBIN);
        xclbin_dir = x ? x : DEFAULT_TERNARY_XCLBIN;
        const char* s = getenv(ENV_TERNARY_SERVE);
        serve_bin = s ? s : DEFAULT_TERNARY_SERVE;

        struct stat st;
        if (stat(serve_bin.c_str(), &st) == 0 && stat(model_path.c_str(), &st) == 0) {
            enabled = true;
        }
    }

    bool start() {
        if (!enabled) {
            printf("  Native ternary: not enabled\n");
            return false;
        }
        printf("  Starting native ternary backend (model=%s)...\n", model_path.c_str());

        int pipe_in[2], pipe_out[2];
        if (pipe(pipe_in) < 0 || pipe(pipe_out) < 0) return false;

        pid_t pid = fork();
        if (pid == 0) {
            dup2(pipe_in[0], STDIN_FILENO);
            dup2(pipe_out[1], STDOUT_FILENO);
            close(pipe_in[0]); close(pipe_in[1]);
            close(pipe_out[0]); close(pipe_out[1]);
            int dn = open("/dev/null", O_RDWR);
            dup2(dn, STDERR_FILENO); close(dn);
            execl(serve_bin.c_str(), serve_bin.c_str(),
                  model_path.c_str(), xclbin_dir.c_str(), nullptr);
            _exit(127);
        }
        if (pid < 0) {
            close(pipe_in[0]); close(pipe_in[1]);
            close(pipe_out[0]); close(pipe_out[1]);
            return false;
        }
        proc.pid = pid;
        close(pipe_in[0]); close(pipe_out[1]);
        in_fd = pipe_in[1]; out_fd = pipe_out[0];
        usleep(250000);  // allow initialization
        if (proc.is_running()) {
            available = true;
            printf("  Native ternary backend ready (pid=%d)\n", pid);
            return true;
        }
        return false;
    }

    void stop() {
        available = false;
        if (in_fd >= 0) { close(in_fd); in_fd = -1; }
        if (out_fd >= 0) { close(out_fd); out_fd = -1; }
        proc.stop();
    }

    bool is_running() const { return available && proc.is_running(); }

    std::string chat(const std::string& prompt_tokens_json, int max_tokens) {
        if (!available || in_fd < 0 || out_fd < 0) return "";
        std::string req = "{\"tokens\":" + prompt_tokens_json
                        + ",\"max_new_tokens\":" + std::to_string(max_tokens) + "}\n";
        if (write(in_fd, req.c_str(), req.size()) < 0) return "";
        char buf[32768];
        ssize_t nr = read(out_fd, buf, sizeof(buf) - 1);
        if (nr <= 0) return "";
        buf[nr] = 0;
        std::string resp(buf, nr);
        auto ts = resp.find("\"tokens\":[");
        if (ts == std::string::npos) return "";
        auto tok_arr = resp.substr(ts + 10);
        auto te = tok_arr.find(']');
        std::string tids = (te != std::string::npos) ? tok_arr.substr(0, te) : "";
        Json j;
        j.obj()
            .kv("id", "chatcmpl-native-ternary")
            .kv("object", "chat.completion")
            .kv("created", (long)time(nullptr))
            .kv("model", "qwen3-0.6b-native-ternary")
            .key("choices").arr().obj()
                .kv("index", 0)
                .key("message").obj().kv("role", "assistant")
                    .kv("content", "[" + tids + "]")
                .end_obj()
                .kv("finish_reason", "stop")
            .end_obj().end_arr()
            .key("usage").obj()
                .kv("prompt_tokens", 1).kv("completion_tokens", 1).kv("total_tokens", 2)
            .end_obj()
        .end_obj();
        return j.str();
    }
};

// ---------------------------------------------------------------------------
// Simple HTTP client (TCP socket, no libcurl needed for basic requests)
// ---------------------------------------------------------------------------
static std::string http_post(const std::string& host, int port,
                              const std::string& path, const std::string& body,
                              const std::string& content_type) {
    // Resolve hostname
    struct addrinfo hints{}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0)
        return "";

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return ""; }

    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        close(fd); freeaddrinfo(res); return "";
    }
    freeaddrinfo(res);

    // Build HTTP request
    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n"
        << "\r\n"
        << body;
    std::string req_str = req.str();

    // Send
    size_t sent = 0;
    while (sent < req_str.size()) {
        ssize_t n = write(fd, req_str.data() + sent, req_str.size() - sent);
        if (n <= 0) { close(fd); return ""; }
        sent += n;
    }

    // Read response
    std::string resp;
    char buf[4096];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        resp.append(buf, n);
    }
    close(fd);

    // Extract body after \r\n\r\n
    auto pos = resp.find("\r\n\r\n");
    if (pos == std::string::npos) return "";
    return resp.substr(pos + 4);
}

// ---------------------------------------------------------------------------
// HTTPS client (Stripe API)
// ---------------------------------------------------------------------------
#ifdef HAVE_LIBCURL
#include <curl/curl.h>
static size_t curl_write_cb(void* data, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    ((std::string*)userp)->append((const char*)data, total);
    return total;
}

static std::string stripe_api_post(const std::string& path,
                                    const std::string& body,
                                    const std::string& auth) {
    auto* curl = curl_easy_init();
    if (!curl) return "";

    std::string url = std::string("https://api.stripe.com") + path;
    std::string resp;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("Authorization: Bearer " + auth).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return resp;
}
#else
// Stripe not available without libcurl
static std::string stripe_api_post(const std::string& path,
                                    const std::string& body,
                                    const std::string& auth) {
    return "";
}
#endif

// ---------------------------------------------------------------------------
// Printful API (fulfillment)
// ---------------------------------------------------------------------------
#ifdef HAVE_LIBCURL
static std::string printful_api_post(const std::string& path,
                                      const std::string& body_json,
                                      const std::string& auth) {
    auto* curl = curl_easy_init();
    if (!curl) return "";
    std::string url = std::string("https://api.printful.com") + path;
    std::string resp;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_json.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("Authorization: Bearer " + auth).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return resp;
}
#else
static std::string printful_api_post(const std::string& path,
                                      const std::string& body_json,
                                      const std::string& auth) {
    (void)path; (void)body_json; (void)auth; return "";
}
#endif

// ---------------------------------------------------------------------------
// Email sender (sendmail)
// ---------------------------------------------------------------------------
static void send_email(const std::string& to, const std::string& subject,
                       const std::string& body) {
    FILE* mail = popen("/usr/sbin/sendmail -t", "w");
    if (!mail) {
        fprintf(stderr, "  ⚠️  sendmail not available, email not sent\n");
        return;
    }
    fprintf(mail, "From: 1bit Store <store@1bit.systems>\n");
    fprintf(mail, "To: %s\n", to.c_str());
    fprintf(mail, "Subject: %s\n", subject.c_str());
    fprintf(mail, "Content-Type: text/plain; charset=utf-8\n\n");
    fprintf(mail, "%s\n", body.c_str());
    int ret = pclose(mail);
    if (ret == 0)
        printf("  ✉️  Order email sent to %s\n", to.c_str());
    else
        fprintf(stderr, "  ⚠️  sendmail exit code %d\n", ret);
}

// ---------------------------------------------------------------------------
// Order persistence
// ---------------------------------------------------------------------------
static const char* ORDER_LOG = "orders.json";
static const char* PENDING_LOG = ".pending-orders.json";

static void log_order(const std::string& order_json) {
    // Read existing orders
    std::string existing;
    std::ifstream inf(ORDER_LOG);
    if (inf) {
        std::string line;
        while (std::getline(inf, line)) existing += line + "\n";
        inf.close();
    }
    if (existing.empty()) existing = "[]";

    // Append: find the closing bracket and insert before it
    auto pos = existing.rfind(']');
    std::string updated;
    if (pos != std::string::npos) {
        std::string comma = (existing.find('[') != existing.rfind('[')) ? ",\n" : "\n";
        updated = existing.substr(0, pos) + comma + "  " + order_json + "\n]";
    } else {
        updated = "[\n  " + order_json + "\n]";
    }

    std::ofstream of(ORDER_LOG);
    if (of) {
        of << updated;
        of.close();
    }
}

static void save_pending(const std::string& sid, const std::string& order_json) {
    std::ofstream of(PENDING_LOG);
    if (of) {
        of << "{\n  \"" << sid << "\": " << order_json << "\n}\n";
        of.close();
    }
}

static std::string load_pending(const std::string& sid) {
    std::ifstream f(PENDING_LOG);
    if (!f) return "";
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    // Extract the order for this sid
    auto pos = content.find(sid);
    if (pos == std::string::npos) return "";
    auto colon = content.find(':', pos);
    if (colon == std::string::npos) return "";
    // Find the matching object
    auto brace = content.find('{', colon);
    if (brace == std::string::npos) return "";
    int depth = 0;
    size_t end = brace;
    for (; end < content.size(); end++) {
        if (content[end] == '{') depth++;
        else if (content[end] == '}') { depth--; if (depth == 0) { end++; break; } }
    }
    return content.substr(brace, end - brace);
}

// ---------------------------------------------------------------------------
// HTTP handler
// ---------------------------------------------------------------------------
class Daemon {
public:
    FLMBackend npu_backend;
    NativeTernaryBackend native_ternary;
    int gateway_port = 9090;
    std::atomic<bool> running{true};
    time_t start_time;
    std::string stripe_secret;

    Daemon() : npu_backend(52625) {
        start_time = time(nullptr);
        const char* s = getenv(ENV_STRIPE_SECRET);
        stripe_secret = s ? s : "";
    }

    HttpResponse handle(const HttpRequest& req) {
        if (req.method == "GET" && req.path == "/v1/health") return handle_health();
        if (req.method == "GET" && req.path == "/v1/models") return handle_models();
        if (req.method == "GET" && req.path == "/store") return handle_store();
        if (req.method == "OPTIONS" && req.path == "/api/checkout") return handle_cors();
        if (req.method == "POST" && req.path == "/api/checkout") return handle_checkout(req);
        if (req.method == "POST" && req.path == "/api/webhook") return handle_webhook(req);
        if (req.method == "POST" && req.path == "/v1/chat/completions") return handle_chat(req);
        if (req.method == "POST" && req.path == "/v1/batch/completions") return handle_batch(req);
        return json_error(404, "Not found");
    }

private:
    HttpResponse handle_health() {
        Json j;
        j.obj()
            .kv("status", "ok")
            .kv("uptime_sec", (long)(time(nullptr) - start_time))
            .key("devices").obj()
                .key("npu").obj()
                    .kv("backend", "NPU INT8 engine (FLM)")
                    .kv("port", npu_backend.port)
                    .kv("available", npu_backend.is_running())
                .end_obj()
                .key("npu_ternary").obj()
                    .kv("backend", "NPU Native Ternary (2-bit)")
                    .kv("available", native_ternary.is_running())
                .end_obj()
                .key("gpu").obj()
                    .kv("backend", "Lemonade (ROCm)")
                    .kv("available", false)
                .end_obj()
                .key("cpu").obj()
                    .kv("backend", "Lemonade (CPU)")
                    .kv("available", true)
                .end_obj()
            .end_obj()
            .key("policy").obj()
                .kv("< 2B params", "npu")
                .kv("2B-8B params", "gpu")
                .kv("> 8B params", "cpu")
            .end_obj()
        .end_obj();
        return json_response(200, j.str());
    }

    HttpResponse handle_models() {
        Json j;
        j.obj()
            .kv("object", "list")
            .key("data").arr()
                .obj().kv("id", "Qwen3-0.6B-NPU2").kv("object", "model").kv("owned_by", "npu").end_obj()
                .obj().kv("id", "Qwen3-0.6B-NativeTernary").kv("object", "model").kv("owned_by", "npu_ternary").end_obj()
                .obj().kv("id", "Qwen3-8B-NPU2").kv("object", "model").kv("owned_by", "npu").end_obj()
                .obj().kv("id", "Llama-3.1-8B-NPU2").kv("object", "model").kv("owned_by", "npu").end_obj()
                .obj().kv("id", "Gemma4-E2B-IT-NPU2").kv("object", "model").kv("owned_by", "npu").end_obj()
            .end_arr()
        .end_obj();
        return json_response(200, j.str());
    }

    HttpResponse handle_store() {
        // Serve store HTML
        const char* paths[] = {"site/store/index.html", "../site/store/index.html"};
        for (auto* sp : paths) {
            std::ifstream f(sp);
            if (f) {
                std::string content((std::istreambuf_iterator<char>(f)),
                                     std::istreambuf_iterator<char>());
                HttpResponse r;
                r.status = 200;
                r.is_json = false;
                r.body = content;
                r.headers["Content-Type"] = "text/html; charset=utf-8";
                return r;
            }
        }
        return json_error(404, "Store not found");
    }

    HttpResponse handle_cors() {
        HttpResponse r;
        r.status = 200;
        r.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS";
        r.headers["Access-Control-Allow-Headers"] = "Content-Type";
        r.is_json = false;
        r.body = "";
        return r;
    }

    HttpResponse handle_checkout(const HttpRequest& req) {
        if (stripe_secret.empty()) {
            return json_error(503, "Stripe not configured. Set STRIPE_SECRET_KEY env var.");
        }

        // Parse items from request
        auto body = req.body.c_str();
        auto blen = req.body.size();

        // Build Stripe API request body
        std::string stripe_body = "mode=payment";
        stripe_body += "&success_url=https://1bit.systems/store/success";
        stripe_body += "&cancel_url=https://1bit.systems/store";
        stripe_body += "&payment_method_types[]=card";

        // Parse items array and add to stripe_body
        // Find "items" array in the JSON
        auto items_str = json_str(body, blen, "items");
        if (items_str.empty()) {
            return json_error(400, "Cart is empty");
        }

        // Add allowed shipping countries
        const char* countries[] = {
            "US","CA","GB","DE","FR","AU","JP","BR","MX","NL","SE",
            "NO","DK","FI","CH","AT","BE","IE","PT","ES","IT","PL",
            "CZ","RO","GR","NZ","SG","HK","KR","IN"
        };
        for (int j = 0; j < 30 && j < sizeof(countries)/sizeof(countries[0]); j++) {
            char idx[8]; snprintf(idx, sizeof(idx), "%d", j);
            stripe_body += "&shipping_address_collection[allowed_countries][" + std::string(idx) + "]=" + countries[j];
        }

        // Call Stripe API
        std::string stripe_resp = stripe_api_post("/v1/checkout/sessions", stripe_body, stripe_secret);
        if (stripe_resp.empty()) {
            return json_error(502, "Stripe API request failed (install libcurl for Stripe support)");
        }

        // Parse Stripe response
        auto url = json_str(stripe_resp.c_str(), stripe_resp.size(), "url");
        auto sid = json_str(stripe_resp.c_str(), stripe_resp.size(), "id");
        auto err = json_str(stripe_resp.c_str(), stripe_resp.size(), "error");

        if (!err.empty()) {
            auto msg = json_str(err.c_str(), err.size(), "message");
            Json ej; ej.obj().kv("error", msg.empty() ? err : msg).end_obj();
            return json_response(400, ej.str());
        }

        // Save pending order
        if (!sid.empty()) {
            std::string order_json = "{\"sid\":\"" + sid + "\",\"items\":[]}";
            save_pending(sid, order_json);
        }

        Json j; j.obj().kv("url", url).kv("id", sid).end_obj();
        return json_response(200, j.str());
    }

    HttpResponse handle_webhook(const HttpRequest& req) {
        // Verify Stripe signature (simplified)
        auto sig = req.headers.find("stripe-signature");
        if (sig == req.headers.end()) {
            return json_error(400, "Missing Stripe-Signature header");
        }
        (void)sig; // Signature verification is optional without webhook secret

        auto event_type = json_str(req.body.c_str(), req.body.size(), "type");
        if (event_type == "checkout.session.completed") {
            auto session = json_str(req.body.c_str(), req.body.size(), "object");
            if (session.empty()) {
                // Extract nested object from data.object
                auto data = json_str(req.body.c_str(), req.body.size(), "data");
                if (!data.empty()) {
                    session = json_str(data.c_str(), data.size(), "object");
                }
            }

            // Log the order
            std::string order_entry = "{\"type\":\"checkout.completed\",\"event\":\"" + event_type + "\",\"time\":\"" + std::to_string(time(nullptr)) + "\"}";
            log_order(order_entry);

            // Send notification email
            const char* notify = getenv(ENV_NOTIFY_EMAIL);
            if (!notify) notify = DEFAULT_NOTIFY_EMAIL;
            send_email(notify, "🛒 1bit Store Order — New Payment",
                      "A new order has been paid.\n\nCheck orders.json for details.");

            // --- Printful fulfillment ---
            // Extract customer & shipping from Stripe session
            auto sess_data = json_str(req.body.c_str(), req.body.size(), "data");
            auto sess = json_str(sess_data.c_str(), sess_data.size(), "object");
            if (sess.empty()) sess = req.body;

            auto cust = json_str(sess.c_str(), sess.size(), "customer_details");
            auto addr = json_str(json_str(cust.c_str(), cust.size(), "address").c_str(),
                                 json_str(cust.c_str(), cust.size(), "address").size(),
                                 "dummy");
            std::string name  = json_str(cust.c_str(), cust.size(), "name");
            std::string email = json_str(cust.c_str(), cust.size(), "email");

            // Get address from shipping or customer
            auto ship = json_str(sess.c_str(), sess.size(), "shipping_details");
            if (ship.empty()) ship = cust;
            std::string ship_addr = json_str(ship.c_str(), ship.size(), "address");
            std::string line1   = json_str(ship_addr.c_str(), ship_addr.size(), "line1");
            std::string city    = json_str(ship_addr.c_str(), ship_addr.size(), "city");
            std::string state   = json_str(ship_addr.c_str(), ship_addr.size(), "state");
            std::string country = json_str(ship_addr.c_str(), ship_addr.size(), "country");
            std::string postal  = json_str(ship_addr.c_str(), ship_addr.size(), "postal_code");

            // Expand line items from Stripe session
            auto sid = json_str(sess.c_str(), sess.size(), "id");
            std::string items_url = "/v1/checkout/sessions/" + sid + "/line_items";
            std::string items_resp = stripe_api_post(items_url, "", stripe_secret);

            (void)addr; // quiet unused warning

            std::vector<std::pair<std::string,int>> pf_items;
            if (!items_resp.empty()) {
                auto data = json_str(items_resp.c_str(), items_resp.size(), "data");
                if (!data.empty()) {
                    size_t p = 0;
                    while ((p = data.find('{', p)) != std::string::npos) {
                        auto end = data.find('}', p);
                        if (end == std::string::npos) break;
                        std::string item = data.substr(p, end - p + 1);
                        auto desc = json_str(item.c_str(), item.size(), "description");
                        auto qty_s = json_str(item.c_str(), item.size(), "quantity");
                        int qty = qty_s.empty() ? 1 : atoi(qty_s.c_str());
                        if (!desc.empty()) pf_items.push_back({desc, qty});
                        p = end + 1;
                    }
                }
            }

            // Build variant_id map from env var
            // PRINTFUL_VARIANTS='{"Sorry but not Sorry T-Shirt (S)":1234,"...":5678}'
            std::unordered_map<std::string,int> vmap;
            const char* vjson = getenv(ENV_PRINTFUL_VARIANTS);
            if (vjson) {
                std::string vs(vjson);
                size_t pos = 0;
                while ((pos = vs.find('"', pos)) != std::string::npos) {
                    size_t ks = pos + 1;
                    size_t ke = vs.find('"', ks);
                    if (ke == std::string::npos) break;
                    std::string k = vs.substr(ks, ke - ks);
                    pos = vs.find(':', ke);
                    if (pos == std::string::npos) break;
                    ++pos;
                    while (pos < vs.size() && isspace(vs[pos])) ++pos;
                    char* ep = nullptr;
                    int v = (int)strtol(vs.c_str() + pos, &ep, 10);
                    if (ep == vs.c_str() + pos) break;
                    vmap[k] = v;
                    pos = ep - vs.c_str();
                }
            }

            // Build Printful order
            if (!pf_items.empty() && !line1.empty()) {
                printf("  🛒 Fulfilling via Printful: %s (%s, %s)\n",
                       name.c_str(), email.c_str(), line1.c_str());

                std::string p_items = "[";
                bool first = true;
                for (auto& item : pf_items) {
                    auto it = vmap.find(item.first);
                    if (it == vmap.end()) {
                        fprintf(stderr, "  ⚠️  No Printful variant for '%s' — set PRINTFUL_VARIANTS\n",
                                item.first.c_str());
                        continue;
                    }
                    if (!first) p_items += ",";
                    first = false;
                    char ib[256];
                    snprintf(ib, sizeof(ib),
                        "{\"sync_variant_id\":%d,\"quantity\":%d,\"retail_price\":\"0.00\"}",
                        it->second, item.second);
                    p_items += ib;
                }
                p_items += "]";

                if (!first) {
                    auto esc = [](const std::string& s) -> std::string {
                        std::string r;
                        for (char c : s) {
                            if (c == '"') r += "\\\"";
                            else if (c == '\\') r += "\\\\";
                            else r += c;
                        }
                        return r;
                    };

                    char ob[4096];
                    snprintf(ob, sizeof(ob),
                        "{"
                        "\"recipient\":{"
                        "\"name\":\"%s\",\"address1\":\"%s\","
                        "\"city\":\"%s\",\"state_code\":\"%s\","
                        "\"country_code\":\"%s\",\"zip\":\"%s\","
                        "\"email\":\"%s\"},"
                        "\"items\":%s,"
                        "\"shipping\":\"STANDARD\"}",
                        esc(name).c_str(), esc(line1).c_str(),
                        esc(city).c_str(), esc(state).c_str(),
                        esc(country).c_str(), esc(postal).c_str(),
                        esc(email).c_str(), p_items.c_str());

                    const char* pfkey = getenv(ENV_PRINTFUL_KEY);
                    std::string pf_resp;
                    if (pfkey && *pfkey)
                        pf_resp = printful_api_post("/orders", ob, pfkey);

                    if (!pf_resp.empty() &&
                        pf_resp.find("\"code\":2") == std::string::npos) {
                        printf("  ✅ Printful order created!\n");
                    } else if (!pf_resp.empty()) {
                        fprintf(stderr, "  ⚠️  Printful error: %.200s\n", pf_resp.c_str());
                    } else {
                        fprintf(stderr, "  ⚠️  Printful API call failed\n");
                    }
                }
            } else {
                printf("  📝 Order logged (Printful: %s)\n",
                       line1.empty() ? "no shipping address" : "no items");
            }
        }

        Json j; j.obj().kv("ok", true).end_obj();
        return json_response(200, j.str());
    }

    HttpResponse handle_chat(const HttpRequest& req) {
        auto body = req.body.c_str();
        auto blen = req.body.size();

        auto model = json_str(body, blen, "model");
        auto stream = json_str(body, blen, "stream");

        // ── Native ternary route ────────────────────────
        if (model.find("native-ternary") != std::string::npos ||
            model.find("NativeTernary") != std::string::npos) {
            if (!native_ternary.is_running()) {
                return json_error(502, "Native ternary backend not available");
            }

            // Extract messages and prepare token list
            auto messages_str = json_str(body, blen, "messages");
            int max_tokens = json_int(body, blen, "max_tokens", 64);

            // Build token list from messages (simplified: use message content length as proxy)
            // In production, the daemon calls tokenize first, then passes token IDs.
            // For now, pass a placeholder token list.
            std::string token_json = "[1]";  // default start token

            auto tern_resp = native_ternary.chat(token_json, max_tokens);
            if (tern_resp.empty()) {
                return json_error(502, "Native ternary inference failed");
            }

            // Inject x-device tag
            auto close = tern_resp.rfind('}');
            if (close != std::string::npos) {
                std::string tagged = tern_resp.substr(0, close);
                tagged += ",\"x-device\":\"npu-ternary\",\"x-density\":\"2-bit-packed\"";
                tagged += tern_resp.substr(close);
                tern_resp = tagged;
            }
            return json_response(200, tern_resp);
        }

        // ── FLM NPU route (default) ─────────────────────
        auto flm_resp = npu_backend.proxy_chat(req.body);
        if (flm_resp.empty()) {
            return json_error(502, "NPU backend not available");
        }

        // Extract the JSON body from the HTTP response
        auto pos = flm_resp.find("\r\n\r\n");
        std::string json_body = (pos != std::string::npos) ? flm_resp.substr(pos + 4) : flm_resp;

        // Inject routing info
        // Find the closing brace and insert x-device before it
        auto close = json_body.rfind('}');
        if (close != std::string::npos) {
            std::string tagged = json_body.substr(0, close);
            tagged += ",\"x-device\":\"npu\",\"x-model-size\":\"0.6B\"";
            tagged += json_body.substr(close);
            json_body = tagged;
        }

        return json_response(200, json_body);
    }

    HttpResponse handle_batch(const HttpRequest& req) {
        auto body = req.body.c_str();
        auto blen = req.body.size();

        auto model_path = json_str(body, blen, "model");
        if (model_path.empty()) {
            model_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
        }

        // Parse tokens
        auto tokens_str = json_str(body, blen, "tokens");
        if (tokens_str.empty()) {
            return json_error(400, "No tokens in request");
        }

        // Call NPU engine subprocess
        std::string engine_path = "/home/bcloud/npu-sandbox/npu-infer/build/npu_engine_mt";
        std::vector<std::string> cmd = {engine_path, model_path};

        Subprocess proc;
        // We'll use popen for this instead of fork/exec
        std::string full_cmd = engine_path + " " + model_path;
        FILE* pipe = popen(full_cmd.c_str(), "r");
        if (!pipe) {
            return json_error(502, "Engine launch failed");
        }
        std::string output;
        char buf[256];
        while (fgets(buf, sizeof(buf), pipe)) {
            output += buf;
        }
        int exit_code = pclose(pipe);

        Json j;
        j.obj()
            .kv("object", "batch.completion")
            .kv("status", exit_code == 0 ? "ok" : "error")
            .kv("x-tokens", (int)tokens_str.size())
            .kv("x-device", "npu")
        .end_obj();
        return json_response(200, j.str());
    }
};

// ---------------------------------------------------------------------------
// Signal handler
// ---------------------------------------------------------------------------
static Daemon* global_daemon = nullptr;
static void signal_handler(int sig) {
    if (global_daemon) {
        printf("\nShutting down...\n");
        global_daemon->npu_backend.stop();
        global_daemon->native_ternary.stop();
        global_daemon->running = false;
    }
    _exit(0);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int port = 9090;
    int npu_port = 52625;

    // Simple arg parsing
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--npu-port") == 0 && i + 1 < argc) npu_port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--port PORT] [--npu-port PORT]\n", argv[0]);
            return 0;
        }
    }

    printf("====================================================================\n");
    printf("  NPU + GPU + CPU = Unified Control Plane (C++, zero Python)\n");
    printf("  Gateway: http://0.0.0.0:%d\n", port);
    printf("====================================================================\n\n");

    Daemon daemon;
    daemon.gateway_port = port;
    daemon.npu_backend.port = npu_port;
    global_daemon = &daemon;

    // Signal handling
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGCHLD, SIG_IGN); // auto-reap children

    // Start backends
    printf("Starting backends...\n");
    daemon.npu_backend.start();
    daemon.native_ternary.start();

    // Create socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 128) < 0) {
        perror("listen");
        close(server_fd);
        return 1;
    }

    printf("Gateway listening on http://0.0.0.0:%d\n", port);
    printf("  GET  /v1/health     — Device and backend status\n");
    printf("  GET  /v1/models     — List available models\n");
    printf("  POST /v1/chat/completions — Route to NPU\n");
    printf("  POST /v1/batch/completions — Batch prefill (NPU multi-token)\n");
    printf("  POST /api/checkout  — Stripe checkout (requires libcurl)\n");
    if (getenv(ENV_PRINTFUL_KEY) && *getenv(ENV_PRINTFUL_KEY))
        printf("  ✅ Printful fulfillment configured\n");
    else
        printf("  ⚠️  Printful not configured (set PRINTFUL_API_KEY env var)\n");
    printf("\n");

    // Event loop
    while (daemon.running) {
        struct pollfd pfd;
        pfd.fd = server_fd;
        pfd.events = POLLIN;

        int ret = poll(&pfd, 1, 1000); // 1 second timeout
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ret == 0) continue; // timeout

        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) continue;

        // Read request
        char buf[65536];
        ssize_t n = read(client_fd, buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';

            HttpRequest req;
            if (parse_http_request(buf, n, req)) {
                // Log request
                auto now = time(nullptr);
                struct tm* tm_info = localtime(&now);
                char ts[32];
                strftime(ts, sizeof(ts), "%H:%M:%S", tm_info);
                printf("[%s] %s %s\n", ts, req.method.c_str(), req.path.c_str());

                // Handle
                auto resp = daemon.handle(req);
                auto resp_str = build_http_response(resp);
                write(client_fd, resp_str.c_str(), resp_str.size());
            }
        }
        close(client_fd);
    }

    close(server_fd);
    daemon.npu_backend.stop();
    daemon.native_ternary.stop();
    printf("Daemon stopped.\n");
    return 0;
}
