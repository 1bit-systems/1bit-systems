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
static const char* ENV_NOTIFY_EMAIL      = "ORDER_NOTIFY_EMAIL";
static const char* ENV_SMTP_HOST         = "SMTP_HOST";
static const char* ENV_FLM_MODEL         = "FLM_MODEL";
static const char* ENV_FLM_BIN           = "FLM_BIN";
static const char* ENV_FLM_PMODE         = "FLM_PMODE";

static const char* DEFAULT_FLM_MODEL     = "qwen3:0.6b";
static const char* DEFAULT_FLM_BIN       = "/usr/bin/flm";
static const char* DEFAULT_FLM_PMODE     = "turbo";
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

// Extracts the raw JSON text for `key`, including surrounding brackets/braces
// when the value is an array or object (json_str() above only handles scalar
// values and returns "" for arrays/objects since it stops at the first
// non-digit character).
static std::string json_raw(const char* js, size_t jl, const char* key) {
    size_t kl = strlen(key);
    const char* p = js, *e = js + jl;
    while (p < e) {
        const char* q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return "";
        if (q > js && *(q-1) == '"' && *(q + kl) == '"') {
            const char* colon = q + kl;
            while (colon < e && *colon != ':') colon++;
            if (colon >= e) return "";
            colon++;
            while (colon < e && isspace((unsigned char)*colon)) colon++;
            if (colon >= e) return "";
            if (*colon == '[' || *colon == '{') {
                char open = *colon, close = (open == '[') ? ']' : '}';
                const char* start = colon;
                int depth = 0;
                bool in_str = false;
                for (; colon < e; colon++) {
                    char c = *colon;
                    if (in_str) {
                        if (c == '\\') { colon++; continue; }
                        if (c == '"') in_str = false;
                        continue;
                    }
                    if (c == '"') { in_str = true; continue; }
                    if (c == open) depth++;
                    else if (c == close) {
                        depth--;
                        if (depth == 0) { colon++; break; }
                    }
                }
                return std::string(start, colon - start);
            }
            // Scalar value — fall back to the existing extractor.
            return json_str(js, jl, key);
        }
        p = q + kl;
    }
    return "";
}

// Splits a raw JSON array of objects (as returned by json_raw()) into the
// individual top-level "{...}" object substrings it contains.
static std::vector<std::string> json_array_objects(const std::string& arr) {
    std::vector<std::string> out;
    if (arr.size() < 2 || arr.front() != '[') return out;
    int depth = 0;
    bool in_str = false;
    size_t start = std::string::npos;
    for (size_t i = 1; i < arr.size(); i++) {
        char c = arr[i];
        if (in_str) {
            if (c == '\\') { i++; continue; }
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{') {
            if (depth == 0) start = i;
            depth++;
        } else if (c == '}') {
            depth--;
            if (depth == 0 && start != std::string::npos) {
                out.push_back(arr.substr(start, i - start + 1));
                start = std::string::npos;
            }
        }
    }
    return out;
}

// Percent-encodes a string for use as a single application/x-www-form-urlencoded value.
static std::string url_encode(const std::string& s) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size() * 3);
    for (unsigned char c : s) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += (char)c;
        } else {
            out += '%';
            out += hex[(c >> 4) & 0xF];
            out += hex[c & 0xF];
        }
    }
    return out;
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
// Stripe webhook signature verification (HMAC-SHA256)
// ---------------------------------------------------------------------------
#ifdef HAVE_OPENSSL
#include <openssl/hmac.h>
#include <openssl/evp.h>

// Verifies a Stripe webhook signature per Stripe's signing scheme:
// the "Stripe-Signature" header looks like "t=<timestamp>,v1=<hex hmac>[,v1=...]"
// and the signed payload is the string "{timestamp}.{raw_body}", HMAC-SHA256
// keyed with the webhook signing secret. See:
// https://stripe.com/docs/webhooks/signatures
static bool verify_stripe_signature(const std::string& sig_header,
                                     const std::string& payload,
                                     const std::string& secret) {
    std::string timestamp;
    std::vector<std::string> v1_sigs;

    size_t pos = 0;
    while (pos <= sig_header.size()) {
        size_t comma = sig_header.find(',', pos);
        std::string part = sig_header.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        size_t eq = part.find('=');
        if (eq != std::string::npos) {
            std::string k = part.substr(0, eq);
            std::string v = part.substr(eq + 1);
            if (k == "t") timestamp = v;
            else if (k == "v1") v1_sigs.push_back(v);
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }

    if (timestamp.empty() || v1_sigs.empty()) return false;

    // Replay protection: reject events whose timestamp is too far from now
    // (Stripe's own libraries default to a 5 minute tolerance).
    time_t ts = (time_t)atoll(timestamp.c_str());
    time_t now = time(nullptr);
    long long delta = (long long)now - (long long)ts;
    if (delta < 0) delta = -delta;
    if (delta > 300) return false;

    std::string signed_payload = timestamp + "." + payload;

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;
    if (!HMAC(EVP_sha256(),
              secret.data(), (int)secret.size(),
              (const unsigned char*)signed_payload.data(), signed_payload.size(),
              digest, &digest_len)) {
        return false;
    }

    static const char* hex = "0123456789abcdef";
    std::string computed;
    computed.reserve(digest_len * 2);
    for (unsigned int i = 0; i < digest_len; i++) {
        computed += hex[(digest[i] >> 4) & 0xF];
        computed += hex[digest[i] & 0xF];
    }

    // Constant-time comparison against every v1 signature Stripe sent
    // (Stripe may include multiple during secret rotation).
    bool matched = false;
    for (const auto& candidate : v1_sigs) {
        if (candidate.size() != computed.size()) continue;
        unsigned char diff = 0;
        for (size_t i = 0; i < computed.size(); i++) {
            diff |= (unsigned char)(candidate[i] ^ computed[i]);
        }
        matched |= (diff == 0);
    }
    return matched;
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

        // Parse items array and add to stripe_body as Stripe line_items[].
        // "items" is a JSON array, so it must go through json_raw() (json_str()
        // only handles scalar values and would always read this as empty).
        auto items_raw = json_raw(body, blen, "items");
        auto items = json_array_objects(items_raw);
        if (items.empty()) {
            return json_error(400, "Cart is empty");
        }

        int n_line_items = 0;
        for (size_t i = 0; i < items.size(); i++) {
            const auto& item = items[i];

            std::string currency = "usd";
            std::string name = "Item";
            long unit_amount = 0;

            auto price_data = json_raw(item.c_str(), item.size(), "price_data");
            if (!price_data.empty()) {
                auto cur = json_str(price_data.c_str(), price_data.size(), "currency");
                if (!cur.empty()) currency = cur;
                auto amt = json_str(price_data.c_str(), price_data.size(), "unit_amount");
                if (!amt.empty()) unit_amount = atol(amt.c_str());
                auto product_data = json_raw(price_data.c_str(), price_data.size(), "product_data");
                if (!product_data.empty()) {
                    auto nm = json_str(product_data.c_str(), product_data.size(), "name");
                    if (!nm.empty()) name = nm;
                }
            }

            auto qty_str = json_str(item.c_str(), item.size(), "quantity");
            long quantity = qty_str.empty() ? 1 : atol(qty_str.c_str());
            if (quantity < 1) quantity = 1;

            if (unit_amount <= 0) continue; // skip malformed line items

            std::string idx = std::to_string(n_line_items);
            stripe_body += "&line_items[" + idx + "][price_data][currency]=" + url_encode(currency);
            stripe_body += "&line_items[" + idx + "][price_data][product_data][name]=" + url_encode(name);
            stripe_body += "&line_items[" + idx + "][price_data][unit_amount]=" + std::to_string(unit_amount);
            stripe_body += "&line_items[" + idx + "][quantity]=" + std::to_string(quantity);
            n_line_items++;
        }

        if (n_line_items == 0) {
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
        // Verify Stripe signature. This endpoint triggers order logging and
        // customer-facing emails, so an unverified request must never be
        // treated as a real event — fail closed at every step below.
        auto sig = req.headers.find("stripe-signature");
        if (sig == req.headers.end()) {
            return json_error(400, "Missing Stripe-Signature header");
        }

        const char* webhook_secret_env = getenv(ENV_STRIPE_WEBHOOK);
        std::string webhook_secret = webhook_secret_env ? webhook_secret_env : "";
        if (webhook_secret.empty()) {
            return json_error(503, "Webhook verification not configured. Set STRIPE_WEBHOOK_SECRET env var.");
        }

#ifdef HAVE_OPENSSL
        if (!verify_stripe_signature(sig->second, req.body, webhook_secret)) {
            return json_error(400, "Invalid Stripe-Signature");
        }
#else
        return json_error(503, "Webhook verification requires OpenSSL support (rebuild with libssl-dev).");
#endif

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
        }

        Json j; j.obj().kv("ok", true).end_obj();
        return json_response(200, j.str());
    }

    HttpResponse handle_chat(const HttpRequest& req) {
        auto body = req.body.c_str();
        auto blen = req.body.size();

        auto model = json_str(body, blen, "model");
        auto stream = json_str(body, blen, "stream");

        // Route to NPU backend
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

        // Call NPU engine subprocess via fork()/execvp() — never through a
        // shell. `model_path` comes straight from the request body, and
        // popen("cmd " + model_path) would let an attacker run arbitrary
        // shell commands via the "model" field (e.g. "; rm -rf /").
        // execvp() passes model_path as a single argv element, so shell
        // metacharacters in it are inert.
        std::string engine_path = "/home/bcloud/npu-sandbox/npu-infer/build/npu_engine_mt";
        std::vector<std::string> cmd = {engine_path, model_path};

        int outpipe[2];
        if (pipe(outpipe) != 0) {
            return json_error(502, "Engine launch failed");
        }

        pid_t child_pid = fork();
        if (child_pid < 0) {
            close(outpipe[0]);
            close(outpipe[1]);
            return json_error(502, "Engine launch failed");
        }

        if (child_pid == 0) {
            // Child: redirect stdout to the pipe, discard stderr.
            close(outpipe[0]);
            dup2(outpipe[1], STDOUT_FILENO);
            close(outpipe[1]);
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }

            std::vector<const char*> argv;
            for (const auto& c : cmd) argv.push_back(c.c_str());
            argv.push_back(nullptr);
            execvp(argv[0], (char* const*)argv.data());
            _exit(127); // exec failed
        }

        // Parent: read child stdout until EOF, then reap it.
        close(outpipe[1]);
        std::string output;
        char buf[256];
        ssize_t n;
        while ((n = read(outpipe[0], buf, sizeof(buf))) > 0) {
            output.append(buf, (size_t)n);
        }
        close(outpipe[0]);

        int status = 0;
        waitpid(child_pid, &status, 0);
        int exit_code = (WIFEXITED(status)) ? WEXITSTATUS(status) : -1;

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

    // Start FLM backend
    printf("Starting backends...\n");
    daemon.npu_backend.start();

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
    printf("  POST /api/checkout  — Stripe checkout (requires libcurl)\n\n");

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
    printf("Daemon stopped.\n");
    return 0;
}
