/** spec_proxy.cpp — Speculative decoding proxy for FLM daemon
 *  Draft model runs on CPU. Target verification uses the FLM HTTP API.
 *  Exposes OpenAI-compatible chat endpoint on port 9091.
 *
 *  Build:
 *    g++ -std=c++17 -O3 -o spec_proxy spec_proxy.cpp \
 *      ../draft/mtp_draft.h -lpthread -lcurl
 *
 *  Run:
 *    ./spec_proxy [--flm-port 52625] [--port 9091]
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <curl/curl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include "../draft/mtp_draft.h"

// Config
static constexpr int VOCAB = 151936;
static constexpr int HIDDEN = 1024;
static constexpr int BLOCK_SIZE = 5;

// FLM API helpers
static size_t write_cb(void* contents, size_t sz, size_t nmemb, void* userp) {
    size_t total = sz * nmemb;
    ((std::string*)userp)->append((char*)contents, total);
    return total;
}

// Call FLM daemon for completion. Returns JSON response.
static std::string flm_complete(const std::string& prompt, int max_tokens,
                                const std::string& flm_host, int flm_port) {
    CURL* curl = curl_easy_init();
    if (!curl) return "";
    char url[128];
    snprintf(url, sizeof(url), "http://%s:%d/v1/chat/completions", flm_host.c_str(), flm_port);
    
    char body[4096];
    // Escape the prompt for JSON
    std::string escaped;
    for (char c : prompt) {
        if (c == '"') escaped += "\\\"";
        else if (c == '\\') escaped += "\\\\";
        else if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else if (c == '\t') escaped += "\\t";
        else escaped += c;
    }
    snprintf(body, sizeof(body),
        "{\"model\":\"qwen3:0.6b\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],"
        "\"max_tokens\":%d,\"temperature\":0.0,\"stream\":false}",
        escaped.c_str(), max_tokens);
    
    std::string response;
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        fprintf(stderr, "FLM call failed: %s\n", curl_easy_strerror(res));
        return "";
    }
    return response;
}

// Simple JSON parser: extract string field value
static std::string json_str(const std::string& json, const std::string& key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    auto colon = json.find(':', pos + key.size() + 2);
    if (colon == std::string::npos) return "";
    auto start = json.find('"', colon + 1);
    if (start == std::string::npos) return "";
    auto end = json.find('"', start + 1);
    if (end == std::string::npos) return "";
    return json.substr(start + 1, end - start - 1);
}

// Extract content from chat completion JSON
static std::string extract_content(const std::string& json) {
    // Find "content":"..." in the response
    auto cpos = json.find("\"content\"");
    if (cpos == std::string::npos) return "";
    auto colon = json.find(':', cpos + 9);
    if (colon == std::string::npos) return "";
    auto start = json.find('"', colon + 1);
    if (start == std::string::npos) return "";
    auto end = json.find('"', start + 1);
    if (end == std::string::npos) return "";
    std::string content = json.substr(start + 1, end - start - 1);
    // Unescape
    std::string out;
    for (size_t i = 0; i < content.size(); i++) {
        if (content[i] == '\\' && i + 1 < content.size()) {
            if (content[i+1] == 'n') out += '\n';
            else if (content[i+1] == 't') out += '\t';
            else if (content[i+1] == '"') out += '"';
            else if (content[i+1] == '\\') out += '\\';
            else { out += content[i]; out += content[i+1]; }
            i++;
        } else {
            out += content[i];
        }
    }
    return out;
}

struct SpecProxy {
    std::string flm_host = "127.0.0.1";
    int flm_port = 52625;
    MTPDraftModel* draft = nullptr;
    bool draft_loaded = false;
    
    bool init(const char* checkpoint, const std::string& host, int port) {
        flm_host = host;
        flm_port = port;
        MTPDraftConfig cfg;
        cfg.block_size = BLOCK_SIZE;
        draft = new MTPDraftModel(cfg);
        draft_loaded = draft->load_weights(checkpoint);
        printf("Draft checkpoint: %s\n", draft_loaded ? "✅ loaded" : "❌ not found (using untrained)");
        return draft_loaded;
    }
    
    // Simple speculative decode: ask FLM for next tokens, use draft for candidates
    std::string generate(const std::string& user_prompt, int max_tokens) {
        auto t0 = std::chrono::steady_clock::now();
        
        // For now, simple flow: ask FLM and return (baseline)
        // TODO: Implement proper speculative loop using token-level access
        std::string result = flm_complete(user_prompt, max_tokens, flm_host, flm_port);
        auto content = extract_content(result);
        
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        fprintf(stderr, "[spec] %.0f ms for %zu chars (%.0f tok/s est)\n",
                ms, content.size(), content.size() / (ms / 1000.0) / 4.0);
        
        return content;
    }
};

// Simple HTTP server
static void handle_client(int client_fd, SpecProxy& proxy) {
    char buf[65536];
    int n = read(client_fd, buf, sizeof(buf) - 1);
    if (n <= 0) { close(client_fd); return; }
    buf[n] = 0;
    
    std::string request(buf);
    
    // Parse request line
    auto path_start = request.find(' ') + 1;
    auto path_end = request.find(' ', path_start);
    std::string path = request.substr(path_start, path_end - path_start);
    
    std::string response;
    
    if (path == "/v1/models") {
        response = "{\"data\":[{\"id\":\"qwen3:0.6b+spec\",\"object\":\"model\"}]}";
    } else if (path == "/v1/chat/completions") {
        // Parse body
        auto body_start = request.find("\r\n\r\n");
        if (body_start == std::string::npos) {
            response = "{\"error\":\"bad request\"}";
        } else {
            std::string body = request.substr(body_start + 4);
            // Extract message content
            auto content_key = std::string("\"content\"");
            auto cpos = body.find(content_key);
            if (cpos != std::string::npos) {
                auto colon = body.find(':', cpos + content_key.size());
                auto start = body.find('"', colon + 1);
                auto end = body.find('"', start + 1);
                std::string user_msg = body.substr(start + 1, end - start - 1);
                
                // Extract max_tokens
                int max_tok = 128;
                auto mtpos = body.find("max_tokens");
                if (mtpos != std::string::npos) {
                    auto colon2 = body.find(':', mtpos + 11);
                    auto end2 = body.find_first_of(",} ", colon2 + 1);
                    if (end2 != std::string::npos)
                        max_tok = std::stoi(body.substr(colon2 + 1, end2 - colon2 - 1));
                }
                
                auto content = proxy.generate(user_msg, max_tok);
                
                // Build response
                char resp[131072];
                // Escape for JSON
                std::string escaped;
                for (char c : content) {
                    if (c == '"') escaped += "\\\"";
                    else if (c == '\\') escaped += "\\\\";
                    else if (c == '\n') escaped += "\\n";
                    else if (c == '\r') escaped += "\\r";
                    else if (c == '\t') escaped += "\\t";
                    else escaped += c;
                }
                snprintf(resp, sizeof(resp),
                    "{\"id\":\"chatcmpl-spec\",\"object\":\"chat.completion\","
                    "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                    "\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0}}",
                    escaped.c_str());
                response = resp;
            } else {
                response = "{\"error\":\"missing content\"}";
            }
        }
    } else {
        response = "{\"error\":\"not found\"}";
    }
    
    char http_resp[131072];
    snprintf(http_resp, sizeof(http_resp),
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Content-Length: %zu\r\n\r\n%s",
        response.size(), response.c_str());
    
    auto r = write(client_fd, http_resp, strlen(http_resp));
    (void)r;
    close(client_fd);
}

int main(int argc, char** argv) {
    int port = 9091;
    int flm_port = 52625;
    
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i+1 < argc) port = atoi(argv[++i]);
        if (!strcmp(argv[i], "--flm-port") && i+1 < argc) flm_port = atoi(argv[++i]);
    }
    
    printf("═══ Speculative Decode Proxy ═══\n");
    printf("FLM backend: 127.0.0.1:%d\n", flm_port);
    printf("Spec proxy: 127.0.0.1:%d\n", port);
    printf("Block size: %d\n\n", BLOCK_SIZE);
    
    SpecProxy proxy;
    proxy.init("/home/bcloud/spec-decode/checkpoints/dspark_draft.bin",
               "127.0.0.1", flm_port);
    
    // Create socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    listen(server_fd, 5);
    
    printf("Listening on http://127.0.0.1:%d\n", port);
    printf("Test: curl -X POST http://127.0.0.1:%d/v1/chat/completions -d "
           "'{\"model\":\"qwen3:0.6b+spec\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'\n\n",
           port);
    
    curl_global_init(CURL_GLOBAL_ALL);
    
    while (true) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) continue;
        std::thread(handle_client, client_fd, std::ref(proxy)).detach();
    }
    
    curl_global_cleanup();
    return 0;
}
