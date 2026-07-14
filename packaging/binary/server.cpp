/** 1bit-systems HTTP API — OpenAI-compatible /v1/chat/completions
 *  Pure C++. Single binary. No dependencies beyond C++23 + POSIX sockets.
 *  Build: g++ -std=c++23 -O3 -o 1bit-server server.cpp
 *  Run:   ./1bit-server [port]
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sstream>

static std::string run_engine(const std::string& prompt, int max_tokens=8) {
    // Paths are configurable so the shipped binary isn't tied to one machine (#149).
    //   NPU_ENGINE_BIN        — path to the npu_engine_cb binary (default: relative)
    //   NPU_LD_LIBRARY_PATH   — extra dirs for the XRT/MLIR-AIE shared libraries
    // NOTE: the engine CLI takes token counts, not text; wiring the user's
    // `prompt` through requires a tokenizer this single binary does not embed.
    (void)prompt;
    const char* engine = getenv("NPU_ENGINE_BIN");
    if (!engine || !*engine) engine = "./engine/npu/build/npu_engine_cb";
    const char* libpath = getenv("NPU_LD_LIBRARY_PATH");
    std::string cmd;
    if (libpath && *libpath) cmd += std::string("LD_LIBRARY_PATH=") + libpath + " ";
    cmd += std::string(engine) + " 9 " + std::to_string(max_tokens) + " 2>/dev/null";
    FILE* f = popen(cmd.c_str(), "r");
    if (!f) return "{\"error\": \"engine not found\"}";
    char buf[8192]; std::string out;
    while (fgets(buf, sizeof(buf), f)) out += buf;
    pclose(f);

    // Parse tokens from engine output
    std::vector<std::string> tokens;
    size_t pos = 0;
    while ((pos = out.find("] ", pos)) != std::string::npos) {
        pos += 2;
        size_t end = out.find(" (", pos);
        if (end != std::string::npos) tokens.push_back(out.substr(pos, end - pos));
    }

    // Parse ms/tok
    std::string speed = "unknown";
    if (auto p = out.rfind("ms/tok"); p != std::string::npos) {
        auto start = out.rfind(" ", p - 2);
        if (start != std::string::npos) speed = out.substr(start + 1, p - start - 1) + " ms/tok";
    }

    // Build JSON
    std::string json = "{\"object\":\"chat.completion\",\"model\":\"qwen3-0.6b-npu\",";
    json += "\"speed\":\"" + speed + "\",";
    json += "\"choices\":[{\"message\":{\"content\":\"";
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) json += " ";
        json += "[tok:" + tokens[i] + "]";
    }
    json += "\"}}],";
    json += "\"usage\":{\"total_tokens\":" + std::to_string(tokens.size()) + "}";
    json += "}";
    return json;
}

static void handle_client(int fd) {
    // Loop the read so requests larger than one buffer / split across TCP
    // segments aren't truncated (#149). Cap total to a sane limit.
    std::string req;
    char buf[4096];
    for (int i = 0; i < 64; i++) { // up to 256 KiB
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n <= 0) break;
        req.append(buf, n);
        // Stop once we have headers + any body and the socket has no more queued.
        if (n < (ssize_t)sizeof(buf)) break;
    }
    if (req.empty()) { close(fd); return; }

    std::string prompt = "Hello";
    // Quick parse: look for "content" in JSON body
    if (auto p = strstr(req.c_str(), "\"content\"")) {
        auto start = strchr(p, '"');
        if (start) {
            start = strchr(start + 1, '"');
            if (start) {
                start++;
                auto end = strchr(start, '"');
                if (end) prompt = std::string(start, end - start);
            }
        }
    }

    std::string body = run_engine(prompt, 4);
    std::string response =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "\r\n" + body;

    write(fd, response.c_str(), response.size());
    close(fd);
}

int main(int argc, char** argv) {
    int port = (argc > 1) ? atoi(argv[1]) : 8080;
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Bind to loopback by default (unauthenticated exec-spawning server); set
    // ONEBIT_BIND_ALL=1 to expose on 0.0.0.0 intentionally (#149).
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    const char* bind_all = getenv("ONEBIT_BIND_ALL");
    addr.sin_addr.s_addr = (bind_all && *bind_all == '1') ? INADDR_ANY : htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    bind(sock, (struct sockaddr*)&addr, sizeof(addr));
    listen(sock, 5);

    printf("1bit.systems NPU API → http://localhost:%d/v1/chat/completions\n", port);
    printf("  Model: Qwen3-0.6B INT8 (4-live contexts, 244 ms/tok)\n");
    printf("  Entry: admin@1bit.systems\n\n");

    while (true) {
        int client = accept(sock, nullptr, nullptr);
        std::thread(handle_client, client).detach();
    }
}
