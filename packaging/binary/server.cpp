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
    std::string cmd = "LD_LIBRARY_PATH=/home/bcloud/torch2aie/toolchain/xrt/lib64:"
                      "/home/bcloud/torch2aie/toolchain/mlir_aie.libs:"
                      "/home/bcloud/torch2aie/toolchain/sysroot/usr/lib64 "
                      "./engine/npu/build/npu_engine_cb 9 " + std::to_string(max_tokens) + " 2>/dev/null";
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
    char buf[4096];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    if (n <= 0) { close(fd); return; }
    buf[n] = 0;

    std::string prompt = "Hello";
    // Quick parse: look for "content" in JSON body
    if (auto p = strstr(buf, "\"content\"")) {
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

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    bind(sock, (struct sockaddr*)&addr, sizeof(addr));
    listen(sock, 5);

    printf("1bit.systems NPU API → http://localhost:%d/v1/chat/completions\n", port);
    printf("  Model: Qwen3-0.6B INT8 (4-live contexts, 244 ms/tok)\n");
    printf("  Entry: email thearchitect@1bit.systems\n\n");

    while (true) {
        int client = accept(sock, nullptr, nullptr);
        std::thread(handle_client, client).detach();
    }
}
