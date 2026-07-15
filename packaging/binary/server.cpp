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

static std::string run_engine(const std::string& prompt, int max_tokens=8) {
    (void)prompt;
    const char* engine = getenv("NPU_ENGINE_BIN");
    if (!engine || !*engine) engine = "./engine/npu/build/npu_engine_cb";
    const char* libpath = getenv("NPU_LD_LIBRARY_PATH");

    auto has_metachar = [](const char* s) -> bool {
        for (const char* p = s; *p; ++p) {
            if (*p <= 32 || *p == ';' || *p == '|' || *p == '&' || *p == '$' || *p == '`' ||
                *p == '(' || *p == ')' || *p == '{' || *p == '}' || *p == '<' || *p == '>' ||
                *p == '!' || *p == '~' || *p == '\'' || *p == '"' || *p == '\\') return true;
        }
        return false;
    };
    if (has_metachar(engine)) {
        fprintf(stderr, "[1bit-server] refused: NPU_ENGINE_BIN contains shell metacharacters\n");
        return "{\"error\": \"invalid engine path\"}";
    }
    if (libpath && *libpath && has_metachar(libpath)) {
        fprintf(stderr, "[1bit-server] refused: NPU_LD_LIBRARY_PATH contains shell metacharacters\n");
        return "{\"error\": \"invalid library path\"}";
    }

    std::string cmd;
    if (libpath && *libpath) cmd += std::string("LD_LIBRARY_PATH=") + libpath + " ";
    cmd += std::string(engine) + " 9 " + std::to_string(max_tokens) + " 2>/dev/null";
    FILE* f = popen(cmd.c_str(), "r");
    if (!f) return "{\"error\": \"engine not found\"}";
    char buf[8192]; std::string out;
    while (fgets(buf, sizeof(buf), f)) out += buf;
    pclose(f);

    std::vector<std::string> tokens;
    size_t pos = 0;
    while ((pos = out.find("] ", pos)) != std::string::npos) {
        pos += 2;
        size_t end = out.find(" (", pos);
        if (end != std::string::npos) tokens.push_back(out.substr(pos, end - pos));
    }

    std::string speed = "unknown";
    if (auto p = out.rfind("ms/tok"); p != std::string::npos) {
        auto start = out.rfind(" ", p - 2);
        if (start != std::string::npos) speed = out.substr(start + 1, p - start - 1) + " ms/tok";
    }

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

static std::string extract_content(const std::string& req) {
    std::string prompt = "Hello";
    size_t pos = 0;
    while (true) {
        pos = req.find("\"content\"", pos);
        if (pos == std::string::npos) break;
        size_t ck = pos + 9;
        while (ck < req.size() && (req[ck] == ' ' || req[ck] == '\t')) ++ck;
        if (ck < req.size() && req[ck] == ':') {
            size_t vs = ck + 1;
            while (vs < req.size() && (req[vs] == ' ' || req[vs] == '\t')) ++vs;
            if (vs < req.size() && req[vs] == '"') {
                ++vs; size_t ve = vs;
                while (ve < req.size() && req[ve] != '"') {
                    if (req[ve] == '\\' && ve + 1 < req.size()) ve += 2;
                    else ++ve;
                }
                prompt = req.substr(vs, ve - vs);
                size_t w = 0; std::string unescaped;
                while (w < prompt.size()) {
                    if (prompt[w] == '\\' && w + 1 < prompt.size()) {
                        switch (prompt[w+1]) {
                            case 'n': unescaped += '\n'; break;
                            case 'r': unescaped += '\r'; break;
                            case 't': unescaped += '\t'; break;
                            case '"': unescaped += '"'; break;
                            case '\\': unescaped += '\\'; break;
                            default: unescaped += prompt[w+1]; break;
                        }
                        w += 2;
                    } else { unescaped += prompt[w++]; }
                }
                prompt = unescaped;
            }
            break;
        }
        pos = ck;
    }
    return prompt;
}

static void handle_client(int fd) {
    std::string req;
    char buf[4096];
    for (int i = 0; i < 64; i++) {
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n <= 0) break;
        req.append(buf, n);
        if (n < (ssize_t)sizeof(buf)) break;
    }
    if (req.empty()) { close(fd); return; }

    std::string prompt = extract_content(req);
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
    const char* bind_all = getenv("ONEBIT_BIND_ALL");
    addr.sin_addr.s_addr = (bind_all && *bind_all == '1') ? INADDR_ANY : htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    bind(sock, (struct sockaddr*)&addr, sizeof(addr));
    listen(sock, 5);
    printf("1bit.systems NPU API -> http://localhost:%d/v1/chat/completions\n", port);
    printf("  Model: Qwen3-0.6B INT8\n\n");
    while (true) {
        int client = accept(sock, nullptr, nullptr);
        std::thread(handle_client, client).detach();
    }
}