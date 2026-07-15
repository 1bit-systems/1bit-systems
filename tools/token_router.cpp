#include "rocm_cpp/ck_gemm.h"
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
#include <curl/curl.h>
#include <json/json.h>

static void send_json(int cl, int code, const std::string& body) {
    const char* reason = "OK";
    switch (code) {
        case 200: reason = "OK"; break;
        case 400: reason = "Bad Request"; break;
        case 404: reason = "Not Found"; break;
        case 413: reason = "Payload Too Large"; break;
        case 500: reason = "Internal Server Error"; break;
    }
    std::string header = "HTTP/1.1 " + std::to_string(code) + " " + reason + "\r\n"
        "Content-Type: application/json\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n"
        "\r\n";
    write(cl, header.data(), header.size());
    write(cl, body.data(), body.size());
    close(cl);
}