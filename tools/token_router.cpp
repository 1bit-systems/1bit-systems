#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <curl/curl.h>
struct CurlBuffer { std::string data; bool headers_done; };
static size_t curl_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    size_t total = size * nmemb; auto* buf = (CurlBuffer*)userdata;
    buf->data.append(ptr, total); return total;
}
static std::string http_get(const std::string& url, long timeout=5) {
    CURL* curl = curl_easy_init(); if (!curl) return "";
    CurlBuffer buf;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str()); curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb); curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf); curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout); curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 3);
    curl_easy_perform(curl); curl_easy_cleanup(curl); return buf.data;
}
static std::string http_post(const std::string& url, const std::string& body, long timeout=30) {
    CURL* curl = curl_easy_init(); if (!curl) return "";
    CurlBuffer buf; struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str()); curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str()); curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body.size()); curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb); curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf); curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout); curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5);
    curl_easy_perform(curl); curl_slist_free_all(headers); curl_easy_cleanup(curl); return buf.data;
}
static void send_json(int cl, int code, const std::string& body) {
    const char* reason = "OK";
    switch (code) { case 200: reason = "OK"; break; case 400: reason = "Bad Request"; break; case 404: reason = "Not Found"; break; case 413: reason = "Payload Too Large"; break; case 500: reason = "Internal Server Error"; break; }
    std::string header = "HTTP/1.1 " + std::to_string(code) + " " + reason + "\r\n"
        "Content-Type: application/json\r\n" "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n" "Connection: close\r\n" "\r\n";
    write(cl, header.data(), header.size()); write(cl, body.data(), body.size()); close(cl);
}