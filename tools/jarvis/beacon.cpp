#include "beacon.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

using json = nlohmann::json;

namespace jarvis {
namespace {

std::string lan_ip() {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) return "127.0.0.1";

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(80);
    inet_pton(AF_INET, "8.8.8.8", &addr.sin_addr);

    // connect() on a UDP socket doesn't send a packet — it just picks the
    // outbound-facing local address for getsockname(), same trick the
    // original Python used.
    std::string result = "127.0.0.1";
    if (connect(sock, (sockaddr*)&addr, sizeof(addr)) == 0) {
        sockaddr_in local{};
        socklen_t len = sizeof(local);
        if (getsockname(sock, (sockaddr*)&local, &len) == 0) {
            char buf[INET_ADDRSTRLEN];
            if (inet_ntop(AF_INET, &local.sin_addr, buf, sizeof(buf))) result = buf;
        }
    }
    close(sock);
    return result;
}

std::string hostname() {
    char buf[256];
    if (gethostname(buf, sizeof(buf)) == 0) return buf;
    return "unknown";
}

} // namespace

void start_beacon(int server_port, double interval_seconds) {
    std::string url = "http://" + lan_ip() + ":" + std::to_string(server_port);
    std::string host = hostname();
    json payload = {{"service", "1bit"}, {"hostname", host}, {"url", url}};
    std::string payload_str = payload.dump();

    fprintf(stderr, "Beacon: advertising %s on UDP :%d\n", url.c_str(), kBeaconPort);

    std::thread([payload_str, interval_seconds]() {
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) return;
        int broadcast_enable = 1;
        setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast_enable, sizeof(broadcast_enable));

        sockaddr_in dest{};
        dest.sin_family = AF_INET;
        dest.sin_port = htons(kBeaconPort);
        dest.sin_addr.s_addr = INADDR_BROADCAST; // 255.255.255.255

        while (true) {
            sendto(sock, payload_str.data(), payload_str.size(), 0, (sockaddr*)&dest, sizeof(dest));
            std::this_thread::sleep_for(std::chrono::milliseconds((long)(interval_seconds * 1000)));
        }
    }).detach();
}

} // namespace jarvis
