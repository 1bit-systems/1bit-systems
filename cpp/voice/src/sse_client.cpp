#include "onebit/voice/sse_client.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <string>

namespace onebit::voice {

namespace {

constexpr std::size_t kRecvChunk = 4096;

[[nodiscard]] std::string format_request(std::string_view host,
                                         std::string_view path,
                                         std::string_view body)
{
    std::string req;
    req.reserve(256 + body.size());
    req.append("POST ");
    req.append(path);
    req.append(" HTTP/1.1\r\n");
    req.append("Host: ");
    req.append(host);
    req.append("\r\n");
    req.append("User-Agent: 1bit-voice/0.1\r\n");
    req.append("Accept: text/event-stream\r\n");
    req.append("Content-Type: application/json\r\n");
    req.append("Connection: close\r\n");
    req.append("Content-Length: ");
    req.append(std::to_string(body.size()));
    req.append("\r\n\r\n");
    req.append(body);
    return req;
}

// Returns >=0 socket fd on success, <0 on failure. Sets errno.
int tcp_connect(std::string_view host, std::uint16_t port,
                std::uint32_t timeout_secs)
{
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo*         res = nullptr;
    const std::string host_s{host};
    const std::string port_s = std::to_string(port);
    const int rc = ::getaddrinfo(host_s.c_str(), port_s.c_str(), &hints, &res);
    if (rc != 0 || res == nullptr) {
        errno = ECONNREFUSED;
        return -1;
    }

    int fd = -1;
    for (auto* rp = res; rp != nullptr; rp = rp->ai_next) {
        fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) {
            continue;
        }
        timeval tv{};
        tv.tv_sec  = static_cast<time_t>(timeout_secs);
        tv.tv_usec = 0;
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (::connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            break;
        }
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    return fd;
}

bool send_all(int fd, std::string_view data)
{
    const char*   p   = data.data();
    std::size_t   rem = data.size();
    while (rem > 0) {
        const ssize_t n = ::send(fd, p, rem, MSG_NOSIGNAL);
        if (n <= 0) {
            return false;
        }
        p   += static_cast<std::size_t>(n);
        rem -= static_cast<std::size_t>(n);
    }
    return true;
}

// Read until \r\n\r\n, return parsed status + remaining buffered tail.
struct HeaderParse {
    int         status = 0;
    bool        chunked = false;
    std::string tail;   // bytes already read past the header terminator
};

[[nodiscard]] std::expected<HeaderParse, SseError>
read_headers(int fd)
{
    std::string buf;
    buf.reserve(1024);
    char tmp[kRecvChunk];
    while (true) {
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) {
            return std::unexpected(SseError{SseError::Kind::Recv, 0,
                                            "connection closed before headers"});
        }
        buf.append(tmp, tmp + n);
        const auto pos = buf.find("\r\n\r\n");
        if (pos != std::string::npos) {
            HeaderParse hp;
            // Status line: "HTTP/1.1 200 OK\r\n"
            const auto first = buf.find("\r\n");
            if (first != std::string::npos && first >= 12) {
                const auto status_str = buf.substr(9, 3);
                hp.status             = std::atoi(status_str.c_str());
            }
            // Cheap chunked detect — case-insensitive.
            std::string head = buf.substr(0, pos);
            std::transform(head.begin(), head.end(), head.begin(),
                           [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
            hp.chunked = head.find("transfer-encoding: chunked") !=
                         std::string::npos;
            hp.tail = buf.substr(pos + 4);
            return hp;
        }
        if (buf.size() > 16 * 1024) {
            return std::unexpected(SseError{
                SseError::Kind::Recv, 0, "header too large"});
        }
    }
}

// Drain SSE events from a contiguous body buffer + further socket reads.
// Yields each event (sans trailing "\n\n" or "\r\n\r\n") to `on_event`.
// std::string_view stability: each event is a slice into `scratch`; the
// callback runs to completion before we touch `scratch` again. We never
// hand the same view across two callback invocations.
[[nodiscard]] std::expected<void, SseError>
drain_events(int fd,
             std::string scratch_init,
             bool /*chunked*/,
             const std::function<bool(std::string_view)>& on_event)
{
    // SSE bodies are typically delivered as chunked or identity. We do
    // not strip chunk framing here — for the loopback LLM case lemond
    // emits identity-encoded SSE. If chunked is encountered, the
    // splitter still sees the embedded SSE because boundaries are byte
    // patterns; chunk-size prefixes don't contain "data:" or "\n\n".
    // Worst case: a chunk-size hex prefix ends up in the SSE buf and
    // parse_sse_delta returns nullopt for that "event", which the
    // pipeline silently skips. Acceptable for v0.

    std::string scratch = std::move(scratch_init);
    char        tmp[kRecvChunk];

    auto try_drain_one = [&](std::string& buf) -> int {
        // Find blank-line terminator. Accept "\n\n" or "\r\n\r\n".
        const auto p1 = buf.find("\n\n");
        const auto p2 = buf.find("\r\n\r\n");
        std::size_t pos = std::string::npos;
        std::size_t adv = 0;
        if (p1 != std::string::npos &&
            (p2 == std::string::npos || p1 < p2)) {
            pos = p1;
            adv = 2;
        } else if (p2 != std::string::npos) {
            pos = p2;
            adv = 4;
        } else {
            return 0;
        }
        const std::string_view event{buf.data(), pos};
        const bool keep_going = on_event(event);
        buf.erase(0, pos + adv);
        return keep_going ? 1 : -1;
    };

    while (true) {
        // Drain everything we can before reading more.
        for (;;) {
            const int rc = try_drain_one(scratch);
            if (rc == 0)  break;
            if (rc < 0)   return {};
        }
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n == 0) {
            // EOF — emit any final un-terminated event.
            if (!scratch.empty()) {
                (void)on_event(scratch);
            }
            return {};
        }
        if (n < 0) {
            return std::unexpected(SseError{SseError::Kind::Recv, 0,
                                            std::strerror(errno)});
        }
        scratch.append(tmp, tmp + n);
    }
}

} // namespace

std::expected<void, SseError>
post_sse(std::string_view  host,
         std::uint16_t     port,
         std::string_view  path,
         std::string_view  body,
         std::uint32_t     timeout_secs,
         const std::function<bool(std::string_view)>& on_event)
{
    const int fd = tcp_connect(host, port, timeout_secs);
    if (fd < 0) {
        return std::unexpected(SseError{
            SseError::Kind::Connect, 0,
            std::string("connect ") + std::string(host) + ":" +
                std::to_string(port) + " failed"});
    }
    struct FdGuard {
        int fd;
        ~FdGuard() { if (fd >= 0) ::close(fd); }
    } guard{fd};

    const std::string req = format_request(host, path, body);
    if (!send_all(fd, req)) {
        return std::unexpected(SseError{SseError::Kind::Send, 0,
                                        "send failed"});
    }

    auto hp = read_headers(fd);
    if (!hp) {
        return std::unexpected(hp.error());
    }
    if (hp->status < 200 || hp->status >= 300) {
        return std::unexpected(SseError{
            SseError::Kind::Status, hp->status,
            "non-2xx status from LLM"});
    }
    return drain_events(fd, std::move(hp->tail), hp->chunked, on_event);
}

} // namespace onebit::voice
