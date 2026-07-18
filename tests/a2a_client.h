// a2a_client.h — A2A (Agent-to-Agent) Protocol v1.0 Client
// Discovers remote agents, sends tasks, routes inference across peers.
// Part of Google's open A2A protocol: https://github.com/a2aproject/a2a

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <netinet/in.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

struct A2AClient {
    struct AgentInfo {
        std::string name;
        std::string base_url;      // e.g. "http://host:port"
        std::string api_path;      // e.g. "/a2a/v1"
        std::string description;
        bool streaming = false;
        std::vector<std::string> skill_ids;
        std::map<std::string, json> skill_configs;
        bool alive = false;
    };

    std::vector<AgentInfo> peers;  // discovered or manually registered

    // ── Simple raw-socket HTTP client ──────────────────────────────
    // Returns (status_code, body)
    static std::pair<int, std::string> http_request(
        const std::string& method, const std::string& url,
        const std::string& body = "", const std::string& content_type = "")
    {
        // Parse URL: http://host:port/path
        std::string host, path = "/";
        int port = 80;
        size_t p = url.find("://");
        size_t start = (p == std::string::npos) ? 0 : p + 3;
        size_t slash = url.find('/', start);
        std::string hostpart;
        if (slash == std::string::npos) {
            hostpart = url.substr(start);
        } else {
            hostpart = url.substr(start, slash - start);
            path = url.substr(slash);
        }
        size_t colon = hostpart.find(':');
        if (colon != std::string::npos) {
            host = hostpart.substr(0, colon);
            port = atoi(hostpart.substr(colon + 1).c_str());
        } else {
            host = hostpart;
        }
        if (port <= 0) port = 80;
        if (path.empty()) path = "/";

        // Resolve host
        struct hostent* he = gethostbyname(host.c_str());
        if (!he) return {0, "DNS resolution failed"};

        // Connect
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return {0, "socket() failed"};
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        memcpy(&addr.sin_addr, he->h_addr, he->h_length);
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(sock);
            return {0, "connect() failed"};
        }

        // Build & send request
        std::string req = method + " " + path + " HTTP/1.1\r\n"
            "Host: " + host + "\r\n"
            "User-Agent: 1bit-systems-a2a/1.0\r\n"
            "Connection: close\r\n";
        if (!body.empty()) {
            req += "Content-Type: " + (content_type.empty() ? "application/a2a+json" : content_type) + "\r\n";
            req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
        }
        req += "\r\n" + body;
        write(sock, req.data(), req.size());

        // Read response (up to 64KB)
        std::vector<char> buf(65536);
        int n = read(sock, buf.data(), buf.size() - 1);
        close(sock);
        if (n <= 0) return {0, "no response"};

        buf[n] = 0;
        std::string resp(buf.data());

        // Parse status line
        auto sp = resp.find(' ');
        auto cr = resp.find("\r\n");
        int code = 0;
        if (sp != std::string::npos && cr != std::string::npos)
            code = atoi(resp.substr(sp + 1, cr - sp - 1).c_str());

        // Parse body (after \r\n\r\n)
        auto bp = resp.find("\r\n\r\n");
        std::string rbody = (bp == std::string::npos) ? "" : resp.substr(bp + 4);
        return {code, rbody};
    }

    // ── Discover an agent by fetching its Agent Card ───────────────
    bool discover(const std::string& base_url) {
        // Check if already registered
        for (auto& p : peers)
            if (p.base_url == base_url) return true;

        std::string card_url = base_url + "/.well-known/agent-card.json";
        auto [code, body] = http_request("GET", card_url);
        if (code != 200 || body.empty()) {
            fprintf(stderr, "  [a2a] discover %s → %d (no card)\n", card_url.c_str(), code);
            return false;
        }

        try {
            json j = json::parse(body);
            AgentInfo info;
            info.base_url = base_url;
            info.name = j.value("name", "unknown");
            info.description = j.value("description", "");
            info.streaming = j.value("capabilities", json::object()).value("streaming", false);

            // Register API path from supportedInterfaces
            if (j.contains("supportedInterfaces") && j["supportedInterfaces"].is_array()) {
                for (auto& iface : j["supportedInterfaces"]) {
                    std::string url = iface.value("url", "");
                    if (!url.empty()) {
                        info.api_path = url;
                        break;
                    }
                }
            }
            if (info.api_path.empty()) info.api_path = base_url + "/a2a/v1";

            // Extract skills
            if (j.contains("skills") && j["skills"].is_array()) {
                for (auto& skill : j["skills"]) {
                    std::string sid = skill.value("id", "");
                    if (!sid.empty()) {
                        info.skill_ids.push_back(sid);
                        info.skill_configs[sid] = skill;
                    }
                }
            }

            info.alive = true;
            peers.push_back(info);
            fprintf(stderr, "  [a2a] discovered '%s' @ %s (%zu skills)\n",
                    info.name.c_str(), base_url.c_str(), info.skill_ids.size());
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "  [a2a] parse error from %s: %s\n", card_url.c_str(), e.what());
            return false;
        }
    }

    // ── Send a text-generation task to a remote agent ─────────────
    struct TaskResult {
        bool success = false;
        std::string text;
        std::string task_id;
        int prompt_tokens = 0;
        int completion_tokens = 0;
        std::string error;
    };

    TaskResult send_task(const std::string& agent_url, const std::string& prompt,
                          int max_tokens = 256, double temperature = 0.7)
    {
        TaskResult result;

        // Build A2A message
        json msg = {
            {"message", {
                {"role", "ROLE_USER"},
                {"parts", json::array({{{"text", prompt}}})}
            }},
            {"configuration", {
                {"maxTokens", max_tokens},
                {"temperature", temperature}
            }}
        };

        // Find the agent's message:send endpoint
        std::string send_url = agent_url;
        // If agent_url has a base like http://host:port/a2a/v1, append /message:send
        if (send_url.back() == '/') send_url += "message:send";
        else send_url += "/message:send";

        auto [code, body] = http_request("POST", send_url, msg.dump());

        if (code != 200) {
            result.error = "HTTP " + std::to_string(code) + ": " + body.substr(0, 200);
            return result;
        }

        try {
            json resp = json::parse(body);
            auto& task = resp["task"];
            result.task_id = task.value("id", "");

            std::string state = task["status"].value("state", "");
            if (state == "TASK_STATE_FAILED") {
                result.error = task["status"].value("message", json::object()).value("parts", json::array())[0].value("text", "unknown error");
                return result;
            }

            // Extract generated text from artifacts
            if (task.contains("artifacts") && task["artifacts"].is_array() && !task["artifacts"].empty()) {
                auto& artifact = task["artifacts"][0];
                if (artifact.contains("parts") && artifact["parts"].is_array() && !artifact["parts"].empty()) {
                    result.text = artifact["parts"][0].value("text", "");
                }
                if (artifact.contains("metadata")) {
                    result.prompt_tokens = artifact["metadata"].value("promptTokens", 0);
                    result.completion_tokens = artifact["metadata"].value("completionTokens", 0);
                }
            }

            result.success = !result.text.empty();
            return result;
        } catch (const std::exception& e) {
            result.error = std::string("parse: ") + e.what();
            return result;
        }
    }

    // ── Route a task to the best agent by skill match ──────────────
    TaskResult route_by_skill(const std::string& prompt, const std::string& skill_id = "",
                               int max_tokens = 256, double temperature = 0.7)
    {
        // Filter alive peers
        std::vector<AgentInfo*> candidates;
        for (auto& p : peers) {
            if (!p.alive) continue;
            if (skill_id.empty()) {
                candidates.push_back(&p);
            } else {
                for (auto& sid : p.skill_ids) {
                    if (sid == skill_id) {
                        candidates.push_back(&p);
                        break;
                    }
                }
            }
        }

        if (candidates.empty()) {
            TaskResult r;
            r.error = "no peer agent available" + (skill_id.empty() ? "" : " with skill '" + skill_id + "'");
            return r;
        }

        // Pick the first matching peer (round-robin could go here)
        AgentInfo* agent = candidates[0];
        TaskResult r = send_task(agent->api_path.empty() ? agent->base_url : agent->api_path,
                                  prompt, max_tokens, temperature);
        if (r.success) {
            fprintf(stderr, "  [a2a] routed to '%s' (%s): %d tokens\n",
                    agent->name.c_str(), agent->base_url.c_str(), r.completion_tokens);
        }
        return r;
    }

    // ── Health check all registered peers ──────────────────────────
    void health_check() {
        for (auto& p : peers) {
            auto [code, _] = http_request("GET", p.base_url + "/");
            p.alive = (code == 200);
            if (!p.alive)
                fprintf(stderr, "  [a2a] peer '%s' @ %s unreachable (HTTP %d)\n",
                        p.name.c_str(), p.base_url.c_str(), code);
        }
    }
};
