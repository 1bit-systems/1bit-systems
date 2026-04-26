// 1bit-mcp — stdio JSON-RPC entry point.
//
// Logs go to stderr so they never corrupt the JSON-RPC stream on stdout.
// Tool surface is empty post-2026-04-25 cull; preserved as a slot for
// the GAIA agent-core re-target.

#include "onebit/mcp/server.hpp"

#include <cstdio>
#include <iostream>

int main()
{
    std::fprintf(stderr,
                 "1bit-mcp starting (version=%s, tools=0; post-agents-cull stub)\n",
                 std::string(onebit::mcp::SERVER_VERSION).c_str());

    onebit::mcp::StdioServer server{};
    server.run(std::cin, std::cout);

    std::fprintf(stderr, "1bit-mcp stdin closed, exiting 0\n");
    return 0;
}
