// 1bit-mcp-linuxgsm — stdio JSON-RPC entry point.
//
// stderr only — never write to stdout outside the JSON-RPC stream.

#include "onebit/mcp_linuxgsm/server.hpp"

#include <cstdio>
#include <iostream>

int main()
{
    namespace lx = onebit::mcp_linuxgsm;
    const auto root = lx::gsm_root();
    std::fprintf(stderr,
                 "1bit-mcp-linuxgsm starting (version=%s, root=%s)\n",
                 std::string(lx::SERVER_VERSION).c_str(),
                 root.string().c_str());

    lx::run(std::cin, std::cout, root, lx::run_driver_process);
    return 0;
}
