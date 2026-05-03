# Binary Packaging Status

This directory is experimental and currently behind the shell/Node implementation in `scripts/`.

Canonical runtime today:

- `scripts/1bit`
- `scripts/1bit-proxy.js`
- `install.sh`
- systemd `1bit-stack.target`

The Bun binary port must not be shipped as parity until it has:

- GAIA subcommands matching `scripts/1bit`: `gaia up`, `gaia down`, `gaia status`, `gaia ui`, `gaia cli`, `gaia api`, `gaia mcp`, `gaia logs`.
- OmniRouter helper command parity.
- FLM default endpoint parity: `http://127.0.0.1:52625`.
- Proxy behavior parity with `scripts/1bit-proxy.js`: multipart audio model parsing, request body limit, embeddings/audio/responses routing, realtime WebSocket upgrade handling, and fresh sockets for FLM.
- Bind parity: Lemonade `0.0.0.0`, proxy `0.0.0.0`, Open WebUI `0.0.0.0`.
- Site/home copy updated to the GAIA + Lemonade + FastFlowLM architecture.

Until then, treat files in this directory as packaging work-in-progress, not a release surface.
