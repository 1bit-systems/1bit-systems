# 1bit install mcp — runbook

## What this installs

`1bit-mcp` — stdio JSON-RPC MCP bridge. Claude Code, Continue.dev, and
any other MCP-aware client launches it on demand. Binary only, **no
systemd unit**.

The tool list is currently minimal (post-2026-04-25 cull); the C++
bridge is being re-targeted at GAIA agent-core in the next pass.

## Prereqs

- `core` installed (the CLI registers the binary path)
- nlohmann/json available at build time (vendored in `cpp/third_party`)

## Install

```sh
1bit install mcp
```

Builds `cpp/build/strix/mcp/1bit-mcp` and installs to
`${HOME}/.local/bin/1bit-mcp`.

## Wiring into Claude Code

Add to `~/.config/Claude/mcp.json`:

```json
{
  "servers": {
    "1bit": {
      "command": "/home/<you>/.local/bin/1bit-mcp",
      "args":    []
    }
  }
}
```

Then restart Claude Code. `tools/list` should return the registered
methods.

## Common errors

- **`parse error -32700` on every request** — the client is sending
  raw JSON without newline framing. `1bit-mcp` reads line-delimited
  JSON; verify the client uses LF terminators.
- **Tools list is empty** — expected for now; this binary is being
  retargeted. See `cpp/mcp/src/server.cpp` for the registered set.

## Logs

`1bit-mcp` writes diagnostics to stderr. The launching client
typically captures these — check the client's MCP debug log.

## Rollback

```sh
rm -f ${HOME}/.local/bin/1bit-mcp
```

Then remove the entry from the client's `mcp.json` and restart it.
