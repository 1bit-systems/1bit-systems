# Lemonade Compatibility

**Status:** Active · targeting Lemonade v10.9.0+

`1bit.systems` NPU and GPU engines are compatible with [Lemonade](https://github.com/lemonade-sdk/lemonade) via the OpenAI-compatible `/v1/chat/completions` endpoint.

## How it works

`1bit-halo-server` (port `:8180` — NPU engine) and the ZINC GPU engine both expose the standard OpenAI chat completions API. Lemonade can route omni-modal requests to either backend as a sub-process or remote server.

```yaml
# lemonade config.yaml
models:
  - name: qwen3-npu
    server:
      url: http://127.0.0.1:9090  # FLM proxy or 1bit-halo-server
      type: openai
```

## Quick start

```bash
# 1. Start the 1bit NPU daemon
./npu_engine_all model.q4nx 16

# 2. Lemond auto-discovers or point manually
lemond launch --model qwen3-npu
```

## v10.9.0+ integration

- **`*_bin` config keys** — Lemonade v10.3+ (PR [#1713](https://github.com/lemonade-sdk/lemonade/pull/1713)) accepts `builtin` / `latest` / version tag / local path values for every backend binary. The `1bit-systems` packaging can supply `ryzenai.server_bin` as `latest` or a pinned tag.
- **First-party contribution** — `bong-water-water-bong` is a [listed contributor](https://github.com/lemonade-sdk/lemonade/releases/tag/v10.9.0) in v10.9.0 (test/documentation: PR [#2447](https://github.com/lemonade-sdk/lemonade/pull/2447)).
- **Omni-modal routing** — No-code integration: point Lemonade's omni-modal UI at the 1bit OpenAI-compatible endpoint. Zero config changes on the `1bit.systems` side.

## Version compatibility table

| 1bit.systems | Lemonade | Status |
|---|---|---|
| v2026.07+ | v10.9.0 | ✅ Verified (Jul 2026) |
| v2026.04+ | v10.3 | ✅ Compatible (API stable) |
| v2026.04+ | v10.0+ | ✅ Basic chat |

## Updating

When a new Lemonade release ships:

1. Check the [Lemonade releases](https://github.com/lemonade-sdk/lemonade/releases) for breaking API changes
2. Smoke-test with `1bit-halo-server` on port `:9090`
3. Update this doc and the site wiki

## MCP support

Both projects expose MCP servers. See:
- [Lemonade MCP docs](https://github.com/lemonade-sdk/lemonade)
- 1bit.systems packaging docs
