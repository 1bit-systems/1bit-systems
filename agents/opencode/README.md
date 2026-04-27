# OpenCode + halo-server

OpenCode (sst/opencode) BYOK preset pointing at local halo-server :8180.

## Install

```sh
mkdir -p ~/.config/opencode
cp halo.json ~/.config/opencode/opencode.json
```

Or merge `provider.halo-local` into existing `~/.config/opencode/opencode.json`.

## Auth

halo-server uses bearer middleware. Export key before launching opencode:

```sh
export HALO_API_KEY="$(cat ~/.config/halo/api-key)"
opencode
```

## Models

- `halo-1bit-2b-4t-q` — Sherry 1.25 bpw (post Run 6 ship, currently pending Path A)
- `halo-1bit-2b-4t-bf16` — bf16 baseline

Adjust `contextWindow` per actual model card. Defaults assume 8K rope.

## Endpoint

halo-server binds 127.0.0.1:8180, OpenAI-compatible (`/v1/chat/completions`, `/v1/models`).

For LAN access, port-forward via Headscale (100.64.0.1) or override `baseURL`.

## See also

- Lemonade ships `lemonade launch opencode` (v10.2.0) — same shape, hosted helper.
- halo equivalent CLI launcher (`halo launch opencode`) is post-Sherry-ship work.
