# Add your own app

Third-party apps attach on the client side only. Rule A hard stop: no Python, no Node, no interpreted runtime inside `1bit-halo-server` or downstream. Anything above it — UIs, agents, game bots, IDE plugins — is fair game in any language.

## The shape of a caller-side app

1. Speak OpenAI-compatible HTTP to `:8180`. Every SDK works.
2. Hit `GET /v1/models` to discover the real `.h1b` registry on this box — the response is derived from what the server actually loaded, not a hardcoded list.
3. Always send a `model` field on `/v1/chat/completions` that matches one of those IDs. Unknown model IDs now return a structured error (the server validates against the registry).
4. For richer introspection, connect to `1bit-halo-mcp` at `:8181`.
5. Use `1bit-halo-server`'s session header `X-1bit-halo-Session` to pin a conversation to a KV-cache slot.
6. Handle `429` / `503` with exponential back-off — the server returns `Retry-After`.

## Example — minimal agent harness

```typescript
// minimal-agent.ts · run with `bun run minimal-agent.ts`
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8180/v1",
  apiKey: "none",
});

const session = crypto.randomUUID();
const history: OpenAI.ChatCompletionMessageParam[] = [
  { role: "system", content: "You are a terse assistant." },
];

async function turn(user: string) {
  history.push({ role: "user", content: user });
  const res = await client.chat.completions.create(
    { model: "1bit-halo-v2", messages: history },
    { headers: { "X-1bit-halo-Session": session } },
  );
  const reply = res.choices[0].message.content ?? "";
  history.push({ role: "assistant", content: reply });
  return reply;
}

console.log(await turn("Two facts about RDNA 3.5."));
console.log(await turn("And one that contradicts a common myth."));
```

## Where your app lives

If the app is a serving surface (game integration, Discord bot, MCP bridge, API adapter), it belongs in **1bit.services**, not in 1bit.systems core. Core stays kernel + serving only.

If the app is a library meant to be embedded (SDK wrapper, client helper), keep it in your own repo. The project maintains the HTTP contract; you maintain the client surface.

> **API stability** — the OpenAI-compatible surface is the stable contract. The FFI boundary between `1bit-halo-server` and `bitnet_decode` is internal and changes without notice. Build on HTTP, not on FFI.
