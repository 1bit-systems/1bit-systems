# Cross-agent consensus routing + echo voice

Each daemon is armed with one specialty and one model. Two specialty
recipes from two different sources cover the surface:

| Agent          | Specialty            | Model            | Source            |
| -------------- | -------------------- | ---------------- | ----------------- |
| halo-helpdesk  | install / runbook    | halo-1bit-2b     | own (BitNet 1.58) |
| halo-coder     | C++ tower + kernels  | Qwen3-8B-GGUF    | Alibaba (Qwen3)   |

Both run as `halo-agent@<name>.service` instances. Both speak to lemond
on `:8180/v1`. The brain endpoint is shared; only `agent.model` and
`agent.system_prompt` differ.

## Consensus protocol

`agent_consult` is a normal MCP tool. The agent that calls it issues a
one-shot OpenAI completion against the peer's recipe and folds the reply
into its own context as a `role=tool` message before composing its final
answer.

```
user → halo-helpdesk
       ├── apply install_runbook
       ├── agent_consult{peer="halo-coder", q="kernel pack layout"}
       │      └── lemond completes with Qwen3-8B + halo-coder system
       │            prompt → returns text → halo-helpdesk sees it as tool result
       └── compose final answer (own model, own prompt + peer's reply)
```

No special wire format. Both agents share lemond, so the round-trip is
just one extra POST. There is no leader election; whoever the user
addressed drives the conversation, and consults the peer when its
specialty applies.

Cap consult depth at `max_tool_iters` (default 5–6). The peer cannot
re-consult back into the asker (the tool refuses self-target by name to
avoid loops).

## Speak through echo

`speak_to_echo` is opt-in. Triggered when:

1. User explicitly asks for voice ("voice:" prefix or "speak this back").
2. `auto_speak = true` in the agent's TOML (off by default).

The tool POSTs the agent's final reply (post-consult) to
`echo_url = http://127.0.0.1:8181/v1/tts` and returns the audio URL.
The text reply still goes back to Discord/Telegram; only the voice
mirror is opt-in.

## Wiring summary

```
configs/halo-helpdesk.toml   ← owns install_runbook + agent_consult{halo-coder}
configs/halo-coder.toml      ← owns repo_search + agent_consult{halo-helpdesk}
                                both register speak_to_echo for voice opt-in.
```

To start both:

```
systemctl --user enable --now halo-agent@halo-helpdesk
systemctl --user enable --now halo-agent@halo-coder
```

They survive reboot. They consult each other autonomously. Voice replies
flow through the echo daemon when explicitly requested.
