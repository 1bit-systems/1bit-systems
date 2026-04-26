# 1bit install voice — runbook

## What this installs

`1bit-voice` — sentence-boundary TTS streaming CLI. It reads SSE from
the local LLM, splits at sentence boundaries, ships each chunk to the
TTS engine, and pipes the resulting Opus frames to stdout / a file.

Binary only. **No systemd unit.** Invoked on demand by `1bit say`,
`1bit-echo`, and ad-hoc shell pipelines.

## Prereqs

- `core` already installed (the CLI provides `1bit say`)
- A TTS backend reachable on `:8095` (the `tts-engine` component) or
  `:8083` (whisper-kokoro lane)
- `libopus` from `pacman -S opus`

## Install

```sh
1bit install voice
```

Builds `cpp/build/strix/voice/1bit-voice` and installs to
`${HOME}/.local/bin/1bit-voice`.

## Common errors

- **`opus.h: No such file or directory`** — `pacman -S opus`.
- **`connect: connection refused :8095`** — TTS server not up. Either
  start `1bit-tts.service` or point at `:8083` via
  `1bit-voice --tts-url http://127.0.0.1:8083`.
- **Audio drops every ~2s** — sentence-splitter regex bug; bump
  `--max-sentence-ms 1500` and file an issue if it still drops.

## Logs

- TTS server: `journalctl --user -u 1bit-tts.service -f`
- voice itself logs to stderr; pipe with `2>>~/.local/share/1bit-agent/voice.log`.

## Rollback

```sh
rm -f ${HOME}/.local/bin/1bit-voice
```

No service to stop; nothing else to clean up.
