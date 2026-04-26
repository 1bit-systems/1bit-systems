# 1bit install echo — runbook

## What this installs

`1bit-echo` — persistent WebSocket voice server on `127.0.0.1:8085`.
Wraps `1bit-voice` and ships Opus frames to browser clients via WS.

Runs as a `--user` systemd unit (`strix-echo.service`).

## Prereqs

- `voice` component installed (echo depends on `1bit-voice`)
- Port `:8085` free (`ss -tlnp | grep 8085` returns nothing)

## Install

```sh
1bit install echo
```

Builds `cpp/build/strix/echo/1bit-echo`, installs to
`${HOME}/.local/bin/1bit-echo`, then enables + starts
`strix-echo.service`.

## Common errors

- **`bind: address already in use`** — something else owns `:8085`.
  Either kill it or change the bind via
  `~/.config/1bit-echo/echo.toml`.
- **WS handshake 400** — likely a browser sending a stale Origin
  header. Check `journalctl --user -u strix-echo.service` for
  `rejected origin: ...` and adjust the allowlist.
- **No audio in browser** — verify the browser actually got the Opus
  codec; Chrome on Linux occasionally needs `chrome://flags/#enable-webrtc-allow-input-volume-adjustment` toggled.

## Logs

```sh
journalctl --user -u strix-echo.service -f
```

## Rollback

```sh
systemctl --user stop strix-echo.service
systemctl --user disable strix-echo.service
rm -f ${HOME}/.local/bin/1bit-echo
rm -f ${HOME}/.config/systemd/user/strix-echo.service
systemctl --user daemon-reload
```
