# 1bit install helm — runbook

## What this installs

`1bit-helm` — the Qt6 desktop client. Service controls (start/stop
the local lemond + STT + TTS), model picker, log tail, and a chat
pane that talks to lemond on `:8180`.

`helm` is bundled in `core` since 2026-04-20; the standalone component
exists as a back-compat alias.

## Prereqs

- Qt6 base + declarative + websockets:
  `pacman -S qt6-base qt6-declarative qt6-websockets`
- Wayland session (X11 fallback works but font scaling is wrong on
  HiDPI)
- `core` installed first

## Install

```sh
1bit install helm
```

Or — since helm is bundled in core — `1bit install core` already
covers it.

## Common errors

- **`could not find Qt6Config.cmake`** — install qt6-base. CachyOS
  bare install does not include it.
- **Window opens, then closes immediately, no error** — usually a
  Wayland protocol mismatch. Run from a terminal with
  `QT_LOGGING_RULES='*=true' 1bit-helm` to surface the actual
  diagnostic.
- **Tray icon missing on KDE** — install `xdg-desktop-portal-kde`
  and restart the Plasma session.

## Logs

`1bit-helm` writes to stderr. Run from a terminal during debug;
otherwise nothing is captured. Service controls inside the app pipe
to `journalctl --user -u <unit>` directly.

## Rollback

```sh
rm -f ${HOME}/.local/bin/1bit-helm
rm -rf ${HOME}/.config/1bit-helm/
```
