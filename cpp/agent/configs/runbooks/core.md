# 1bit install core — runbook

## What this installs

The `core` component is the unified operator surface: `1bit` CLI,
`1bit-helm` desktop, and the on-box landing page on `:8190`. It also
chains `voice`, `echo`, `mcp`, and `npu` deps so a fresh box gets the
full local stack from one command.

`lemond` (the inference server on :8180) lives **out of tree** at
`/home/bcloud/repos/lemonade/`. `1bit install core` does NOT manage it.

## Prereqs

- CachyOS / Arch with `pacman -S base-devel cmake ninja`
- ROCm 7.x or TheRock dist tree at `~/therock/build/dist/rocm`
- 8 GB free under `${HOME}/.local/`
- `lemond` already installed and `1bit-halo-lemonade.service` running

## Install

```sh
1bit install core
```

This builds three CMake targets under `cpp/build/strix/` and installs
each binary into `${HOME}/.local/bin/`.

## Common errors

- **`cmake: command not found`** — install via `pacman -S cmake ninja`.
- **`fatal error: hip/hip_runtime.h`** — ROCm headers missing. Either
  `pacman -S rocm-hip-sdk` or build TheRock first.
- **`cannot find -lonebit_core`** — stale build dir; run `rm -rf
  cpp/build/strix && cmake --preset release-strix` and retry.
- **`/v1/models` 503 in `--check`** — lemond not running. Start it:
  `systemctl --user start 1bit-halo-lemonade.service`.

## Logs

- Build: `cpp/build/strix/CMakeFiles/CMakeOutput.log`
- Landing service: `journalctl --user -u strix-landing.service -f`
- lemond: `journalctl --user -u 1bit-halo-lemonade.service -f`

## Rollback

```sh
1bit install --uninstall core
# or manually:
systemctl --user stop strix-landing.service
rm -f ${HOME}/.local/bin/{1bit,1bit-helm,1bit-landing}
```

The Btrfs snapper snapshot from before the install is the safety net
for any deeper damage; rollback with `snapper rollback <id>` and
reboot.
