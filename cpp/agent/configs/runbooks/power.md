# 1bit install power — runbook

## What this installs

`1bit power` — Ryzen APU power-profile CLI. Wraps FlyGoat/ryzenadj to
flip between perf, balanced, and quiet profiles on Strix Halo.
Replaces the closed-source RyzenZPilot tray app on Linux.

Subcommand of the main `1bit` CLI; `core` ships it. The `power`
component name exists as an alias for documentation symmetry.

## Prereqs

- `core` installed
- `ryzenadj` from AUR: `yay -S ryzenadj`
- Passwordless sudo for `ryzenadj` (it pokes MSRs):
  `echo "$USER ALL=(root) NOPASSWD: /usr/bin/ryzenadj" | sudo tee /etc/sudoers.d/99-ryzenadj`

## Install

```sh
1bit install power
```

This is a no-op build (handled by `core`) — it exists to surface
prereq messages.

## Use

```sh
1bit power status              # show current TDP / fan / governor
1bit power set perf            # boost
1bit power set balanced
1bit power set quiet           # silent-closet target
```

## Common errors

- **`ryzenadj: command not found`** — `yay -S ryzenadj`.
- **`sudo: password required`** — sudoers rule above wasn't applied.
  Re-run after `sudo -k` to flush the cached prompt state.
- **`ryzenadj: failed to open /dev/cpu/0/msr`** — `sudo modprobe msr`.
  Make it persistent via `echo msr | sudo tee /etc/modules-load.d/msr.conf`.
- **OPTC CRTC hang under load** — known gfx1151 kernel bug. Mitigation
  applied via `amdgpu.dcdebugmask=0x410` (`project_optc_tier3_applied`).
  If you see `REG_WAIT timeout` in dmesg, escalate.

## Logs

stderr only. Run with `--verbose` for the raw ryzenadj output.

## Rollback

```sh
sudo rm -f /etc/sudoers.d/99-ryzenadj
yay -R ryzenadj
```

The `1bit` binary itself uninstalls with `core`.
