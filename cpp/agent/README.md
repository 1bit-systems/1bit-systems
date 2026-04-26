# halo-agent

Autonomous specialist daemon. One process per Discord/Telegram persona,
backed by lemond on `:8180`, zero Python at runtime.

## Build

From the workspace root:

```bash
cmake --preset release-strix
cmake --build --preset release-strix --target halo-agent
ctest --preset release-strix -R onebit_agent --output-on-failure
```

## Install

```bash
1bit install agent
```

This drops the `halo-agent` binary at `~/.local/bin/halo-agent`, the
templated systemd unit at `~/.config/systemd/user/halo-agent@.service`,
and creates `~/.config/halo-agents/` + `~/.local/share/halo-agents/`.

## Configure

Copy an example config into place and edit it:

```bash
mkdir -p ~/.config/halo-agents
cp cpp/agent/configs/halo-helpdesk.toml.example \
   ~/.config/halo-agents/halo-helpdesk.toml
$EDITOR ~/.config/halo-agents/halo-helpdesk.toml
```

The `agent.name` field is the systemd instance handle and must match
the config filename (without `.toml`). Tokens come from environment
variables (`DISCORD_BOT_TOKEN`, `TELEGRAM_BOT_TOKEN`, `GH_TOKEN`)
in preference to inline values; never commit tokens.

## Enable

```bash
systemctl --user enable --now halo-agent@halo-helpdesk
```

The unit is `Type=exec`, restarts on crash with a 5 s backoff, and
appends stdout + stderr to `~/.local/share/halo-agents/<instance>.log`.

## Watch logs

```bash
tail -F ~/.local/share/halo-agents/halo-helpdesk.log
# or
journalctl --user -u halo-agent@halo-helpdesk -f
```

## Stop / disable

```bash
systemctl --user disable --now halo-agent@halo-helpdesk
```

## Configs in this directory

- `halo-helpdesk.toml.example` — install-support persona; can call
  `repo_search`, `bench_lookup`, `install_runbook`. `gh_issue_create`
  is disabled by default and must be opted into per-deployment.
- `halo-echo.toml.example` — read-only community chatter persona;
  smaller toolbox, no write surface.

Spawn additional personas by copying either example, renaming, and
launching another `halo-agent@<name>` instance. Each instance gets its
own SQLite memory file under `~/.local/share/halo-agents/<name>/`.
