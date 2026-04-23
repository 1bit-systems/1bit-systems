# Discord setup — canonical

This directory is the source of truth for the 1bit.systems Discord server
state as far as our code is concerned. Goal: a new mini-PC with a fresh
clone can restore the whole Discord integration in one command.

## Files

| File | What |
|------|------|
| `config.toml`   | Non-secret state (channel ids, tag set, bot roles, permissions). Tracked in git. |
| `restore.sh`    | Idempotent restore — reads `config.toml` + secrets env, writes systemd drop-in, runs `1bit-helpdesk-setup`, restarts the unit. |
| `README.md`     | This file. |

## Secrets

Nothing secret ships here. Bot tokens live at:

```
~/.config/1bit/discord-secrets.env
```

Format (fish/bash compatible; one export per line):

```
DISCORD_BOT_TOKEN="<halo gateway token>"
ECHO_BOT_TOKEN="<echo posting token>"
```

That file is chmod 600 and NEVER committed. If it's missing, `restore.sh`
bails with a helpful error.

## What gets restored

1. `~/.config/systemd/user/strix-watch-discord.service.d/token.conf` is
   (re)written from the canonical template + secrets env. `HALO_HELP_DESK_CHANNEL_ID`,
   `HALO_DISCORD_CHANNELS`, and both bot tokens are set here.
2. `1bit-helpdesk-setup` runs against the `[channels.help_desk.id]` in
   `config.toml`, applying the canonical tag set (bug / feature /
   question / pending / resolved / escalated) + topic + slowmode.
3. `strix-watch-discord.service` is reloaded and restarted.

## Restoring from scratch

```bash
# 1. Install the bots
cargo install --path crates/1bit-agents --bin 1bit-watch-discord --bin 1bit-helpdesk-setup

# 2. Drop your secrets in place (one-time)
mkdir -p ~/.config/1bit && chmod 700 ~/.config/1bit
$EDITOR ~/.config/1bit/discord-secrets.env
chmod 600 ~/.config/1bit/discord-secrets.env

# 3. Run the restore
./strixhalo/discord/restore.sh
```

After `restore.sh` succeeds:

* `systemctl --user status strix-watch-discord.service` should report active
* The help-desk forum channel has the canonical tag set applied
* A test @mention in any watched channel should route a forum post with
  the right classification tag

## Hand-converting a text channel to a forum

Discord's API can't convert channel types on its own. If the current
`help_desk` is a text channel and you want a forum:

1. Discord UI → channel settings → overview → change channel type → Forum
   (or delete + recreate as Forum and update the id in `config.toml`)
2. Re-run `./strixhalo/discord/restore.sh` — the setup tool detects the
   new shape and applies the tag set.

The bot itself already handles both shapes at runtime — see
`HelpDeskMeta::is_forum` in `crates/1bit-agents/bin/1bit-watch-discord.rs`.

## Adding a new help-desk context (e.g. mc-help-desk)

1. Create the forum channel in Discord, copy its id.
2. Uncomment + fill the `mc_help_desk` block in `config.toml`.
3. Extend `restore.sh` to run `1bit-helpdesk-setup` against the new id.
4. Land the routing branch in the bot (look for the
   `help_desk_channel_id` option field — the current code takes a single
   id; a follow-up will accept a map).

## Backing up to the Pi archive

`strixhalo/discord/` is part of the workspace rsync job to
`100.64.0.4:/ZFSPool/archive/ryzen/`. Nothing extra to do — git has the
non-secret state, the Pi archive has everything else on the box.
