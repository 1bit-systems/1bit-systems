# 1bit-agent — operator runbook

> "There is no spoon."

This is the playbook for running the autonomous fleet on a Strix
Halo box. Every command here is the one you actually type. If the
agent isn't doing what it's supposed to, the answer is in
`journalctl --user -u halo-agent@<name> -f` and 90% of the time it's
the brain at `lemond:8180`, not the agent.

See `ARCHITECTURE.md` for what each piece is. See `SECURITY.md`
before you flip an allowlist or mint a token with broad scopes.

---

## TL;DR

```bash
# bootstrap halo-helpdesk on Discord
1bit install agent
cp /etc/1bit-agent/halo-helpdesk.toml.example \
   ~/.config/halo-agents/halo-helpdesk.toml
$EDITOR ~/.config/halo-agents/halo-helpdesk.toml         # set adapter.token
systemctl --user enable --now halo-agent@halo-helpdesk
journalctl --user -u halo-agent@halo-helpdesk -f
```

That's the whole thing. Everything below is the long form.

---

## 0. Prereqs

- `lemond` running on `127.0.0.1:8180`. Check with:
  ```bash
  systemctl --user is-active 1bit-halo-lemonade.service
  curl -s http://127.0.0.1:8180/v1/models | head
  ```
  If that's down, fix the brain first. The agent will not start
  emitting useful replies without it.
- `1bit` CLI on PATH. `1bit --version` should print.
- A user-mode systemd session (`loginctl enable-linger $USER` if you
  want agents up before the user logs in — recommended).

---

## 1. Bootstrap a new agent

A "new agent" = a new TOML + a new systemd instance + (optionally) a
new bot token on Discord/Telegram.

### 1.1 Mint the surface token

**Discord**:
1. <https://discord.com/developers/applications> → New Application
   → name it (`halo-helpdesk`).
2. Bot tab → Reset Token → copy. Don't paste it anywhere yet.
3. OAuth2 → URL Generator → scopes: `bot`. Permissions: `Send
   Messages`, `Read Message History`. **Do not** grant `Administrator`
   or `Manage Server`.
4. Open the generated URL → invite bot to your server (or skip and
   rely on DMs only — see SECURITY.md).
5. Bot tab → Privileged Gateway Intents → enable `MESSAGE
   CONTENT INTENT`. (Required for DM payloads.)

**Telegram**:
1. DM `@BotFather` → `/newbot` → pick name + username (`halo_echo_bot`).
2. Copy the API token from BotFather's reply.
3. (Optional) `/setjoingroups` → Disable. DM-only by default.
4. (Optional) `/setprivacy` → Enable, so the bot only sees DMs and
   commands by default.

**Web (HTTP adapter)**:
- No external token. The adapter binds on
  `[adapter].bind_host:bind_port` and authenticates incoming
  requests via a shared bearer in `[adapter].token`. Generate one:
  `openssl rand -base64 32`.

**stdin** (for local testing):
- No token. Run the binary in the foreground, type at it.

### 1.2 Write the config TOML

Path convention: `~/.config/halo-agents/<name>.toml`. Schema lives
in `cpp/agent/include/onebit/agent/config.hpp`. Start from the
example in `cpp/agent/configs/`:

```bash
mkdir -p ~/.config/halo-agents
cp /etc/1bit-agent/halo-helpdesk.toml.example \
   ~/.config/halo-agents/halo-helpdesk.toml
$EDITOR ~/.config/halo-agents/halo-helpdesk.toml
```

Fields you actually edit:

```toml
[agent]
name              = "halo-helpdesk"
brain_url         = "http://127.0.0.1:8180"
system_prompt     = """\
You are halo-helpdesk. You answer questions about 1bit-systems on
the operator's behalf — be concise, link to the wiki when relevant,
never claim to be human, never run destructive tools.
"""
model             = "halo-1.58b"      # let lemond resolve the recipe
max_history       = 32
max_tool_iters    = 5
request_timeout_ms = 60000
stream            = true
temperature       = 0.2

[adapter]
kind     = "discord"                  # discord | telegram | http | stdin
token    = "${ENV:DISCORD_TOKEN}"     # never inline the literal

[memory]
sqlite_path   = "/var/lib/1bit-agent/halo-helpdesk.db"
keep_messages = 0                     # 0 = no auto-trim

[tools]
enabled = ["repo_search", "url_fetch", "docs_lookup"]
```

`${ENV:NAME}` is expanded at config load. The token comes from a
systemd credential, not the shell — see 1.3.

### 1.3 Hand the token to systemd

Don't put the token in the TOML file. Don't put it in your shell
rc. Use `systemd-creds`:

```bash
systemd-creds encrypt --name=DISCORD_TOKEN - \
  /etc/credstore.encrypted/halo-helpdesk-discord.cred <<<"$TOKEN"
```

Then in the unit (already in the template — see 1.4) we
`LoadCredentialEncrypted=DISCORD_TOKEN:halo-helpdesk-discord.cred`
and the agent reads it via `${CREDENTIALS_DIRECTORY}/DISCORD_TOKEN`,
which the config loader converts to the in-memory string.

If you're prototyping and don't want to deal with `systemd-creds`,
fall back to a 0600 file under `/etc/1bit-agent/secrets/<name>.env`
and `EnvironmentFile=` it from the unit. Rotate quarterly.

### 1.4 Enable the systemd unit

The template lives in the package at
`/usr/lib/systemd/user/halo-agent@.service`. You don't edit that;
you instantiate it:

```bash
systemctl --user daemon-reload
systemctl --user enable --now halo-agent@halo-helpdesk
```

Template (read-only reference — do not patch in place; drop overrides
under `~/.config/systemd/user/halo-agent@halo-helpdesk.service.d/`):

```ini
[Unit]
Description=1bit autonomous agent — %i
After=network-online.target 1bit-halo-lemonade.service
Wants=1bit-halo-lemonade.service

[Service]
Type=simple
ExecStart=/usr/bin/halo-agent --config %h/.config/halo-agents/%i.toml
Restart=always
RestartSec=5
RestartPreventExitStatus=0
LoadCredentialEncrypted=DISCORD_TOKEN:halo-%i-discord.cred
StateDirectory=1bit-agent
RuntimeDirectory=1bit-agent
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=true

[Install]
WantedBy=default.target
```

`%i` = the instance name (`halo-helpdesk`). One unit instance per
agent. They do not share state.

### 1.5 Tail the log

```bash
journalctl --user -u halo-agent@halo-helpdesk -f
```

What healthy startup looks like:

```
halo-agent[12345]: cfg loaded: halo-helpdesk (discord)
halo-agent[12345]: brain ok: http://127.0.0.1:8180 model=halo-1.58b
halo-agent[12345]: memory ok: /var/lib/1bit-agent/halo-helpdesk.db sqlite=3.45.1
halo-agent[12345]: tools ok: repo_search, url_fetch, docs_lookup
halo-agent[12345]: adapter ok: discord (logged in as halo-helpdesk#1234)
halo-agent[12345]: loop running
```

If you see any of these, fix and restart:

| Line                                  | Meaning                          | Fix                                         |
|---------------------------------------|----------------------------------|---------------------------------------------|
| `brain ERR: connection refused`       | `lemond` down                    | `systemctl --user start 1bit-halo-lemonade` |
| `adapter ERR: 401`                    | bad token                        | rotate token, re-encrypt cred, restart      |
| `cfg ERR: unknown adapter kind`       | typo in TOML                     | fix `[adapter].kind`                        |
| `memory ERR: SQLITE_CANTOPEN`         | StateDirectory missing           | `mkdir -p /var/lib/1bit-agent` (root)       |

---

## 2. Routine ops

### 2.1 Status / health

```bash
systemctl --user status 'halo-agent@*'
journalctl --user -u halo-agent@halo-helpdesk -n 200 --no-pager
```

Per-instance counters land in the journal as periodic `loop stats:`
lines (`messages_in`, `messages_out`, `tool_calls`, `brain_calls`,
`adapter_timeouts`, `hard_errors`). Watch `hard_errors`; that's the
only counter that should always be 0.

### 2.2 Tail one channel

We don't ship a separate inspector — read SQLite directly:

```bash
sqlite3 /var/lib/1bit-agent/halo-helpdesk.db \
  "SELECT created_at, role, substr(content,1,80) \
   FROM messages WHERE channel='<chan>' \
   ORDER BY id DESC LIMIT 20"
```

Channel ids are surface-specific (Discord channel snowflake, Telegram
chat id, websocket session uuid). Find one with `tail -f` of the
journal first.

### 2.3 Restart / reload

There is no "config reload" signal — restart is the contract.

```bash
systemctl --user restart halo-agent@halo-helpdesk
```

Outstanding turn (if any) is dropped; the user message is durable
and the brain re-answers on next inbound.

### 2.4 Stop / disable

```bash
systemctl --user disable --now halo-agent@halo-helpdesk
```

`disable` revokes the boot symlink; `--now` also stops the running
instance. SQLite is left in place so re-enabling resumes history.

### 2.5 Wipe an agent's memory

```bash
systemctl --user stop halo-agent@halo-helpdesk
rm /var/lib/1bit-agent/halo-helpdesk.db
systemctl --user start halo-agent@halo-helpdesk
```

The agent recreates the schema on first message. This is the only
"forget everything" lever; we deliberately don't have a soft variant.

---

## 3. Add a tool

Tools are sibling-agent territory — see `cpp/agent/src/tools/` and
the `IToolRegistry` interface in `tools.hpp`. The 30-second version:

1. Author `cpp/agent/src/tools/<name>.cpp` exposing a factory
   `register_<name>(ToolRegistry&)`.
2. The factory adds an OpenAI `tools[]` schema record + a callback
   that takes a `ToolCall` and returns a `ToolResult`.
3. Add the tool to the binary's registry construction in
   `cpp/agent/src/main.cpp` (or, for sibling agents, their own
   `main.cpp`).
4. Add the name to `[tools].enabled` in the agent's TOML.
5. Restart.

Operator-side checks before you ship a tool:

- Does it touch the network? Add a per-call rate limit.
- Does it write anything (filesystem, GitHub, an API)? Set
  `requires_confirm=true` so the loop demands a `confirm=YES` arg
  the brain has to thread from the user.
- Does it read secrets? Pull them from `${CREDENTIALS_DIRECTORY}`,
  not from `getenv()`.

`SECURITY.md` has the full gating policy.

---

## 4. Swap the brain model

`lemond` resolves `model` to a recipe at load time. The agent never
ships its own weights.

```bash
$EDITOR ~/.config/halo-agents/halo-helpdesk.toml   # change [agent].model
systemctl --user restart halo-agent@halo-helpdesk
```

To swap the *brain endpoint* (e.g. point at a remote `lemond` for
testing), change `[agent].brain_url` and restart. Keep it on
loopback in production — `lemond` doesn't auth.

To A/B two models, run two agents (`halo-helpdesk-a`,
`halo-helpdesk-b`) on two surfaces. We don't multiplex models inside
one instance.

---

## 5. Swap the adapter (e.g. Discord → Telegram on the same agent identity)

Don't. Different surfaces have different `channel` namespaces; if
you point the same SQLite at a new transport, history will collide
in nonsense ways.

What you do instead: make a new agent (`halo-helpdesk-tg`) with its
own DB and its own TOML. Same `name` is fine if you accept that
they keep separate memory. If you want them to share knowledge, do
it through a tool (the tool reads both databases, or a shared
`facts` row that both update).

---

## 6. Revoke / lock down DMs

The agent answers anyone by default — that's the whole point. Locking
down has two levers:

### 6.1 Allowlist mode

```bash
mkdir -p ~/.claude/channels/discord
cat >~/.claude/channels/discord/access.json <<JSON
{
  "policy": "allowlist",
  "allowlist": ["123456789012345678", "987654321098765432"]
}
JSON
```

`policy: "allowlist"` means only listed `user_id`s get answers.
Everyone else is silently dropped (no reply, log line at INFO). The
agent re-reads this file on each inbound; no restart needed.

Other policies:

- `"policy": "open"` — default; anyone may DM.
- `"policy": "denylist"` + `"denylist": [...]` — anyone except.
- `"policy": "operator-only"` — shorthand for allowlist of the
  operator's `user_id` only. Useful as a panic mode.

### 6.2 Revoke a single user

```bash
jq '.allowlist -= ["1234567890"]' ~/.claude/channels/discord/access.json \
  | sponge ~/.claude/channels/discord/access.json
# OR for denylist:
jq '.denylist += ["1234567890"]' ~/.claude/channels/discord/access.json \
  | sponge ~/.claude/channels/discord/access.json
```

If they got abusive, delete their stored history too:

```bash
sqlite3 /var/lib/1bit-agent/halo-helpdesk.db \
  "DELETE FROM messages WHERE user_id='1234567890'"
```

### 6.3 Pause everything fast

```bash
systemctl --user stop 'halo-agent@*'
```

Adapters log out cleanly; bots show offline. SQLite is intact.

---

## 7. Rotate a token

```bash
systemctl --user stop halo-agent@halo-helpdesk
# (re-mint via Discord developer portal / BotFather)
systemd-creds encrypt --name=DISCORD_TOKEN - \
  /etc/credstore.encrypted/halo-helpdesk-discord.cred <<<"$NEWTOKEN"
systemctl --user start halo-agent@halo-helpdesk
```

Old token stops working the moment Discord/Telegram processes the
revocation. Don't reuse credentials across instances.

---

## 8. Alarm signals — what to actually watch

Watch these in the journal. Set up `journalctl --user -p warning`
for a quick eyeball:

| Signal                                   | What it means                                   | Action                                       |
|------------------------------------------|-------------------------------------------------|----------------------------------------------|
| `hard_errors > 0`                        | Loop returned `AgentError` from a turn          | tail journal, look upstream of the WARN line |
| Repeated `brain ERR: timeout`            | `lemond` is slow or stuck                       | check `1bit-halo-lemonade` + GPU temps       |
| `adapter ERR: gateway disconnect` >5/min | Discord rate-limited or token revoked           | back off; verify token; check Discord status |
| `tool ERR: gh_issue_create rate-limited` | Brain is over-eager on a gated tool             | tighten `requires_confirm` policy            |
| `loop WARN: max_tool_iters exceeded`     | Brain stuck in a tool loop                      | inspect last 10 turns; consider lowering cap |
| Spike in `messages_in` from one user     | Possible scripted abuse                         | rate-limit kicks in automatically; consider denylist |
| `sqlite ERR: SQLITE_FULL`                | Disk full                                       | free space; agent restarts itself            |
| Process restart count >5/min             | Crash loop                                      | `systemctl --user status` for last exit code |

We do not page externally. The whole fleet is "log to journal, let
the operator see it when they look." If you want pager integration,
that's a follow-on.

---

## 9. Backups

The TOML and the SQLite are the entire state.

```bash
tar czf halo-agents-$(date +%F).tar.gz \
  ~/.config/halo-agents/ \
  /var/lib/1bit-agent/
```

Restore: stop all instances, untar in place, start. Tokens are not in
that tarball (they're in `/etc/credstore.encrypted/`); back those up
separately if you ever rotate the box.

---

## 10. Cheat sheet

```bash
# install
1bit install agent

# new agent
cp /etc/1bit-agent/<name>.toml.example ~/.config/halo-agents/<name>.toml
$EDITOR ~/.config/halo-agents/<name>.toml
systemd-creds encrypt --name=DISCORD_TOKEN - \
  /etc/credstore.encrypted/<name>-discord.cred <<<"$TOKEN"
systemctl --user enable --now halo-agent@<name>

# watch
journalctl --user -u halo-agent@<name> -f
systemctl --user status 'halo-agent@*'

# revoke
jq '.allowlist -= ["<id>"]' ~/.claude/channels/discord/access.json | sponge ~/.claude/channels/discord/access.json

# swap model
$EDITOR ~/.config/halo-agents/<name>.toml   # [agent].model
systemctl --user restart halo-agent@<name>

# panic
systemctl --user stop 'halo-agent@*'
```

That's it. The fleet runs itself; you only show up when the journal
yells.
