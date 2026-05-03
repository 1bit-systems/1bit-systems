# halo-agent fleet

Autonomous chat agents on your own hardware. They boot when the box
boots, answer DMs on Discord/Telegram/web, and use `1bit-proxy :13306`
as their OpenAI-compatible brain. The proxy defaults to Lemonade on
`:13305` and routes targeted FLM models to FastFlowLM on `:52625`.

The fleet exists because answering the same five support questions
by hand is not a good use of anyone's time. So we made a thing that
does it for us.

## The pieces

A "halo-agent" is one binary running one config file. Configs live
in `~/.config/halo-agents/`; one file per agent.

Each running agent has three replaceable parts:

- **Adapter** — talks to the surface (Discord DMs, Telegram DMs, an
  HTTP endpoint, or your terminal). One adapter per agent.
- **Brain** — POSTs OpenAI-compatible `/v1/chat/completions` to
  `1bit-proxy` on `127.0.0.1:13306`. Same endpoint used by Open WebUI
  and generic SDK clients.
- **Tools** — a small list of named operations the brain can call:
  `repo_search`, `url_fetch`, `docs_lookup`, optionally
  `gh_issue_create` and friends. Each tool is gated by sensible
  defaults (read-only, allowlisted, confirm-required where it
  matters).

A SQLite file under `/var/lib/1bit-agent/` keeps the conversation
log so the agent survives reboots.

## What ships

Out of the box you get one example agent — `halo-helpdesk` — and
the machinery to spin up more. Examples:

- **halo-helpdesk** — answers questions about 1bit-systems on your
  Discord. Reads the wiki, searches the repo, links to docs.
- **halo-echo** — voice-side companion (paired with our STT/TTS
  loop). Same agent core, different adapter.
- **halo-yours** — copy a TOML, change the system prompt, point it
  at a different Discord token, done.

The plumbing is one binary; the personality is the system prompt
and the tool list.

## How a turn works

```
DM arrives  ──►  agent persists it to sqlite
                  │
                  ▼
            history (last 32) + system prompt + tool list
                  │
                  ▼
            1bit-proxy ──►  text  OR  tool_calls
                              │              │
                              │              ▼
                              │         run tools, persist results
                              │              │
                              │              └─► back to lemond
                              ▼
                         agent sends reply on the surface
```

Capped at 5 brain↔tool round trips per turn so a confused brain
can't run away. SQLite is the source of truth — pull the plug
mid-turn and the conversation picks up where it left off when the
box boots.

## Boot survival

The fleet runs under `systemd --user` with one template unit
(`halo-agent@.service`) and one instance per agent
(`halo-agent@halo-helpdesk.service`, etc.). `Restart=always` and
`After=1bit-proxy.service` mean the agent comes back after
every reboot, after every adapter blip, after every brain restart —
without the operator doing anything.

Linger your user
(`loginctl enable-linger $USER`) and the fleet starts before you
log in.

## What this isn't

- **Not a Claude wrapper.** The brain is the local 1bit endpoint. No cloud calls
  on the runtime path.
- **Not a multi-agent framework.** One agent answers one surface.
  If you want two surfaces, run two agents.
- **Not RAG.** The agent has tools, not embeddings. If you want
  search, that's `repo_search`.
- **Not human-in-the-loop.** That's the entire point. The agent
  decides; mutating tools require an explicit "yes" from the user
  before they fire (and even then, scoped narrowly).
- **Not self-modifying.** The binary lives in `/usr/bin`; the agent
  runs as you and can't write it. Config changes need a restart.
  No emergent behaviors.

## What you do

1. Install: `1bit install agent`.
2. Mint a Discord bot token (or Telegram, or skip and run the HTTP
   adapter).
3. Copy the example config to `~/.config/halo-agents/<name>.toml`,
   edit the system prompt and the tool list.
4. Hand the token to systemd via `systemd-creds encrypt`.
5. `systemctl --user enable --now halo-agent@<name>`.
6. Tail the journal: `journalctl --user -u halo-agent@<name> -f`.

Full operator playbook in `cpp/agent/RUNBOOK.md`. Threat model and
gating defaults in `cpp/agent/SECURITY.md`. Architecture deep-dive
in `cpp/agent/ARCHITECTURE.md`.

## Defaults you should know about

- **Allowlist mode** is on for Discord and Telegram out of the box.
  Pair the operator first, anyone else gets a silent drop.
- **5 messages / minute / user** rate limit. Above that the agent
  goes quiet for a minute.
- **Mutating tools** (`gh_issue_create`, `fs_write`) require an
  explicit `confirm: "YES"` argument that the brain can only set
  after the user agreed in plain language. The registry enforces
  this; the brain can't bypass it.
- **System prompts are salted** at boot with a random hex string,
  so "ignore previous instructions" doesn't.
- **Logs to journal**, not to the cloud. SQLite history is local.
  LUKS the disk if that matters to you.

## Where it fits

The fleet is the public-facing surface of a 1bit-systems box. The
brain (`1bit-proxy` -> Lemonade/FastFlowLM), the tools (`mcp`,
`retrieval`), and the current service stack all sit underneath. The agent
is the part of the stack that knows how to take a Discord message
and turn it into "your question, answered, by the box, while you
were asleep."

That's the whole product.
