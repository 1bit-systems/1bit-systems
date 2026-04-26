#!/usr/bin/env bun
// discord-shim.ts — minimal gateway sidecar for halo-agent.
//
// Why a sidecar: our hand-rolled OpenSSL/RFC6455 client in
// cpp/agent/src/discord_ws.cpp gets through HELLO + IDENTIFY but
// Cloudflare on gateway.discord.gg severs the TLS stream
// post-IDENTIFY (SSL_read=0, EPIPE) for reasons that didn't surface
// in five rounds of frame/UA/ALPN/key-order tracing. Python's
// websockets lib identifies the same token cleanly. This bun
// sidecar uses Node's native WebSocket — known-good, ~30 LOC, runs
// as a child of halo-agent (one process per instance, dies with
// the parent).
//
// Wire (stdio JSONL, parent ↔ child):
//
//   parent → child  (stdin):
//     { "op": "send", "channel": "<id>", "text": "..." }
//
//   child → parent (stdout):
//     { "op": "ready", "user": { "id": "...", "username": "..." } }
//     { "op": "message", "channel": "<id>", "user": "<id>",
//       "username": "...", "guild": "<id|null>", "text": "..." }
//     { "op": "error", "msg": "..." }
//
// Env: DISCORD_TOKEN (mandatory), DISCORD_INTENTS (optional, default 37377).
// Exit 1 on token missing, 2 on auth fail, 0 on graceful close.

const TOKEN   = process.env.DISCORD_TOKEN || "";
const INTENTS = parseInt(process.env.DISCORD_INTENTS || "37377", 10);
const GW_URL  = "wss://gateway.discord.gg/?v=10&encoding=json";
const REST    = "https://discord.com/api/v10";

if (!TOKEN) { process.stderr.write("[shim] DISCORD_TOKEN missing\n"); process.exit(1); }

function emit(obj: unknown) { process.stdout.write(JSON.stringify(obj) + "\n"); }
function err(msg: string)   { emit({ op: "error", msg }); }

async function postMessage(channel: string, text: string) {
  const r = await fetch(`${REST}/channels/${channel}/messages`, {
    method: "POST",
    headers: { "Authorization": `Bot ${TOKEN}`, "Content-Type": "application/json" },
    body: JSON.stringify({ content: text }),
  });
  if (!r.ok) err(`send ${r.status}: ${await r.text()}`);
}

async function main() {
  const ws = new WebSocket(GW_URL);
  let heartbeatInterval = 0;
  let lastSeq: number | null = null;
  let hbTimer: ReturnType<typeof setInterval> | null = null;
  let selfId = "";

  ws.onopen = () => process.stderr.write("[shim] gateway open\n");

  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data as string);
    if (m.s !== null && m.s !== undefined) lastSeq = m.s;
    switch (m.op) {
      case 10: { // HELLO
        heartbeatInterval = m.d.heartbeat_interval;
        ws.send(JSON.stringify({
          op: 2,
          d: {
            token: TOKEN, intents: INTENTS,
            properties: { os: "linux", browser: "halo-agent", device: "halo-agent" },
          },
        }));
        hbTimer = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ op: 1, d: lastSeq }));
          }
        }, heartbeatInterval);
        break;
      }
      case 0: { // dispatch
        if (m.t === "READY") {
          selfId = m.d.user.id;
          emit({ op: "ready", user: m.d.user });
        } else if (m.t === "MESSAGE_CREATE") {
          const d = m.d;
          if (d.author?.bot) return;          // ignore bots / self
          // DM (no guild_id) or guild @-mention.
          const isDM = !d.guild_id;
          const mentioned = (d.mentions || []).some((u: any) => u.id === selfId);
          if (!isDM && !mentioned) return;
          emit({
            op:       "message",
            channel:  d.channel_id,
            user:     d.author.id,
            username: d.author.username,
            guild:    d.guild_id || null,
            text:     d.content || "",
          });
        }
        break;
      }
      case 9: { // invalid session
        err(`invalid session, resumable=${m.d}`);
        ws.close();
        break;
      }
      case 11: break; // heartbeat ack
      default: break;
    }
  };

  ws.onerror = (e: any) => err(`ws error: ${e?.message || e}`);
  ws.onclose = (e) => {
    if (hbTimer) clearInterval(hbTimer);
    process.stderr.write(`[shim] gateway closed code=${e.code} reason=${e.reason}\n`);
    process.exit(e.code === 4004 ? 2 : 0);   // 4004 = bad token
  };

  // Stdin: line-delimited JSON commands from the C++ adapter.
  const reader = Bun.stdin.stream().getReader();
  let buf = "";
  const dec = new TextDecoder();
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value);
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      try {
        const cmd = JSON.parse(line);
        if (cmd.op === "send" && cmd.channel && typeof cmd.text === "string") {
          await postMessage(cmd.channel, cmd.text);
        }
      } catch (e: any) {
        err(`bad cmd: ${e.message}`);
      }
    }
  }
}

main().catch((e) => { err(`fatal: ${e?.message || e}`); process.exit(3); });
