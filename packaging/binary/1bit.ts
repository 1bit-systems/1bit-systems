#!/usr/bin/env bun
// 1bit.ts — single-binary control plane for 1bit-systems.
//
// Subcommands (port of scripts/1bit bash):
//   up | down | status | pull <model> | bench | npu [model] | webui [up|down|status]
//
// When invoked as `1bit __proxy__` it runs the embedded reverse proxy in
// the foreground (used internally by `up` to spawn a detached child).

import { spawn, spawnSync, execSync } from "node:child_process";
import { existsSync, openSync, closeSync } from "node:fs";
import { startProxy } from "./proxy.ts";

const CYAN = "\x1b[0;36m";
const GREEN = "\x1b[0;32m";
const YELLOW = "\x1b[1;33m";
const RED = "\x1b[0;31m";
const NC = "\x1b[0m";

const say = (s: string) => console.log(`${CYAN}▸${NC} ${s}`);
const ok = (s: string) => console.log(`${GREEN}✓${NC} ${s}`);
const warn = (s: string) => console.log(`${YELLOW}!${NC} ${s}`);
const die = (s: string): never => {
  console.error(`${RED}✗${NC} ${s}`);
  process.exit(1);
};

const LEMOND_URL = "http://127.0.0.1:13305";
const PROXY_URL = "http://127.0.0.1:13306";
const WEBUI_URL = "http://127.0.0.1:3000";
const LEMOND_LOG = "/tmp/lemond.log";
const FLM_LOG = "/tmp/flm-serve.log";
const PROXY_LOG = "/tmp/1bit-proxy.log";
const WEBUI_LOG = "/tmp/1bit-webui.log";
const DEFAULT_NPU_MODEL = process.env.ONEBIT_NPU_MODEL || "qwen3:1.7b";

// -- helpers --------------------------------------------------------------

function which(cmd: string): string | null {
  const r = spawnSync("sh", ["-c", `command -v ${cmd}`], {
    encoding: "utf8",
  });
  return r.status === 0 ? r.stdout.trim() : null;
}

function pgrepX(name: string): boolean {
  return spawnSync("pgrep", ["-x", name], { stdio: "ignore" }).status === 0;
}
function pgrepF(pat: string): boolean {
  return spawnSync("pgrep", ["-f", pat], { stdio: "ignore" }).status === 0;
}
function pgrepFFirst(pat: string): string {
  const r = spawnSync("pgrep", ["-f", pat], { encoding: "utf8" });
  if (r.status !== 0) return "";
  return r.stdout.split("\n")[0]?.trim() || "";
}
function pkillF(pat: string): boolean {
  return spawnSync("pkill", ["-f", pat], { stdio: "ignore" }).status === 0;
}
function pkillX(name: string): boolean {
  return spawnSync("pkill", ["-x", name], { stdio: "ignore" }).status === 0;
}

async function urlOk(url: string, timeoutMs = 2000): Promise<boolean> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { signal: ctrl.signal });
    return r.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(t);
  }
}

async function urlText(
  url: string,
  timeoutMs = 2000,
): Promise<string | null> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { signal: ctrl.signal });
    if (!r.ok) return null;
    return await r.text();
  } catch {
    return null;
  } finally {
    clearTimeout(t);
  }
}

function flmPort(): string {
  try {
    const out = execSync("flm port", {
      stdio: ["ignore", "pipe", "ignore"],
    }).toString();
    const parts = out.trim().split(/\s+/);
    return parts[parts.length - 1] || "";
  } catch {
    return "";
  }
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

// Spawn a fully detached process with stdio redirected to a log file.
function nohupSpawn(
  cmd: string,
  args: string[],
  logPath: string,
  env?: NodeJS.ProcessEnv,
): void {
  const fd = openSync(logPath, "a");
  const child = spawn(cmd, args, {
    detached: true,
    stdio: ["ignore", fd, fd],
    env: env ? { ...process.env, ...env } : process.env,
  });
  child.unref();
  closeSync(fd);
}

// -- service controls -----------------------------------------------------

async function startLemond(): Promise<void> {
  if (pgrepX("lemond")) {
    ok("lemond already running");
    return;
  }
  say("Starting lemond on :13305");
  nohupSpawn("lemond", [], LEMOND_LOG);
  for (let i = 0; i < 20; i++) {
    await sleep(500);
    if (await urlOk(`${LEMOND_URL}/v1/models`, 1000)) {
      ok(`lemond up at ${LEMOND_URL}`);
      return;
    }
  }
  warn(`lemond didn't respond — see ${LEMOND_LOG}`);
}

async function startFlm(): Promise<void> {
  if (pgrepF("flm.*serve")) {
    ok(`flm already serving (pid ${pgrepFFirst("flm.*serve")})`);
    return;
  }
  say(`Starting flm serve ${DEFAULT_NPU_MODEL}`);
  nohupSpawn("flm", ["serve", DEFAULT_NPU_MODEL], FLM_LOG);
  await sleep(4000);
  const port = flmPort();
  if (port) ok(`flm serving ${DEFAULT_NPU_MODEL} on :${port}`);
  else warn(`flm didn't report a port — see ${FLM_LOG}`);
}

// We start the proxy by re-execing ourselves with the hidden subcommand
// `__proxy__`, fully detached. That avoids needing node anywhere.
async function startProxyService(): Promise<void> {
  if (pgrepF("1bit-proxy")) {
    ok("proxy already running");
    return;
  }
  say(`Starting 1bit-proxy on :13306 (unifies lemond + flm)`);
  // process.execPath is the compiled binary itself.
  nohupSpawn(process.execPath, ["__proxy__"], PROXY_LOG);
  await sleep(1000);
  if (await urlOk(`${PROXY_URL}/health`, 2000)) {
    ok(`proxy up at ${PROXY_URL} — single OpenAI endpoint, both lanes`);
  } else {
    warn(`proxy didn't respond — see ${PROXY_LOG}`);
  }
}

// -- subcommands ----------------------------------------------------------

async function cmdUp(): Promise<void> {
  await startLemond();
  await startFlm();
  await startProxyService();

  say(`Opening browser: ${PROXY_URL}  (1bit.systems home)`);
  if (which("xdg-open")) {
    nohupSpawn("xdg-open", [PROXY_URL], "/dev/null");
  }

  if (which("gaia")) {
    say(`Launching AMD GAIA (point it at ${PROXY_URL}/v1)`);
    nohupSpawn("gaia", [], "/dev/null");
  } else {
    warn(
      `AMD GAIA not installed. Get the .deb from https://amd-gaia.ai and point it at ${PROXY_URL}/v1`,
    );
  }

  console.log();
  ok("All up. Endpoints:");
  console.log(`   Unified (recommended):  ${PROXY_URL}/v1/`);
  console.log(`   Lemonade (GPU only):    ${LEMOND_URL}/v1/`);
  console.log(`   FLM (NPU only):         http://127.0.0.1:${flmPort()}/v1/`);
}

function cmdDown(): void {
  say("Stopping all services");
  if (pkillF("1bit-proxy")) ok("proxy stopped");
  else ok("proxy already stopped");
  if (pkillF("flm.*serve")) ok("flm stopped");
  else ok("flm already stopped");
  if (pkillX("lemond")) ok("lemond stopped");
  else ok("lemond already stopped");
}

async function cmdStatus(): Promise<void> {
  if (await urlOk(`${LEMOND_URL}/v1/models`, 2000))
    ok(`lemond:  up at ${LEMOND_URL}`);
  else warn("lemond:  down");

  if (pgrepF("flm.*serve")) {
    const port = flmPort();
    ok(`flm:     serving on :${port}`);
  } else {
    warn("flm:     not serving");
  }

  const health = await urlText(`${PROXY_URL}/health`, 2000);
  if (health) {
    const m = health.match(/"models":(\d+)/);
    ok(
      `proxy:   up at ${PROXY_URL} (${m ? m[1] : "?"} models unified)`,
    );
  } else {
    warn("proxy:   down");
  }

  console.log();
  // memlock limit: ulimit -l. Use sh.
  try {
    const r = spawnSync("sh", ["-c", "ulimit -l"], { encoding: "utf8" });
    console.log(`memlock limit: ${r.stdout.trim()}`);
  } catch {}
}

function cmdPull(model: string | undefined): void {
  if (!model) die("usage: 1bit pull <model>");
  let onFlm = false;
  try {
    const r = spawnSync("flm", ["list"], { encoding: "utf8" });
    if (r.status === 0) {
      const re = new RegExp(`^\\s*-\\s*${escapeRe(model!)}\\s`, "m");
      onFlm = re.test(r.stdout);
    }
  } catch {}

  if (onFlm) {
    say(`Routing to FLM (NPU): ${model}`);
    spawnSync("flm", ["pull", model!], { stdio: "inherit" });
  } else {
    say(`Routing to lemonade (GPU): ${model}`);
    spawnSync("lemonade", ["pull", model!], { stdio: "inherit" });
  }
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function cmdBench(): void {
  const candidates = [
    "/home/bcloud/Projects/1bit-systems/benchmarks/bench-1bit-pile.sh",
  ];
  let script = "";
  for (const p of candidates) if (existsSync(p)) script = p;
  if (!script) die("bench script not found");
  const r = spawnSync("bash", [script], { stdio: "inherit" });
  if (r.status !== 0) process.exit(r.status || 1);
}

async function cmdWebui(action: string): Promise<void> {
  switch (action) {
    case "up": {
      if (pgrepF("open-webui.*serve")) {
        ok(`open-webui already running at ${WEBUI_URL}`);
        return;
      }
      if (!which("open-webui"))
        die("open-webui not found. Run: uv tool install open-webui");
      say(
        `Starting open-webui on :3000  (1bit.systems-branded, points at ${PROXY_URL}/v1)`,
      );
      nohupSpawn(
        "open-webui",
        ["serve", "--port", "3000", "--host", "127.0.0.1"],
        WEBUI_LOG,
        {
          WEBUI_NAME: "1bit.systems",
          OPENAI_API_BASE_URL: `${PROXY_URL}/v1`,
          OPENAI_API_KEY: "local-no-auth",
          ENABLE_OLLAMA_API: "False",
          WEBUI_AUTH: "False",
        },
      );
      for (let i = 0; i < 30; i++) {
        await sleep(1000);
        if (await urlOk(`${WEBUI_URL}/health`, 1000)) {
          ok(`open-webui up at ${WEBUI_URL}`);
          return;
        }
      }
      warn(`open-webui didn't respond in 30s — see ${WEBUI_LOG}`);
      return;
    }
    case "down": {
      if (pgrepF("open-webui.*serve")) {
        pkillF("open-webui.*serve");
        ok("open-webui stopped");
      } else {
        ok("open-webui already stopped");
      }
      return;
    }
    case "status": {
      if (await urlOk(`${WEBUI_URL}/health`, 2000))
        ok(`open-webui: up at ${WEBUI_URL}`);
      else warn("open-webui: down");
      return;
    }
    default:
      die("usage: 1bit webui [up|down|status]");
  }
}

async function cmdNpu(model: string | undefined): Promise<void> {
  const m = model || DEFAULT_NPU_MODEL;
  if (pgrepF("flm.*serve")) {
    ok(`flm already serving (pid ${pgrepFFirst("flm.*serve")})`);
    return;
  }
  say(`Starting flm serve ${m}`);
  nohupSpawn("flm", ["serve", m], FLM_LOG);
  await sleep(3000);
  const port = flmPort();
  ok(`flm serving ${m} on :${port} (log: ${FLM_LOG})`);
}

function usage(): void {
  console.log(`1bit — 1bit-systems control plane

Usage:
  1bit up                Start lemond + flm + proxy, open browser, launch GAIA if installed
  1bit down              Stop all services
  1bit status            Show service status
  1bit pull <model>      Pull a model (auto-routes NPU vs GPU)
  1bit bench             Run the 1-bit / ternary pile bench
  1bit npu [model]       Start FLM NPU server (default: $DEFAULT_NPU_MODEL)
  1bit webui [up|down|status]  Run Open WebUI on :3000, branded + pointed at proxy

Endpoints (after \`1bit up\`):
  Unified OpenAI:    ${PROXY_URL}/v1/   (recommended — both lanes, one shape)
  Lemonade (GPU):    ${LEMOND_URL}/v1/
  FLM (NPU):         http://127.0.0.1:$(flm port)/v1/

Override the default NPU model: ONEBIT_NPU_MODEL=qwen3:1.7b 1bit up`);
}

// -- entry ----------------------------------------------------------------

async function main() {
  const argv = process.argv.slice(2);
  const sub = argv[0];

  switch (sub) {
    case "up":
      await cmdUp();
      break;
    case "down":
      cmdDown();
      break;
    case "status":
      await cmdStatus();
      break;
    case "pull":
      cmdPull(argv[1]);
      break;
    case "bench":
      cmdBench();
      break;
    case "npu":
      await cmdNpu(argv[1]);
      break;
    case "webui":
      await cmdWebui(argv[1] || "up");
      break;
    case "__proxy__":
      // Internal entrypoint: run the embedded reverse proxy in foreground.
      await startProxy();
      // Keep alive forever — startProxy returns after Bun.serve binds.
      await new Promise(() => {});
      break;
    case "-h":
    case "--help":
    case "help":
    case undefined:
    case "":
      usage();
      break;
    default:
      die(`unknown subcommand: ${sub} (try \`1bit help\`)`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
