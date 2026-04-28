// proxy.ts — OpenAI-compat reverse proxy that fans :13306 out to
// lemond:13305 (GPU) and flm:auto (NPU) based on the requested model.
//
// Bit-exact port of scripts/1bit-proxy.js, rewritten for Bun. Streaming
// SSE works because we pipe the upstream Response.body straight back.

import { execSync } from "node:child_process";
// home.html is embedded into the compiled binary at build time.
// `with { type: "text" }` makes Bun inline it as a string literal.
// @ts-ignore — Bun-specific import attribute for raw text embedding.
import HOME_HTML from "./home.html" with { type: "text" };

const PROXY_PORT = parseInt(process.env.ONEBIT_PROXY_PORT || "13306", 10);
const LEMOND_URL = process.env.LEMOND_URL || "http://127.0.0.1:13305";
const FLM_URL: string = (() => {
  if (process.env.FLM_URL) return process.env.FLM_URL;
  try {
    const out = execSync("flm port", {
      stdio: ["ignore", "pipe", "ignore"],
    }).toString();
    const m = out.match(/(\d{2,5})/);
    if (m) return `http://127.0.0.1:${m[1]}`;
  } catch {}
  return "http://127.0.0.1:11434";
})();

const CACHE_TTL_MS = 5_000;

type ModelEntry = { id: string; backend: "lemond" | "flm" };
let modelCache: { byId: Map<string, ModelEntry>; refreshed: number } = {
  byId: new Map(),
  refreshed: 0,
};

async function getJson(url: string, timeoutMs = 2000): Promise<any> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { signal: ctrl.signal });
    return await r.json();
  } finally {
    clearTimeout(t);
  }
}

async function refreshModels(): Promise<void> {
  const next = new Map<string, ModelEntry>();
  await Promise.all([
    getJson(`${LEMOND_URL}/v1/models`)
      .then((d) => {
        for (const m of d?.data || [])
          next.set(m.id, { id: m.id, backend: "lemond" });
      })
      .catch(() => {}),
    getJson(`${FLM_URL}/v1/models`)
      .then((d) => {
        for (const m of d?.data || [])
          next.set(m.id, { id: m.id, backend: "flm" });
      })
      .catch(() => {}),
  ]);
  modelCache = { byId: next, refreshed: Date.now() };
}

function pickTarget(modelId: string | undefined): string {
  if (!modelId) return LEMOND_URL;
  const entry = modelCache.byId.get(modelId);
  return entry && entry.backend === "flm" ? FLM_URL : LEMOND_URL;
}

export async function startProxy(): Promise<void> {
  await refreshModels();

  const server = Bun.serve({
    port: PROXY_PORT,
    hostname: "127.0.0.1",
    // 30 min idle for long completions.
    idleTimeout: 0,
    async fetch(req) {
      const u = new URL(req.url);

      // Home page.
      if (
        (u.pathname === "/" || u.pathname === "/home") &&
        req.method === "GET"
      ) {
        if (!HOME_HTML) {
          return new Response("1bit-home.html not embedded in binary", {
            status: 404,
            headers: { "content-type": "text/plain" },
          });
        }
        return new Response(HOME_HTML, {
          status: 200,
          headers: { "content-type": "text/html; charset=utf-8" },
        });
      }

      // Health check.
      if (u.pathname === "/health" && req.method === "GET") {
        return Response.json({
          ok: true,
          lemond: LEMOND_URL,
          flm: FLM_URL,
          models: modelCache.byId.size,
          cached_at: modelCache.refreshed,
        });
      }

      // Union of /v1/models from both backends.
      if (u.pathname === "/v1/models" && req.method === "GET") {
        if (Date.now() - modelCache.refreshed > CACHE_TTL_MS)
          await refreshModels();
        const data = Array.from(modelCache.byId.values()).map((m) => ({
          id: m.id,
          object: "model",
          owned_by: m.backend,
          created: 0,
        }));
        return Response.json({ object: "list", data });
      }

      // Everything else: peek at body for `model` and route.
      const bodyBuf =
        req.method === "GET" || req.method === "HEAD"
          ? null
          : await req.arrayBuffer();
      let modelId: string | undefined;
      if (bodyBuf && bodyBuf.byteLength) {
        try {
          const parsed = JSON.parse(new TextDecoder().decode(bodyBuf));
          modelId = parsed?.model;
        } catch {}
      }

      if (Date.now() - modelCache.refreshed > CACHE_TTL_MS)
        await refreshModels();
      const target = pickTarget(modelId);
      const tu = new URL(target);

      // Build forwarded headers (drop hop-by-hop).
      const fwdHeaders = new Headers(req.headers);
      fwdHeaders.set("host", `${tu.hostname}${tu.port ? ":" + tu.port : ""}`);
      fwdHeaders.delete("content-length");
      if (bodyBuf && bodyBuf.byteLength)
        fwdHeaders.set("content-length", String(bodyBuf.byteLength));

      const targetUrl = `${target}${u.pathname}${u.search}`;

      try {
        const upstream = await fetch(targetUrl, {
          method: req.method,
          headers: fwdHeaders,
          body: bodyBuf && bodyBuf.byteLength ? bodyBuf : undefined,
          // @ts-ignore — Bun supports duplex/half for streaming.
          redirect: "manual",
        });
        // Pipe straight through — including SSE chunks.
        return new Response(upstream.body, {
          status: upstream.status,
          headers: upstream.headers,
        });
      } catch (e: any) {
        return new Response(
          JSON.stringify({
            error: {
              message: `upstream error: ${e?.message || e}`,
              target,
              model: modelId,
            },
          }),
          { status: 502, headers: { "content-type": "application/json" } },
        );
      }
    },
  });

  const lemondCount = Array.from(modelCache.byId.values()).filter(
    (m) => m.backend === "lemond",
  ).length;
  const flmCount = Array.from(modelCache.byId.values()).filter(
    (m) => m.backend === "flm",
  ).length;
  console.log(`[1bit-proxy] listening on http://127.0.0.1:${server.port}`);
  console.log(
    `[1bit-proxy]   lemond: ${LEMOND_URL}  (${lemondCount} models)`,
  );
  console.log(`[1bit-proxy]   flm:    ${FLM_URL}  (${flmCount} models)`);

  setInterval(() => {
    refreshModels().catch(() => {});
  }, 30_000);
}
