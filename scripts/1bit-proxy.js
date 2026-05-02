#!/usr/bin/env node
//
// 1bit-proxy — unify Lemonade (GPU) + FastFlowLM (NPU) into one OpenAI
// endpoint on :13306.
//
// Routes by model id:
//   - if the model lives in `flm`, requests go to FLM_URL
//   - otherwise, requests go to LEMOND_URL (default lane)
//
// /v1/models returns the union. /v1/chat/completions, /v1/completions,
// /v1/embeddings (etc) are forwarded transparently — including SSE
// streaming — based on the `model` field in the request body.
//
// API only. Lemonade's native UI lives at :13305 — point browsers
// there. Clients point at :13306 for the unified two-lane endpoint.
//
// Pure Node stdlib. No deps. Configure via env:
//   ONEBIT_PROXY_PORT   (default 13306)
//   LEMOND_URL          (default http://127.0.0.1:13305)
//   FLM_URL             (auto-discovered via `flm port` if unset)

'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');
const { execSync } = require('child_process');

// Locate the home page HTML. Prefer a co-installed copy.
const HOME_HTML = (() => {
  const candidates = [
    path.join(__dirname, '1bit-home.html'),
    path.join(__dirname, '..', 'scripts', '1bit-home.html'),
    '/usr/local/share/1bit-systems/1bit-home.html',
    '/home/bcloud/Projects/1bit-systems/scripts/1bit-home.html',
  ];
  for (const p of candidates) { if (fs.existsSync(p)) return fs.readFileSync(p, 'utf8'); }
  return null;
})();

const PROXY_PORT = parseInt(process.env.ONEBIT_PROXY_PORT || '13306', 10);
// Bind all interfaces by default so LAN apps (Open WebUI, Continue, GitHub
// Copilot, AnythingLLM, etc.) can hit the unified endpoint. Set
// PROXY_HOST=127.0.0.1 to restore loopback-only.
const PROXY_HOST = process.env.PROXY_HOST || '0.0.0.0';
const LEMOND_URL = process.env.LEMOND_URL || 'http://127.0.0.1:13305';
const FLM_URL = (() => {
  if (process.env.FLM_URL) return process.env.FLM_URL;
  try {
    const out = execSync('flm port', { stdio: ['ignore', 'pipe', 'ignore'] }).toString();
    const m = out.match(/(\d{2,5})/);
    if (m) return `http://127.0.0.1:${m[1]}`;
  } catch {}
  return 'http://127.0.0.1:11434';
})();

const CACHE_TTL_MS = 5_000;
let modelCache = { byId: new Map(), refreshed: 0 };

function getJson(url, timeoutMs = 2000) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, r => {
      let buf = '';
      r.on('data', c => (buf += c));
      r.on('end', () => {
        try { resolve(JSON.parse(buf)); } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.setTimeout(timeoutMs, () => { req.destroy(new Error('timeout')); });
  });
}

async function refreshModels() {
  const next = new Map();
  await Promise.all([
    getJson(`${LEMOND_URL}/v1/models`).then(d => {
      for (const m of d.data || []) next.set(m.id, { id: m.id, backend: 'lemond' });
    }).catch(() => {}),
    getJson(`${FLM_URL}/v1/models`).then(d => {
      for (const m of d.data || []) next.set(m.id, { id: m.id, backend: 'flm' });
    }).catch(() => {}),
  ]);
  modelCache = { byId: next, refreshed: Date.now() };
}

function pickTarget(modelId) {
  if (!modelId) return LEMOND_URL;
  const entry = modelCache.byId.get(modelId);
  return entry && entry.backend === 'flm' ? FLM_URL : LEMOND_URL;
}

async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  return Buffer.concat(chunks);
}

const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, `http://127.0.0.1:${PROXY_PORT}`);

  // Home page
  if ((u.pathname === '/' || u.pathname === '/home') && req.method === 'GET') {
    if (!HOME_HTML) {
      res.writeHead(404, { 'content-type': 'text/plain' });
      res.end('1bit-home.html not found alongside proxy');
      return;
    }
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
    res.end(HOME_HTML);
    return;
  }

  // Health check for the proxy itself
  if (u.pathname === '/health' && req.method === 'GET') {
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify({
      ok: true,
      lemond: LEMOND_URL,
      flm: FLM_URL,
      models: modelCache.byId.size,
      cached_at: modelCache.refreshed,
    }));
    return;
  }

  // /v1/models → union of both backends
  if (u.pathname === '/v1/models' && req.method === 'GET') {
    if (Date.now() - modelCache.refreshed > CACHE_TTL_MS) await refreshModels();
    res.setHeader('content-type', 'application/json');
    const data = Array.from(modelCache.byId.values()).map(m => ({
      id: m.id, object: 'model', owned_by: m.backend, created: 0,
    }));
    res.end(JSON.stringify({ object: 'list', data }));
    return;
  }

  // Everything else: peek at body for `model` and route
  const body = await readBody(req);
  let modelId;
  if (body.length) {
    try { modelId = JSON.parse(body.toString()).model; } catch {}
  }

  if (Date.now() - modelCache.refreshed > CACHE_TTL_MS) await refreshModels();
  const target = pickTarget(modelId);
  const tu = new URL(target);

  const fwdHeaders = { ...req.headers, host: `${tu.hostname}:${tu.port || 80}` };
  delete fwdHeaders['content-length'];
  if (body.length) fwdHeaders['content-length'] = String(body.length);

  const upstream = http.request({
    hostname: tu.hostname,
    port: tu.port || 80,
    path: u.pathname + u.search,
    method: req.method,
    headers: fwdHeaders,
    // flm tears down its TCP socket after each response. node's default
    // globalAgent keeps it pooled and reuses it on the next call →
    // ECONNRESET ("socket hang up"). Fresh socket per request is fine for
    // LLM traffic (handshake is µs, inference is seconds).
    agent: false,
  }, ur => {
    res.writeHead(ur.statusCode, ur.headers);
    ur.pipe(res);
  });
  upstream.on('error', e => {
    if (!res.headersSent) {
      res.writeHead(502, { 'content-type': 'application/json' });
      res.end(JSON.stringify({
        error: { message: `upstream error: ${e.message}`, target, model: modelId },
      }));
    } else {
      res.end();
    }
  });
  if (body.length) upstream.write(body);
  upstream.end();
});

(async () => {
  await refreshModels();
  server.listen(PROXY_PORT, PROXY_HOST, () => {
    const lemondCount = Array.from(modelCache.byId.values()).filter(m => m.backend === 'lemond').length;
    const flmCount = Array.from(modelCache.byId.values()).filter(m => m.backend === 'flm').length;
    console.log(`[1bit-proxy] listening on http://${PROXY_HOST}:${PROXY_PORT}`);
    console.log(`[1bit-proxy]   lemond: ${LEMOND_URL}  (${lemondCount} models)`);
    console.log(`[1bit-proxy]   flm:    ${FLM_URL}  (${flmCount} models)`);
    setInterval(() => { refreshModels().catch(() => {}); }, 30_000);
  });
})();
