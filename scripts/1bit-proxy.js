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
//   PROXY_HOST          (default 127.0.0.1; set 0.0.0.0 for LAN)
//   LEMOND_URL          (default http://127.0.0.1:13305)
//   FLM_URL             (default http://127.0.0.1:52625)

'use strict';

const http = require('http');
const fs = require('fs');
const net = require('net');
const path = require('path');
const { URL } = require('url');
const { execSync } = require('child_process');

// Locate the home page HTML. Prefer a co-installed copy.
const HOME_HTML = (() => {
  const candidates = [
    path.join(__dirname, '1bit-home.html'),
    path.join(__dirname, '..', 'scripts', '1bit-home.html'),
    '/usr/local/share/1bit-systems/1bit-home.html',
    '/home/bcloud/1bit-systems/scripts/1bit-home.html',
  ];
  for (const p of candidates) { if (fs.existsSync(p)) return fs.readFileSync(p, 'utf8'); }
  return null;
})();

const PROXY_PORT = parseInt(process.env.ONEBIT_PROXY_PORT || '13306', 10);
const PROXY_HOST = process.env.PROXY_HOST || '127.0.0.1';
const LEMOND_URL = process.env.LEMOND_URL || 'http://127.0.0.1:13305';
const MAX_BODY_BYTES = parseInt(process.env.ONEBIT_PROXY_MAX_BODY || String(50 * 1024 * 1024), 10);
const FLM_URL = (() => {
  if (process.env.FLM_URL) return process.env.FLM_URL;
  try {
    const out = execSync('flm port', { stdio: ['ignore', 'pipe', 'ignore'] }).toString();
    const m = out.match(/(\d{2,5})/);
    if (m) return `http://127.0.0.1:${m[1]}`;
  } catch {}
  return 'http://127.0.0.1:52625';
})();

const CACHE_TTL_MS = 5_000;
let modelCache = { byId: new Map(), refreshed: 0 };

function fillEmptyContentFromReasoning(obj) {
  for (const choice of obj && obj.choices || []) {
    const message = choice.message || choice.delta;
    if (!message) continue;
    if ((message.content === '' || message.content == null) && message.reasoning_content) {
      message.content = message.reasoning_content;
    }
  }
  return obj;
}

function normalizeSseChunk(chunk, state) {
  state.buf += chunk.toString('utf8');
  const parts = state.buf.split(/\r?\n/);
  state.buf = parts.pop() || '';
  return parts.map(line => {
    if (!line.startsWith('data: ')) return line;
    const data = line.slice(6);
    if (data === '[DONE]') return line;
    try {
      return `data: ${JSON.stringify(fillEmptyContentFromReasoning(JSON.parse(data)))}`;
    } catch {
      return line;
    }
  }).join('\n') + '\n';
}

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

function parseMultipartModel(body) {
  if (!body.length) return undefined;
  // Enough for normal multipart headers and a short model field; avoid
  // stringifying large audio payloads.
  const head = body.subarray(0, Math.min(body.length, 1024 * 1024)).toString('utf8');
  const m = head.match(/name="model"\r?\n\r?\n([^\r\n]+)/);
  return m ? m[1].trim() : undefined;
}

async function readBody(req) {
  const chunks = [];
  let total = 0;
  for await (const c of req) {
    total += c.length;
    if (total > MAX_BODY_BYTES) {
      const err = new Error(`request body too large; max ${MAX_BODY_BYTES} bytes`);
      err.statusCode = 413;
      throw err;
    }
    chunks.push(c);
  }
  return Buffer.concat(chunks);
}

function pickTargetForPath(pathname, modelId) {
  // Lemonade is the default OmniRouter modality surface. FLM ASR is an
  // opt-in side lane for its own whisper-v3:* model family.
  if (pathname === '/v1/audio/transcriptions') {
    return /^whisper-v3[:_-]/i.test(modelId || '') ? FLM_URL : LEMOND_URL;
  }
  // FLM's embedding engine is side-loaded and not currently advertised by
  // /v1/models, so route its known embedding model family explicitly.
  if (pathname === '/v1/embeddings' && /^embed-/i.test(modelId || '')) return FLM_URL;
  // Lemonade owns the Responses API and realtime stack.
  if (pathname === '/v1/responses' || pathname.startsWith('/realtime')) return LEMOND_URL;
  return pickTarget(modelId);
}

const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, `http://127.0.0.1:${PROXY_PORT}`);
  const pathname = u.pathname.replace(/^\/api\/v1(?=\/|$)/, '/v1');

  // Home page
  if ((pathname === '/' || pathname === '/home') && req.method === 'GET') {
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
  if (pathname === '/health' && req.method === 'GET') {
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
  if (pathname === '/v1/models' && req.method === 'GET') {
    if (Date.now() - modelCache.refreshed > CACHE_TTL_MS) await refreshModels();
    res.setHeader('content-type', 'application/json');
    const data = Array.from(modelCache.byId.values()).map(m => ({
      id: m.id, object: 'model', owned_by: m.backend, created: 0,
    }));
    res.end(JSON.stringify({ object: 'list', data }));
    return;
  }

  // Everything else: peek at body for `model` and route
  let body;
  try {
    body = await readBody(req);
  } catch (e) {
    res.writeHead(e.statusCode || 400, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: { message: e.message } }));
    return;
  }
  let modelId;
  if (body.length) {
    try { modelId = JSON.parse(body.toString()).model; } catch {}
  }
  if (!modelId && pathname === '/v1/audio/transcriptions') {
    modelId = parseMultipartModel(body);
  }

  if (Date.now() - modelCache.refreshed > CACHE_TTL_MS) await refreshModels();
  const target = pickTargetForPath(pathname, modelId);
  const tu = new URL(target);

  const fwdHeaders = { ...req.headers, host: `${tu.hostname}:${tu.port || 80}` };
  delete fwdHeaders['content-length'];
  if (body.length) fwdHeaders['content-length'] = String(body.length);

  const upstream = http.request({
    hostname: tu.hostname,
    port: tu.port || 80,
    path: pathname + u.search,
    method: req.method,
    headers: fwdHeaders,
    // flm tears down its TCP socket after each response. node's default
    // globalAgent keeps it pooled and reuses it on the next call →
    // ECONNRESET ("socket hang up"). Fresh socket per request is fine for
    // LLM traffic (handshake is µs, inference is seconds).
    agent: false,
  }, ur => {
    const isChatCompletion = pathname === '/v1/chat/completions';
    const contentType = String(ur.headers['content-type'] || '');
    const isSse = contentType.includes('text/event-stream');
    if (!isChatCompletion) {
      res.writeHead(ur.statusCode, ur.headers);
      ur.pipe(res);
      return;
    }
    if (isSse) {
      res.writeHead(ur.statusCode, ur.headers);
      const state = { buf: '' };
      ur.on('data', chunk => res.write(normalizeSseChunk(chunk, state)));
      ur.on('end', () => {
        if (state.buf) res.write(normalizeSseChunk('\n', state));
        res.end();
      });
      return;
    }

    const chunks = [];
    ur.on('data', chunk => chunks.push(chunk));
    ur.on('end', () => {
      const body = Buffer.concat(chunks);
      if (!contentType.includes('application/json')) {
        res.writeHead(ur.statusCode, ur.headers);
        res.end(body);
        return;
      }
      try {
        const normalized = Buffer.from(JSON.stringify(fillEmptyContentFromReasoning(JSON.parse(body.toString('utf8')))));
        const headers = { ...ur.headers, 'content-length': String(normalized.length) };
        res.writeHead(ur.statusCode, headers);
        res.end(normalized);
      } catch {
        res.writeHead(ur.statusCode, ur.headers);
        res.end(body);
      }
    });
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

server.on('upgrade', (req, socket, head) => {
  const u = new URL(req.url, `http://127.0.0.1:${PROXY_PORT}`);
  if (!u.pathname.startsWith('/realtime')) {
    socket.destroy();
    return;
  }
  const target = new URL(LEMOND_URL);
  const upstream = net.connect(target.port || 80, target.hostname, () => {
    upstream.write(`${req.method} ${u.pathname}${u.search} HTTP/${req.httpVersion}\r\n`);
    for (const [k, v] of Object.entries(req.headers)) {
      upstream.write(`${k}: ${k.toLowerCase() === 'host' ? `${target.hostname}:${target.port || 80}` : v}\r\n`);
    }
    upstream.write('\r\n');
    if (head && head.length) upstream.write(head);
    upstream.pipe(socket);
    socket.pipe(upstream);
  });
  upstream.on('error', () => socket.destroy());
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
