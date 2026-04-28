#!/usr/bin/env node
//
// 1bit-proxy — unify Lemonade (GPU) + FastFlowLM (NPU) into one OpenAI
// endpoint on :13306, AND serve the Lemonade webapp with the
// 1bit-menu-sections plugin pre-injected.
//
// Routes by model id:
//   - if the model lives in `flm`, requests go to FLM_URL
//   - otherwise, requests go to LEMOND_URL (default lane)
//
// /v1/models returns the union. /v1/chat/completions, /v1/completions,
// /v1/embeddings (etc) are forwarded transparently — including SSE
// streaming — based on the `model` field in the request body.
//
// HTML responses from lemond have our menu-sections script spliced
// into <head> — so the dropdown groups into "1bit LLM" / "Ryzen LLM"
// without needing Tampermonkey.
//
// Pure Node stdlib. No deps. Configure via env:
//   ONEBIT_PROXY_PORT       (default 13306)
//   LEMOND_URL              (default http://127.0.0.1:13305)
//   FLM_URL                 (auto-discovered via `flm port` if unset)
//   ONEBIT_PLUGIN_PATH      (default: sibling 1bit-menu-sections.user.js)
//   ONEBIT_DISABLE_INJECT   (set to "1" to disable HTML injection)

'use strict';

const http = require('http');
const zlib = require('zlib');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');
const { execSync } = require('child_process');

const PROXY_PORT = parseInt(process.env.ONEBIT_PROXY_PORT || '13306', 10);
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
const INJECT_DISABLED = process.env.ONEBIT_DISABLE_INJECT === '1';

// Locate and load the menu plugin source for inline HTML injection.
const PLUGIN_PATH = (() => {
  if (process.env.ONEBIT_PLUGIN_PATH) return process.env.ONEBIT_PLUGIN_PATH;
  const candidates = [
    path.join(__dirname, '..', 'plugins', '1bit-menu-sections.user.js'),
    path.join(__dirname, '1bit-menu-sections.user.js'),
    '/usr/local/share/1bit-systems/1bit-menu-sections.user.js',
    '/home/bcloud/Projects/1bit-systems/plugins/1bit-menu-sections.user.js',
  ];
  for (const p of candidates) { if (fs.existsSync(p)) return p; }
  return '';
})();
const PLUGIN_SRC = (PLUGIN_PATH && !INJECT_DISABLED)
  ? fs.readFileSync(PLUGIN_PATH, 'utf8').replace(/^\/\/ ==UserScript==[\s\S]*?\/\/ ==\/UserScript==/m, '')
  : '';

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

// Forward a request to a target URL, optionally rewriting HTML responses
// to inject our plugin script.
function forward(req, res, body, targetUrl, opts = {}) {
  const { rewriteHtml = false, modelId } = opts;
  const tu = new URL(targetUrl);

  const fwdHeaders = { ...req.headers, host: `${tu.hostname}:${tu.port || 80}` };
  delete fwdHeaders['content-length'];
  if (body.length) fwdHeaders['content-length'] = String(body.length);

  const upstream = http.request({
    hostname: tu.hostname,
    port: tu.port || 80,
    path: req.url,
    method: req.method,
    headers: fwdHeaders,
  }, ur => {
    const isHtml = (ur.headers['content-type'] || '').includes('text/html');

    if (rewriteHtml && isHtml && PLUGIN_SRC) {
      // Buffer the (possibly compressed) body, decompress if needed,
      // splice in our script, return uncompressed.
      const enc = (ur.headers['content-encoding'] || '').toLowerCase();
      const bufs = [];
      ur.on('data', c => bufs.push(c));
      ur.on('end', () => {
        const raw = Buffer.concat(bufs);
        let plain;
        try {
          if (enc === 'gzip')      plain = zlib.gunzipSync(raw);
          else if (enc === 'deflate') plain = zlib.inflateSync(raw);
          else if (enc === 'br')      plain = zlib.brotliDecompressSync(raw);
          else                        plain = raw;
        } catch (e) {
          // Decompression failed — pass through untouched rather than
          // serving mojibake.
          const headers = { ...ur.headers };
          res.writeHead(ur.statusCode, headers);
          res.end(raw);
          return;
        }
        let html = plain.toString('utf8');
        const tag = `\n<script>/* 1bit-menu-sections (proxy-injected) */\n(function(){${PLUGIN_SRC}\n})();</script>\n`;
        if (html.includes('</head>')) {
          html = html.replace('</head>', tag + '</head>');
        } else if (html.includes('<body')) {
          html = html.replace('<body', tag + '<body');
        } else {
          html = tag + html;
        }
        const out = Buffer.from(html, 'utf8');
        const headers = { ...ur.headers };
        // We're emitting plain utf-8 now — strip transfer/encoding markers
        // and recompute content-length.
        delete headers['content-length'];
        delete headers['content-encoding'];
        delete headers['transfer-encoding'];
        headers['content-length'] = String(out.length);
        res.writeHead(ur.statusCode, headers);
        res.end(out);
      });
    } else {
      // Pass-through stream (handles SSE, binaries, JSON, etc.)
      res.writeHead(ur.statusCode, ur.headers);
      ur.pipe(res);
    }
  });
  upstream.on('error', e => {
    if (!res.headersSent) {
      res.writeHead(502, { 'content-type': 'application/json' });
      res.end(JSON.stringify({
        error: { message: `upstream error: ${e.message}`, target: targetUrl, model: modelId },
      }));
    } else {
      res.end();
    }
  });
  if (body.length) upstream.write(body);
  upstream.end();
}

const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, `http://127.0.0.1:${PROXY_PORT}`);

  // Health check for the proxy itself
  if (u.pathname === '/health' && req.method === 'GET') {
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify({
      ok: true,
      lemond: LEMOND_URL,
      flm: FLM_URL,
      models: modelCache.byId.size,
      cached_at: modelCache.refreshed,
      inject: !INJECT_DISABLED && !!PLUGIN_SRC,
      plugin: PLUGIN_PATH || null,
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

  // For inference traffic on /v1/* with a body, route by model.
  // For everything else (UI HTML, JS bundles, /api/* admin calls), forward
  // to lemond and rewrite HTML responses if applicable.
  const isInferenceRoute = u.pathname.startsWith('/v1/') || u.pathname.startsWith('/api/v1/');
  const body = await readBody(req);

  let modelId;
  if (isInferenceRoute && body.length) {
    try { modelId = JSON.parse(body.toString()).model; } catch {}
  }

  if (modelId) {
    if (Date.now() - modelCache.refreshed > CACHE_TTL_MS) await refreshModels();
  }
  const target = modelId ? pickTarget(modelId) : LEMOND_URL;

  // Only rewrite HTML for top-level UI requests (not inference, not API).
  const shouldRewrite = !isInferenceRoute && !INJECT_DISABLED && PLUGIN_SRC;

  forward(req, res, body, target, { rewriteHtml: shouldRewrite, modelId });
});

(async () => {
  await refreshModels();
  server.listen(PROXY_PORT, '127.0.0.1', () => {
    const lemondCount = Array.from(modelCache.byId.values()).filter(m => m.backend === 'lemond').length;
    const flmCount = Array.from(modelCache.byId.values()).filter(m => m.backend === 'flm').length;
    console.log(`[1bit-proxy] listening on http://127.0.0.1:${PROXY_PORT}`);
    console.log(`[1bit-proxy]   lemond: ${LEMOND_URL}  (${lemondCount} models)`);
    console.log(`[1bit-proxy]   flm:    ${FLM_URL}  (${flmCount} models)`);
    if (PLUGIN_SRC) {
      console.log(`[1bit-proxy]   inject: ${PLUGIN_PATH}  (menu sections active in served UI)`);
    } else {
      console.log(`[1bit-proxy]   inject: disabled`);
    }
    setInterval(() => { refreshModels().catch(() => {}); }, 30_000);
  });
})();
