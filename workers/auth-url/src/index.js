/**
 * Auth URL Worker — SEO verification, OAuth callbacks, URL auth
 *
 * Routes:
 *   GET /auth/googleXXXXX.html   → Google Search Console verification
 *   GET /auth/bingXXXXX.html     → Bing Webmaster Tools verification
 *   GET /auth/yandexXXXXX.html   → Yandex Webmaster verification
 *   GET /auth/pinterestXXXXX.html → Pinterest verification
 *   GET /auth/verify/:provider    → Generic verification endpoint
 *   GET /auth/callback/:provider  → OAuth callback relay
 *   GET /auth/redirect?url=...    → Safe URL redirector
 *   GET /auth/status              → Worker health check
 *
 * Configuration: set these KV namespace bindings or env vars:
 *   VERIFICATION_CODES = { "google": "YOUR_CODE", "bing": "YOUR_CODE", ... }
 */

// ─── Configuration ───────────────────────────────────────────────────────────
// Set these via wrangler secret or env vars.
// For verification: the part after /auth/ before .html

const VERIFICATION_FILES = {
  // Google Search Console — file method
  // Create a file like "google123abc456def.html" at /auth/google123abc456def.html
  // OR use the meta tag method already in index.html
};

// OAuth callback relay config
const OAUTH_CALLBACKS = {};

// Allowed redirect domains (prevent open redirect abuse)
const ALLOWED_REDIRECT_DOMAINS = [
  '1bit.systems',
  'www.1bit.systems',
  'github.com',
  'huggingface.co',
];

// ─── Router ──────────────────────────────────────────────────────────────────

async function handleRequest(request) {
  const url = new URL(request.url);
  const { pathname, searchParams } = url;
  const method = request.method;

  // CORS headers for API routes
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  };

  if (method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  // ── Health check ──
  if (pathname === '/auth/status' || pathname === '/auth/' || pathname === '/status') {
    return new Response(
      JSON.stringify({
        status: 'ok',
        worker: 'auth-url',
        version: '1.0.0',
        site: '1bit.systems',
        timestamp: new Date().toISOString(),
      }),
      {
        status: 200,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache',
        },
      }
    );
  }

  // ── Safe redirector ──
  // /auth/redirect?url=https://github.com/...
  if (pathname === '/auth/redirect') {
    const target = searchParams.get('url');
    if (!target) {
      return new Response('Missing "url" parameter', { status: 400 });
    }

    try {
      const targetUrl = new URL(target);
      const allowed = ALLOWED_REDIRECT_DOMAINS.some((d) =>
        targetUrl.hostname === d || targetUrl.hostname.endsWith('.' + d)
      );
      if (!allowed) {
        return new Response('Redirect domain not allowed', { status: 403 });
      }
    } catch {
      return new Response('Invalid URL', { status: 400 });
    }

    return Response.redirect(target, 302);
  }

  // ── OAuth callback relay ──
  // /auth/callback/google?code=...
  const callbackMatch = pathname.match(/^\/auth\/callback\/(\w+)$/);
  if (callbackMatch) {
    const provider = callbackMatch[1];
    const code = searchParams.get('code');
    const state = searchParams.get('state');

    // Log the callback (for debugging)
    console.log(`OAuth callback from ${provider}: code=${code?.slice(0, 10)}..., state=${state?.slice(0, 10)}...`);

    // Return a JSON response — the client-side app handles the rest
    return new Response(
      JSON.stringify({
        provider,
        received: true,
        code: code ? code.slice(0, 20) + '...' : null,
        state: state || null,
        message: 'OAuth callback received. Close this tab and return to the app.',
      }),
      {
        status: 200,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache',
        },
      }
    );
  }

  // ── BingSiteAuth.xml (root or /auth) ──
  if (pathname === '/BingSiteAuth.xml' || pathname === '/auth/BingSiteAuth.xml') {
    return new Response(
      `<?xml version="1.0"?>\n<users>\n  <user>YOUR_BING_CODE</user>\n</users>`,
      {
        status: 200,
        headers: {
          'Content-Type': 'text/xml; charset=utf-8',
          'Cache-Control': 'public, max-age=86400',
        },
      }
    );
  }

  // ── SEO verification files ──
  // /auth/googleXXXXX.html, /auth/bingXXXXX.html, /googleXXXXX.html (root), etc.
  const verifyMatch = pathname.match(/^(?:\/auth)?\/(\w+)\.html$/);
  if (verifyMatch) {
    const fileId = verifyMatch[1];

    // Google verification: filename is "google" + code
    if (fileId.startsWith('google')) {
      const code = fileId.slice(6); // strip "google"
      return new Response(`google-site-verification: ${code}`, {
        status: 200,
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'Cache-Control': 'public, max-age=86400',
        },
      });
    }

    // Bing verification
    if (fileId.startsWith('bing')) {
      const code = fileId.slice(4);
      return new Response(
        `<?xml version="1.0"?>
<users>
  <user>${code}</user>
</users>`,
        {
          status: 200,
          headers: {
            'Content-Type': 'text/xml; charset=utf-8',
            'Cache-Control': 'public, max-age=86400',
          },
        }
      );
    }

    // Yandex verification
    if (fileId.startsWith('yandex')) {
      const code = fileId.slice(6);
      return new Response(
        `<?xml version="1.0"?>
<document>
  <verification-code>${code}</verification-code>
</document>`,
        {
          status: 200,
          headers: {
            'Content-Type': 'text/xml; charset=utf-8',
            'Cache-Control': 'public, max-age=86400',
          },
        }
      );
    }

    // Pinterest verification
    if (fileId.startsWith('pinterest')) {
      const code = fileId.slice(9);
      return new Response(code, {
        status: 200,
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'Cache-Control': 'public, max-age=86400',
        },
      });
    }

    // Generic verification — serve from env
    const envCode = typeof VERIFICATION_CODES !== 'undefined'
      ? VERIFICATION_CODES[fileId]
      : null;
    if (envCode) {
      return new Response(envCode, {
        status: 200,
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'Cache-Control': 'public, max-age=86400',
        },
      });
    }
  }

  // ── 404 ──
  return new Response(
    JSON.stringify({
      error: 'Not found',
      path: pathname,
      hint: 'Valid routes: /auth/status, /auth/googleXXXXX.html, /auth/redirect?url=..., /auth/callback/:provider',
    }),
    {
      status: 404,
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
      },
    }
  );
}

// ─── Entry Point ─────────────────────────────────────────────────────────────

export default {
  async fetch(request, env, ctx) {
    try {
      return await handleRequest(request);
    } catch (err) {
      console.error('Auth URL Worker error:', err);
      return new Response(
        JSON.stringify({ error: 'Internal error', message: err.message }),
        {
          status: 500,
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache',
          },
        }
      );
    }
  },
};
