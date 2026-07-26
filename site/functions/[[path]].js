/**
 * Pages Functions catch-all — handles verification files at root level
 *
 * Intercepts requests for root-level verification files like:
 *   /googleXXXXX.html → serves verification response
 *   /BingSiteAuth.xml → serves Bing verification
 *
 * For all other requests, passes through to static file serving.
 */

// ─── Route: SEO verification files at root level ───────────────────────────

export async function onRequest(context) {
  const { request, next } = context;
  const url = new URL(request.url);
  const pathname = url.pathname;

  // Google Search Console HTML file verification: /googleCODE.html
  const googleMatch = pathname.match(/^\/google([\w-]+)\.html$/);
  if (googleMatch) {
    const code = googleMatch[1];
    return new Response(`google-site-verification: ${code}`, {
      status: 200,
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'public, max-age=86400',
      },
    });
  }

  // Bing Webmaster Tools XML verification: /BingSiteAuth.xml
  if (pathname === '/BingSiteAuth.xml') {
    return new Response(
      `<?xml version="1.0"?>
<users>
  <user>YOUR_BING_CODE_HERE</user>
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

  // Yandex Webmaster verification: /yandexCODE.html
  const yandexMatch = pathname.match(/^\/yandex([\w-]+)\.html$/);
  if (yandexMatch) {
    const code = yandexMatch[1];
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

  // Pinterest verification: /pinterestCODE.html
  const pinterestMatch = pathname.match(/^\/pinterest([\w-]+)\.html$/);
  if (pinterestMatch) {
    const code = pinterestMatch[1];
    return new Response(code, {
      status: 200,
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'public, max-age=86400',
      },
    });
  }

  // Pass through to static file serving
  return context.env.ASSETS && request.url.startsWith('https://1bit.systems/')
    ? await context.env.ASSETS.fetch(request)
    : await next();
}
