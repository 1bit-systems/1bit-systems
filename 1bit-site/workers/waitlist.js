// Cloudflare Worker — /api/waitlist endpoint.
//
// POST { email, plan?, referrer? }
//   → stores in KV namespace WAITLIST (bound as binding name WAITLIST)
//   → returns { ok, position, tier }
//
// Minimal: no auth, no rate limit beyond Cloudflare's built-in, no
// captcha (WAF handles bot traffic). Email is stored with timestamp
// and reverse-indexed count for position number.
//
// Deploy:
//   wrangler publish 1bit-site/workers/waitlist.js \
//     --name waitlist \
//     --route 1bit.systems/api/waitlist \
//     --kv-namespace WAITLIST \
//     --binding WAITLIST=<kv-id>
//
// Same route attached to 1bit.music / .video / .stream / .audio so
// the form posts to whichever apex the user is on without CORS.

const MAX_EMAIL = 254; // RFC 5321 maximum

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname !== "/api/waitlist") {
      return new Response("not found", { status: 404 });
    }
    if (request.method !== "POST") {
      return new Response("method not allowed", { status: 405 });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ ok: false, error: "body must be JSON" }, 400);
    }

    const email = (body.email || "").trim().toLowerCase();
    const plan  = (body.plan  || "premium-monthly").trim();
    const ref   = (body.referrer || "").trim();

    if (!email || email.length > MAX_EMAIL || !email.includes("@")) {
      return json({ ok: false, error: "invalid email" }, 400);
    }

    const key = `email:${email}`;

    // Dedup — if the email is already on file, return their existing
    // position rather than shifting everyone else's.
    const existing = await env.WAITLIST.get(key, { type: "json" });
    if (existing) {
      return json({
        ok: true,
        tier: "duplicate",
        position: existing.position,
        registered: existing.ts,
      });
    }

    // Count → new position. Race-prone at tiny scale but fine for
    // waitlist purposes. KV is eventually consistent; a +/- 1 on the
    // position number is acceptable.
    const countBefore = parseInt(await env.WAITLIST.get("count") || "0", 10);
    const position = countBefore + 1;
    const ts = new Date().toISOString();

    await env.WAITLIST.put(key, JSON.stringify({
      email, plan, ref, position, ts,
    }));
    await env.WAITLIST.put("count", String(position));

    return json({
      ok: true,
      tier: position <= 1000 ? "founder" : "standard",
      position,
      registered: ts,
      message: position <= 1000
        ? `Locked at $5/mo for life — position ${position}/1000.`
        : `On the waitlist — position ${position}.`,
    });
  },
};

function json(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "POST, OPTIONS",
      "access-control-allow-headers": "content-type",
    },
  });
}
