// geo-pd.js — geographic public-domain indicator.
//
// Hits /cdn-cgi/trace (free, Cloudflare Workers-adjacent endpoint that
// returns the visitor's country code in `loc=XX`) and checks each
// track/film element for a `data-pd` attribute listing the ISO country
// codes where the work is public domain. If the visitor's country is
// NOT in that list, overlay a red-X banner with an honest notice.
//
// We never block playback — this is informational only. Users are
// responsible for their own jurisdictions.

(function () {
  "use strict";

  const BANNER_ID = "geo-pd-banner";

  async function getCountry() {
    // Cloudflare's trace endpoint is served from the same origin when
    // the site is behind Cloudflare. Parses as `key=value\n` per line.
    try {
      const r = await fetch("/cdn-cgi/trace", { cache: "no-store" });
      if (!r.ok) return null;
      const text = await r.text();
      const m = text.match(/^loc=([A-Z]{2})$/m);
      return m ? m[1] : null;
    } catch {
      return null;
    }
  }

  function normaliseList(raw) {
    // `data-pd="CA,AU,NZ,EU-50"` — accept comma-separated, plus some
    // pseudo-codes for regions.
    return (raw || "")
      .toUpperCase()
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  function expand(codes) {
    // Expand region pseudo-codes into ISO country sets.
    const out = new Set();
    for (const c of codes) {
      if (c === "EU-50") {
        // EU jurisdictions where life+50 grandfathered cinematographic
        // terms still hold. Simplification — not a legal opinion.
        ["AT","BE","BG","CY","CZ","DE","DK","EE","ES","FI","FR","GR",
         "HR","HU","IE","IT","LT","LU","LV","MT","NL","PL","PT","RO",
         "SE","SI","SK","IS","LI","NO","GB"].forEach((x) => out.add(x));
      } else if (c === "EN-CW") {
        // English-language Commonwealth countries with broad PD alignment
        ["AU","NZ","ZA","IE"].forEach((x) => out.add(x));
      } else {
        out.add(c);
      }
    }
    return out;
  }

  function attach(country) {
    document.querySelectorAll("[data-pd]").forEach((el) => {
      const allowed = expand(normaliseList(el.dataset.pd));
      const isPd = !country || allowed.has(country);

      // Add a small indicator before the first audio/video element
      // inside the card. Red X if NOT in visitor's jurisdiction.
      const indicator = document.createElement("div");
      indicator.className = "geo-pd-indicator";
      indicator.style.cssText = [
        "display:inline-flex",
        "align-items:center",
        "gap:.4rem",
        "font-size:.78rem",
        "font-family:ui-monospace,monospace",
        "padding:.35rem .7rem",
        "border-radius:4px",
        "margin:.3rem 0 .8rem",
        isPd
          ? "background:rgba(124,255,196,.08);color:#7cffc4;border:1px solid rgba(124,255,196,.25)"
          : "background:rgba(255,124,124,.08);color:#ff7c7c;border:1px solid rgba(255,124,124,.25)",
      ].join(";");

      const icon = document.createElement("span");
      icon.textContent = isPd ? "●" : "✕";
      icon.setAttribute("aria-hidden", "true");
      indicator.appendChild(icon);

      const label = document.createElement("span");
      if (!country) {
        label.textContent = "Public-domain status: location unknown";
      } else if (isPd) {
        label.textContent = `Public domain in ${country} (your location)`;
      } else {
        label.innerHTML =
          `<b>Not public domain in ${country} (your location).</b> ` +
          `Still playable; verify your own jurisdiction. ` +
          `<a href="/legal/pd" style="color:inherit;text-decoration:underline;">why?</a>`;
      }
      indicator.appendChild(label);

      const anchor = el.querySelector("audio, video, .watch") || el.firstElementChild;
      if (anchor) {
        anchor.parentNode.insertBefore(indicator, anchor);
      } else {
        el.appendChild(indicator);
      }
    });
  }

  // Top-of-page banner — only shown when visitor is outside any of the
  // page-wide PD jurisdictions declared on <main data-pd="…">.
  function topBanner(country) {
    const main = document.querySelector("[data-pd-page]");
    if (!main) return;
    const allowed = expand(normaliseList(main.dataset.pdPage));
    if (!country || allowed.has(country)) return;

    const bar = document.createElement("div");
    bar.id = BANNER_ID;
    bar.style.cssText = [
      "background:#2a1616",
      "color:#ffb5b5",
      "padding:.6rem 1rem",
      "text-align:center",
      "font-size:.88rem",
      "border-bottom:1px solid #4a2020",
      "font-family:system-ui,sans-serif",
    ].join(";");
    bar.innerHTML =
      `<span style="color:#ff7c7c">✕</span> ` +
      `Some works on this page are not public domain in <b>${country}</b>. ` +
      `Still viewable here (we operate under Canadian law). Verify your own jurisdiction before reproducing or redistributing. ` +
      `<a href="/legal/pd" style="color:inherit;text-decoration:underline;">Why?</a>`;
    document.body.insertBefore(bar, document.body.firstChild);
  }

  (async () => {
    const country = await getCountry();
    try { topBanner(country); } catch {}
    try { attach(country); } catch {}
  })();
})();
