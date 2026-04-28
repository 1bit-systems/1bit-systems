// ==UserScript==
// @name         1bit-menu-sections — group Lemonade models into "1bit LLM" + "Ryzen LLM"
// @namespace    https://1bit.systems/
// @version      0.1.0
// @description  Reorganizes the Lemonade webapp model list into two sections: "1bit LLM" (the ternary / sub-2-bit pile we care about) and "Ryzen LLM (NPU lane)" (the FastFlowLM / XDNA 2 pile). The NPU section is informational — clicking copies `1bit npu <model>` to clipboard, since FLM runs on a separate port from Lemonade.
// @author       1bit.systems
// @match        http://127.0.0.1:13305/*
// @match        http://localhost:13305/*
// @grant        none
// @run-at       document-end
// ==/UserScript==

(function () {
  'use strict';

  // -- categorization rules ------------------------------------------------
  const ONEBIT_RE = /\b(bitnet|bonsai|trilm|outlier|ternary|tq[12]_0|iq1_[ms]|iq2_bn|sub-?2-?bit)\b/i;

  // FastFlowLM-supported NPU models. Sourced from `flm list` 2026-04-27.
  // Update this list when the upstream pile grows.
  const RYZEN_MODELS = [
    'qwen3:0.6b', 'qwen3:1.7b', 'qwen3:4b', 'qwen3:8b',
    'qwen3.5:0.8b', 'qwen3.5:2b', 'qwen3.5:4b', 'qwen3.5:9b',
    'qwen3-it:4b', 'qwen3-tk:4b', 'qwen3vl-it:4b',
    'qwen2.5-it:3b', 'qwen2.5vl-it:3b',
    'gemma3:1b', 'gemma3:4b', 'gemma4-it:e2b',
    'phi4-mini-it:4b',
    'lfm2:1.2b', 'lfm2:2.6b', 'lfm2-trans:2.6b',
    'lfm2.5-it:1.2b', 'lfm2.5-tk:1.2b',
    'llama3.1:8b', 'llama3.2:1b', 'llama3.2:3b',
    'deepseek-r1:8b', 'deepseek-r1-0528:8b',
    'gpt-oss:20b', 'gpt-oss-sg:20b',
    'nanbeige4.1:3b',
    'medgemma:4b', 'medgemma1.5:4b', 'translategemma:4b',
    'whisper-v3:turbo',
    'embed-gemma:300m',
  ];

  function classify(name) {
    if (!name) return 'other';
    if (ONEBIT_RE.test(name)) return 'onebit';
    return 'other';
  }

  // -- DOM helpers ---------------------------------------------------------
  const SECTION_MARKER = 'data-1bit-section';

  function makeHeader(label, count) {
    const h = document.createElement('div');
    h.className = 'available-models-header onebit-section-header';
    h.setAttribute(SECTION_MARKER, label);
    h.style.cssText = 'padding:10px 14px;margin:14px 0 6px;font-size:0.78rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#ffc832;border-top:1px solid #2a2a2a;display:flex;align-items:center;justify-content:space-between;';
    h.innerHTML = `<span>${label}</span><span style="color:#666;font-weight:500;font-size:0.72rem;">${count}</span>`;
    return h;
  }

  function makeRyzenItem(modelTag) {
    const row = document.createElement('div');
    row.className = 'model-item backend-row-item onebit-ryzen-item';
    row.setAttribute(SECTION_MARKER, 'ryzen');
    row.style.cssText = 'padding:8px 14px;display:flex;align-items:center;gap:10px;cursor:pointer;border-radius:6px;margin:2px 0;transition:background 0.15s;';
    row.innerHTML = `
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ffc832;flex-shrink:0;" title="NPU"></span>
      <span style="flex:1;font-size:0.88rem;color:#ddd;">${modelTag}</span>
      <span style="font-size:0.7rem;color:#888;font-family:monospace;">flm</span>
    `;
    row.title = `Ryzen NPU model. Click to copy command:\n\n  1bit npu ${modelTag}\n\nThe NPU runs on a separate port from Lemonade — see "1bit npu" output.`;
    row.addEventListener('mouseenter', () => row.style.background = '#1f1f1f');
    row.addEventListener('mouseleave', () => row.style.background = '');
    row.addEventListener('click', async (e) => {
      e.stopPropagation();
      const cmd = `1bit npu ${modelTag}`;
      try {
        await navigator.clipboard.writeText(cmd);
        flashToast(`Copied: ${cmd}`);
      } catch {
        flashToast(`Run in terminal: ${cmd}`);
      }
    });
    return row;
  }

  function flashToast(msg) {
    const t = document.createElement('div');
    t.textContent = msg;
    t.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#1f1f1f;color:#ffc832;padding:10px 14px;border-radius:6px;font-size:0.88rem;font-family:monospace;border:1px solid #2a2a2a;z-index:99999;box-shadow:0 4px 12px rgba(0,0,0,0.4);';
    document.body.appendChild(t);
    setTimeout(() => { t.style.transition = 'opacity 0.3s'; t.style.opacity = '0'; }, 1800);
    setTimeout(() => t.remove(), 2200);
  }

  // -- main reorganization -------------------------------------------------
  function reorganize() {
    // Find the available-models container. Lemonade uses
    // `.available-models-section.widget` based on the bundle.
    const sections = document.querySelectorAll('.available-models-section, .loaded-model-list');
    if (!sections.length) return false;

    let touched = false;
    sections.forEach(section => {
      // Already organized? skip
      if (section.querySelector(`[${SECTION_MARKER}="1bit LLM"]`)) return;

      const items = Array.from(section.querySelectorAll('.model-item.backend-row-item'));
      if (items.length === 0) return;

      // Bucket items by classification using their visible text
      const buckets = { onebit: [], other: [] };
      for (const it of items) {
        const txt = (it.textContent || '').trim();
        buckets[classify(txt)].push(it);
      }

      // Build new ordered children: [1bit header, 1bit items, Ryzen header, Ryzen items, Other header, Other items]
      const onebitHdr = makeHeader('1bit LLM', buckets.onebit.length);
      const ryzenHdr  = makeHeader('Ryzen LLM (NPU lane)', RYZEN_MODELS.length);
      const otherHdr  = makeHeader('Other', buckets.other.length);

      // Detach + re-append in order
      buckets.onebit.forEach(el => el.remove());
      buckets.other.forEach(el => el.remove());

      section.appendChild(onebitHdr);
      buckets.onebit.forEach(el => section.appendChild(el));

      section.appendChild(ryzenHdr);
      RYZEN_MODELS.forEach(tag => section.appendChild(makeRyzenItem(tag)));

      section.appendChild(otherHdr);
      buckets.other.forEach(el => section.appendChild(el));

      touched = true;
    });
    return touched;
  }

  // Re-run on every DOM mutation since Lemonade is React and re-renders.
  // Debounce so we don't burn cycles.
  let pending = null;
  const observer = new MutationObserver(() => {
    if (pending) return;
    pending = setTimeout(() => {
      pending = null;
      reorganize();
    }, 150);
  });
  observer.observe(document.body, { childList: true, subtree: true });

  // First pass once DOM is ready
  if (document.readyState !== 'loading') {
    setTimeout(reorganize, 200);
  } else {
    document.addEventListener('DOMContentLoaded', () => setTimeout(reorganize, 200));
  }

  console.log('[1bit-menu-sections] active — grouping Lemonade models into 1bit / Ryzen / Other.');
})();
