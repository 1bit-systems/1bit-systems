// ==UserScript==
// @name         1bit-history — terminal-style up/down history for chat inputs
// @namespace    https://1bit.systems/
// @version      0.1.0
// @description  Up arrow walks back through previous prompts. Down arrow walks forward. Hit Enter to resend. Works in the Lemonade webapp and any OpenAI-compat chat UI on localhost. History persists per-origin in localStorage.
// @author       1bit.systems
// @homepage     https://1bit.systems/
// @match        http://127.0.0.1:13305/*
// @match        http://localhost:13305/*
// @match        http://127.0.0.1:8000/*
// @match        http://localhost:8000/*
// @match        http://127.0.0.1:8080/*
// @match        http://localhost:8080/*
// @grant        none
// @run-at       document-end
// ==/UserScript==

(function () {
  'use strict';

  const STORAGE_KEY = '1bit-history';
  const MAX_ENTRIES = 200;

  function load() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
    catch { return []; }
  }
  function save(h) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(h.slice(-MAX_ENTRIES))); }
    catch {}
  }

  // A chat input = a textarea, OR a text/search input whose placeholder
  // or aria-label suggests it's for messaging. Keep it conservative so we
  // don't hijack every random search box.
  function isChatInput(el) {
    if (!el || el.disabled || el.readOnly) return false;
    if (el.tagName === 'TEXTAREA') return true;
    if (el.tagName === 'INPUT' && (el.type === 'text' || el.type === 'search')) {
      const hint = ((el.placeholder || '') + ' ' + (el.getAttribute('aria-label') || '') + ' ' + (el.name || '')).toLowerCase();
      return /\b(message|prompt|ask|chat|query|send)\b/.test(hint);
    }
    return false;
  }

  let history = load();
  let cursor = history.length;   // index = length means "not in history yet"
  let draft = '';                // in-progress text, stashed when navigating up

  function recall(el, dir) {
    if (cursor === history.length) draft = el.value;
    const next = cursor + dir;
    if (next < 0 || next > history.length) return;
    cursor = next;
    el.value = (cursor === history.length) ? draft : history[cursor];
    // move caret to end
    try { el.setSelectionRange(el.value.length, el.value.length); } catch {}
    // notify React/Vue/etc. that the value changed
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  function record(text) {
    text = (text || '').trim();
    if (!text) return;
    if (history[history.length - 1] === text) return;  // skip consecutive dupes
    history.push(text);
    save(history);
    cursor = history.length;
    draft = '';
  }

  function caretAtStart(el) {
    return el.selectionStart === 0 && el.selectionEnd === 0;
  }
  function caretAtEnd(el) {
    return el.selectionStart === el.value.length && el.selectionEnd === el.value.length;
  }

  // Capture phase so we beat framework handlers.
  document.addEventListener('keydown', (e) => {
    const el = e.target;
    if (!isChatInput(el)) return;

    if (e.key === 'ArrowUp' && !e.shiftKey && !e.ctrlKey && !e.metaKey && caretAtStart(el)) {
      e.preventDefault();
      e.stopPropagation();
      recall(el, -1);
      return;
    }
    if (e.key === 'ArrowDown' && !e.shiftKey && !e.ctrlKey && !e.metaKey && caretAtEnd(el)) {
      e.preventDefault();
      e.stopPropagation();
      recall(el, +1);
      return;
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      // record before the app's handler fires; do NOT preventDefault
      record(el.value);
    }
  }, true);

  // Some UIs submit via form rather than Enter handler — catch that too.
  document.addEventListener('submit', (e) => {
    const form = e.target;
    const ta = form && form.querySelector && form.querySelector('textarea, input[type=text], input[type=search]');
    if (ta && isChatInput(ta)) record(ta.value);
  }, true);

  console.log('[1bit-history] active — ↑/↓ recall prompts. ' + history.length + ' entries loaded.');
})();
