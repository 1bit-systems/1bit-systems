// Attach a Copy button to every <pre> that contains a <code> block.
// Pure vanilla JS — no framework, no CDN, no build step.
(() => {
  "use strict";

  const SELECTOR = "pre > code";
  const LABEL_IDLE = "copy";
  const LABEL_DONE = "copied";
  const LABEL_FAIL = "retry";

  function attach(codeEl) {
    const pre = codeEl.parentElement;
    if (!pre || pre.querySelector(".copy-btn")) return;
    pre.classList.add("has-copy");

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.setAttribute("aria-label", "Copy code to clipboard");
    btn.textContent = LABEL_IDLE;

    btn.addEventListener("click", async () => {
      const text = codeEl.innerText;
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
        } else {
          const ta = document.createElement("textarea");
          ta.value = text;
          ta.setAttribute("readonly", "");
          ta.style.position = "absolute";
          ta.style.left = "-9999px";
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          document.body.removeChild(ta);
        }
        btn.textContent = LABEL_DONE;
      } catch (_) {
        btn.textContent = LABEL_FAIL;
      }
      setTimeout(() => (btn.textContent = LABEL_IDLE), 1400);
    });

    pre.appendChild(btn);
  }

  function scan() {
    document.querySelectorAll(SELECTOR).forEach(attach);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scan);
  } else {
    scan();
  }
})();
