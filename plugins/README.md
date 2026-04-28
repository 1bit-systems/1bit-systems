# plugins/

Tiny, optional add-ons for 1bit-systems clients. Each one is a single file you can install in seconds.

## `1bit-history.user.js`

Terminal-style **up-arrow / down-arrow command history** for the Lemonade webapp and any OpenAI-compat chat UI on localhost. Press `↑` to walk back through prior prompts, `↓` to walk forward, Enter to resend. History persists per-origin in `localStorage`.

### Install (Tampermonkey, easiest)

1. Install [Tampermonkey](https://www.tampermonkey.net/) for your browser.
2. Click the Tampermonkey icon → **Create a new script** → paste the contents of `1bit-history.user.js` → Ctrl+S.
3. Visit `http://127.0.0.1:13305/`. Open devtools console — you should see `[1bit-history] active`.

### Install (raw bookmarklet, no extension)

If you don't want Tampermonkey, drop the script body into a `javascript:` bookmarklet and click it whenever the chat UI loads. Less convenient — the userscript path is recommended.

### What it does

- `↑` when caret is at start of input → previous prompt
- `↓` when caret is at end of input → next prompt
- Multi-line drafts: arrow keys still navigate the textarea normally; history only triggers at the input boundaries (so you don't lose your half-written prompt by accident)
- Stashes the in-progress draft when you walk back; restores it when you walk forward past the last history entry
- Skips consecutive duplicates so spamming the same prompt doesn't fill up history
- Cap: 200 entries, oldest evicted

### What it doesn't do

- No cross-device sync (localStorage is per-browser, per-origin)
- No fuzzy search (yet) — Ctrl+R reverse-search would be a nice follow-up
- Doesn't work in AMD GAIA (it's Electron, not a webview we can userscript-into). If you want this in GAIA, a fork + upstream PR is the path.
