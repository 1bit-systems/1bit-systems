//! LLM-backed specialists — replaces the stub `{}` returns with a real
//! round-trip to 1bit-halo-server's `/v1/chat/completions` endpoint.
//!
//! One struct, [`LlmSpecialist`], parametrised by `(Name, system_prompt)`.
//! Each instance is a single-turn prompt-the-LLM-and-relay-the-text worker.
//! Registered by [`crate::Registry::default_live`] for the four
//! specialists the Discord → echo pipeline actually delivers today:
//!
//! * [`Name::Herald`]        — conversational Q&A + chat-overflow
//! * [`Name::Sentinel`]      — bug-triage in auto-created threads
//! * [`Name::Magistrate`]    — feature-request review
//! * [`Name::Quartermaster`] — GitHub event triage
//!
//! Output contract: every response carries a top-level `"text"` field,
//! even on error. `1bit-watch-discord` reads `resp["text"]` and relays
//! it through echo's Discord client. A network failure must not break
//! the pipeline — instead we surface `(<name> down)` so operators see
//! something in channel rather than silent nothing.

use anyhow::Result;
use async_trait::async_trait;
use onebit_retrieval::{WikiIndex, format_for_system_prompt};
use reqwest::Client;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

use crate::{Name, Specialist};

/// Default base URL for 1bit-halo-server on the local box. The full
/// chat-completions path is `{base}/v1/chat/completions`.
pub fn default_base_url() -> String {
    "http://127.0.0.1:8180".to_string()
}

/// Default model id served by 1bit-halo-server today.
pub fn default_model_id() -> String {
    "halo-1bit-2b".to_string()
}

/// Hard cap on how long we'll wait for the LLM. Cold-cache first
/// completions on strixhalo have been clocked at ~7-12s for a 400-tok
/// reply; warm calls land in 2-4s. 30s gives enough slack for the
/// first-dispatch-after-restart case without making users wait forever
/// if the backend is genuinely wedged.
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// Upper bound on characters of retrieved-doc context we'll paste into
/// the system prompt. ~4k chars ≈ ~1k tokens — leaves plenty of room
/// under a 2k-token context budget for the user's message + reply.
/// Retrieved chunks are trimmed (not dropped) to fit.
pub(crate) const MAX_RETRIEVAL_CHARS: usize = 4000;

/// Number of top-k chunks Herald pulls per turn. Three is enough to give
/// the model a shot at the right section + a fallback, without crowding
/// the system prompt.
pub(crate) const HERALD_RETRIEVAL_K: usize = 3;

/// Generic LLM-backed specialist. Holds a pre-built reqwest client so
/// keep-alive reuses the same TCP connection across Discord messages.
pub struct LlmSpecialist {
    name: Name,
    base_url: String,
    model_id: String,
    system_prompt: String,
    client: Client,
    /// Shared BM25 index over docs/wiki. `Arc` so one index can back
    /// every specialist that uses retrieval without cloning the corpus.
    /// `None` means retrieval is disabled for this instance.
    retrieval: Option<Arc<WikiIndex>>,
    /// Gate the retrieval path. Default `false` — existing callers that
    /// don't invoke `with_retrieval` get the bare-prompt behaviour they
    /// had before, so this change is backwards-compatible with
    /// Sentinel/Magistrate/Quartermaster wiring.
    use_retrieval: bool,
}

impl LlmSpecialist {
    /// Construct with explicit wiring. Callers pick the name, prompt,
    /// and server URL. Tests override `base_url`; production uses
    /// [`default_base_url`].
    pub fn new(
        name: Name,
        base_url: impl Into<String>,
        model_id: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        let client = Client::builder()
            .timeout(HTTP_TIMEOUT)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            name,
            base_url: base_url.into(),
            model_id: model_id.into(),
            system_prompt: system_prompt.into(),
            client,
            retrieval: None,
            use_retrieval: false,
        }
    }

    /// Attach a shared BM25 wiki index and enable retrieval-augmented
    /// prompting. Idempotent: calling twice just swaps the index. The
    /// specialist will query `top_k = HERALD_RETRIEVAL_K` against the
    /// user turn and paste the formatted chunks between the base system
    /// prompt and the role rules, capped at [`MAX_RETRIEVAL_CHARS`].
    ///
    /// Builder-style so callers can chain:
    /// `LlmSpecialist::herald(url, model).with_retrieval(idx)`.
    pub fn with_retrieval(mut self, idx: Arc<WikiIndex>) -> Self {
        self.retrieval = Some(idx);
        self.use_retrieval = true;
        self
    }

    /// Herald — conversational Q&A voice of 1bit.systems on Discord.
    pub fn herald(base_url: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self::new(Name::Herald, base_url, model_id, HERALD_PROMPT)
    }

    /// Sentinel — bug-triage specialist. Restates, asks for missing
    /// repro info, suggests the most likely known cause.
    pub fn sentinel(base_url: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self::new(Name::Sentinel, base_url, model_id, SENTINEL_PROMPT)
    }

    /// Magistrate — feature-request reviewer. Accept / defer / reject
    /// verdicts tied to the ternary-kernel roadmap.
    pub fn magistrate(base_url: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self::new(Name::Magistrate, base_url, model_id, MAGISTRATE_PROMPT)
    }

    /// Quartermaster — GitHub event triage (issue opened/closed,
    /// PR opened/merged, push).
    pub fn quartermaster(base_url: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self::new(
            Name::Quartermaster,
            base_url,
            model_id,
            QUARTERMASTER_PROMPT,
        )
    }

    /// Exposed for tests — confirm the wired prompt covers the role.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    fn down_text(&self) -> String {
        // Friendly triage ack. Used when the upstream LLM is unreachable
        // OR when its output is pathological (stuck in a fence-loop, etc).
        // Never returns empty — help-desk consumers rely on a non-empty
        // string to anchor the thread.
        match self.name {
            Name::Sentinel => {
                "Sentinel ack — triaged as bug. A human will follow up here shortly.".to_string()
            }
            Name::Magistrate => {
                "Magistrate ack — logged as feature request. Review follows.".to_string()
            }
            Name::Herald => {
                "Herald ack — your question is in the queue. Human follow-up shortly.".to_string()
            }
            Name::Quartermaster => "Quartermaster ack — event logged.".to_string(),
            _ => format!("{} ack — handoff to human.", self.name.as_str()),
        }
    }

    /// Detect LLM output that looks pathological — stuck in a code-fence
    /// repetition loop, near-empty, or otherwise unusable on Discord.
    /// Returns true if the caller should fall back to `down_text()`.
    ///
    /// Heuristics:
    ///   * more `` ``` `` fence lines than content lines → fence loop
    ///   * total printable content < 16 chars → stub-ish
    ///   * same line repeated >= 5× in a row → loop
    fn looks_pathological(text: &str) -> bool {
        let lines: Vec<&str> = text.lines().collect();
        let fences = lines.iter().filter(|l| l.trim().starts_with("```")).count();
        let content_lines = lines
            .iter()
            .filter(|l| !l.trim().is_empty() && !l.trim().starts_with("```"))
            .count();
        if fences >= 3 && fences > content_lines {
            return true;
        }
        let printable: String = text
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '`')
            .collect();
        if printable.chars().count() < 16 {
            return true;
        }
        // Consecutive repeat detection.
        let mut run = 1;
        for w in lines.windows(2) {
            if !w[0].trim().is_empty() && w[0].trim() == w[1].trim() {
                run += 1;
                if run >= 5 {
                    return true;
                }
            } else {
                run = 1;
            }
        }
        false
    }

    fn completions_url(&self) -> String {
        format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        )
    }

    /// Assemble the system prompt for this turn. When retrieval is
    /// disabled (or the index yields zero hits) this returns the base
    /// `system_prompt` unchanged — guaranteeing the bare-Herald path
    /// stays byte-identical to the pre-retrieval behaviour.
    ///
    /// When retrieval IS enabled and the query hits, the formatted
    /// chunks are prepended to the base prompt and the whole thing is
    /// trimmed to [`MAX_RETRIEVAL_CHARS`] + `system_prompt.len()` at
    /// worst. We keep the retrieval block first so the role rules (end
    /// of `system_prompt`) are the last thing the model reads, which
    /// empirically wins over role-first ordering on small models.
    pub(crate) fn build_system_prompt(&self, user: &str) -> String {
        if !self.use_retrieval {
            return self.system_prompt.clone();
        }
        let Some(idx) = self.retrieval.as_ref() else {
            return self.system_prompt.clone();
        };
        let hits = idx.top_k(user, HERALD_RETRIEVAL_K);
        if hits.is_empty() {
            return self.system_prompt.clone();
        }
        let mut block = format_for_system_prompt(&hits);
        if block.len() > MAX_RETRIEVAL_CHARS {
            // Trim to a char boundary — `truncate` panics if it lands
            // mid-codepoint. Walking backwards to the nearest boundary
            // is cheap (<=3 bytes).
            let mut cap = MAX_RETRIEVAL_CHARS;
            while cap > 0 && !block.is_char_boundary(cap) {
                cap -= 1;
            }
            block.truncate(cap);
            block.push_str("\n[...retrieval truncated]\n");
        }
        // Keep a visible separator so the model can tell context from rules.
        format!("{block}\n---\n{}", self.system_prompt)
    }

    async fn call_llm(&self, user_content: &str) -> Result<String, String> {
        let system_prompt = self.build_system_prompt(user_content);
        let body = json!({
            "model": self.model_id,
            "temperature": 0.3,
            "max_tokens": 400,
            "stream": false,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        });
        let resp = self
            .client
            .post(self.completions_url())
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("http send: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("http status: {}", resp.status()));
        }
        let v: Value = resp.json().await.map_err(|e| format!("json decode: {e}"))?;
        extract_content(&v).ok_or_else(|| "no choices[0].message.content".to_string())
    }
}

/// Pull `choices[0].message.content` out of an OpenAI-shaped response.
/// Exposed `pub(crate)` so tests can pin the parse logic without HTTP.
pub(crate) fn extract_content(v: &Value) -> Option<String> {
    v.get("choices")?
        .get(0)?
        .get("message")?
        .get("content")?
        .as_str()
        .map(|s| s.to_string())
}

/// Pull the user-facing content out of a dispatch request. Discord
/// populates `content`; GitHub payloads may not, so we fall back to
/// the whole JSON serialised as a string — Quartermaster's prompt
/// explicitly says "given a single GitHub event JSON".
fn extract_user_content(req: &Value) -> String {
    if let Some(s) = req.get("content").and_then(|v| v.as_str())
        && !s.trim().is_empty()
    {
        return s.to_string();
    }
    // No bare `content` string — serialise the full payload. Used by
    // Quartermaster for GitHub event JSON.
    serde_json::to_string(req).unwrap_or_default()
}

#[async_trait]
impl Specialist for LlmSpecialist {
    fn name(&self) -> Name {
        self.name
    }

    fn description(&self) -> &'static str {
        // Static description — each specialist wires its own role via
        // the prompt, but MCP clients see a generic line here. The
        // load-bearing role text is in `system_prompt`.
        "LLM-backed specialist — calls 1bit-halo-server /v1/chat/completions"
    }

    async fn handle(&self, req: Value) -> Result<Value> {
        let user = extract_user_content(&req);
        let specialist = self.name.as_str();
        let started = Instant::now();
        let result = self.call_llm(&user).await;
        let elapsed_ms = started.elapsed().as_millis();
        match result {
            Ok(text) if !Self::looks_pathological(&text) => {
                info!(
                    specialist,
                    bytes = text.len(),
                    elapsed_ms = elapsed_ms as u64,
                    "LLM reply ok"
                );
                Ok(json!({ "text": text }))
            }
            Ok(bad) => {
                // Model emitted something but it's unusable (fence loop,
                // near-empty, repeat spam). Fall through to a friendly
                // triage ack so the help-desk thread is still populated
                // and a human can take over.
                warn!(
                    specialist,
                    bytes = bad.len(),
                    elapsed_ms = elapsed_ms as u64,
                    "LLM output looks pathological — falling back to down_text"
                );
                Ok(json!({
                    "error": "pathological LLM output",
                    "text": self.down_text(),
                    "raw_truncated": bad.chars().take(200).collect::<String>(),
                }))
            }
            Err(e) => {
                // Graceful degradation: ALWAYS return a `text` field so
                // echo still has something to post. The `error` field
                // is surfaced alongside for operator logs.
                warn!(
                    specialist,
                    error = %e,
                    elapsed_ms = elapsed_ms as u64,
                    "LLM call failed — falling back to down_text"
                );
                Ok(json!({
                    "error": e,
                    "text": self.down_text(),
                }))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// System prompts. Kept as module-level `const` so tests can assert against
// them and operators can grep for exact wording.

/// Shared output-format rule appended to every specialist prompt. The
/// watch-discord binary strips the outer fence before posting so the
/// message renders as clean prose on Discord. The fence forces the LLM to
/// produce one self-contained block with no editorial padding before or
/// after. Commands or stack traces INSIDE the response should still use
/// inner fenced blocks; those are preserved.
// OUTPUT_FORMAT_RULE was retired 2026-04-23 after the Sparse-BitNet
// Run 3 checkpoint proved too weak at "wrap ENTIRE response in one
// fenced block" — it routinely echoed the fence header + a prompt
// fragment and nothing else, tripping `looks_pathological` and
// collapsing every specialist reply to the `down_text` stub.
// Unwrap + sanitize on the Discord side handle bare prose and
// incidental inner fences just fine, so we let the model speak
// plainly and stop forcing a shape it can't produce reliably.
pub const OUTPUT_FORMAT_RULE: &str = " Write the answer as plain prose. Use triple-backtick fenced \
     blocks only for literal shell commands, file paths, or code \
     samples (```bash ... ``` renders as code on Discord). Do not \
     wrap the whole reply in a single code block.";

pub const HERALD_PROMPT: &str = concat!(
    "You are Herald, the conversational voice of 1bit.systems on Discord. \
Answer the user's question concisely in 1-3 short paragraphs. If the answer requires specifics you \
don't have (benchmark numbers, internal paths, ROCm versions), ask ONE follow-up question. Tone: \
calm technical, no emoji, no marketing, no exclamations, no movie quotes. If the question isn't \
about 1bit.systems, say so and redirect to #water-cooler.",
    " Write the answer as plain prose on Discord. Use triple-backtick \
fenced blocks only for literal shell commands, file paths, or code \
samples; never wrap the whole reply in one fenced block."
);

pub const SENTINEL_PROMPT: &str = concat!(
    "You are Sentinel, bug-triage specialist for 1bit.systems. The user \
reported an issue. Reply with a copy-paste-ready triage playbook: \
(1) one-line restatement of the bug, (2) the single most likely cause \
from the known-issue list (amdgpu OPTC hang on kernel 7.x, mlock \
missing, RoPE convention mismatch, ROCm gfx1151 Tier-1 absent, Caddy \
bearer missing, KV-cache mutex contention), (3) concrete shell \
commands the user can run right now to diagnose or fix — journalctl \
greps, systemctl commands, sysfs pokes, a one-line sed/patch, or a \
specific build flag, (4) explicit close asking the user to paste the \
output of those commands back in this thread so a human can follow \
up. Mention any missing context you'd need (kernel version, ROCm \
version, commit SHA, minimal repro) inside that close. Output \
<= 400 words. Calm technical tone. Actionable > exhaustive.",
    " Plain prose on Discord. Wrap every shell command, file path, \
systemctl invocation, or patch snippet in a fenced block with a \
language tag (```bash ... ```, ```diff ... ```); real runnable \
commands are the load-bearing output here. Do not wrap the whole \
reply in a single fenced block — only the command snippets."
);

pub const MAGISTRATE_PROMPT: &str = concat!(
    "You are Magistrate, feature-request reviewer for 1bit.systems. \
The user proposed a feature or change. Output: (1) restatement in one line, (2) ranked verdict \
[accept, defer, reject] with one-sentence reasoning tied to roadmap priorities (ternary kernels on \
gfx1151 = primary, gfx1201 port = second, Sparse-BitNet retrain = in flight, NPU = deferred until \
XDNA2 unblocks), (3) if accept, suggest where in the codebase it lands. <= 300 words. Calm \
technical tone.",
    " Write the answer as plain prose on Discord. Use triple-backtick \
fenced blocks only for literal shell commands, file paths, or code \
samples; never wrap the whole reply in one fenced block."
);

pub const QUARTERMASTER_PROMPT: &str = concat!(
    "You are Quartermaster, GitHub event triage for 1bit.systems. \
Given a single GitHub event JSON (issue opened/closed, PR opened/merged, push), output a one-line \
human summary of what changed and whether it needs operator attention. <= 60 words. No preamble.",
    " Write the answer as plain prose on Discord. Use triple-backtick \
fenced blocks only for literal shell commands, file paths, or code \
samples; never wrap the whole reply in one fenced block."
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener as StdListener;

    /// One-shot TCP mock. Accepts a single connection, reads the
    /// request bytes, writes a canned 200 response. Returns the
    /// base URL (no path) so callers build `{base}/v1/chat/completions`.
    fn spawn_mock(canned_body: &'static str) -> String {
        let listener = StdListener::bind("127.0.0.1:0").expect("bind mock");
        let addr = listener.local_addr().expect("addr").to_string();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let _ = stream.read(&mut buf);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    canned_body.len(),
                    canned_body
                );
                let _ = stream.write_all(resp.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn herald_returns_text_field_from_mock_completion() {
        // Happy-path: mock returns a synthetic OpenAI completion; the
        // specialist's handle() output must carry `"text"` verbatim.
        let canned =
            r#"{"choices":[{"message":{"content":"Herald here: bench is 83 tok/s on gfx1151."}}]}"#;
        let base = spawn_mock(canned);

        let herald = LlmSpecialist::herald(base, default_model_id());
        let out = herald
            .handle(json!({
                "source": "discord",
                "content": "what's the current tok/s?",
            }))
            .await
            .unwrap();
        assert_eq!(
            out["text"], "Herald here: bench is 83 tok/s on gfx1151.",
            "text field must echo the mock completion, got {out}"
        );
        // Happy path: no `error` key.
        assert!(
            out.get("error").is_none(),
            "no error expected on happy path: {out}"
        );
    }

    #[tokio::test]
    async fn network_error_returns_graceful_down_text() {
        // Port 1 is reserved and refuses connections in practice.
        // Specialist must still return a Value with a `"text"` field
        // containing "down" so echo posts something readable.
        let sentinel =
            LlmSpecialist::sentinel("http://127.0.0.1:1".to_string(), default_model_id());
        let out = sentinel
            .handle(json!({
                "source": "discord",
                "content": "panic in decode loop",
            }))
            .await
            .unwrap();
        let text = out["text"].as_str().expect("text field must be a string");
        // Down-text shape changed 2026-04-23 to a friendly triage ack —
        // what matters here is that it's non-empty so help-desk
        // consumers have something to anchor the thread with.
        assert!(
            !text.trim().is_empty(),
            "down_text must not be empty on network failure, got: {text:?}"
        );
        assert!(
            text.to_lowercase().contains("sentinel"),
            "sentinel down_text should name the specialist, got: {text:?}"
        );
        // Error reason should be surfaced for operator logs.
        assert!(
            out.get("error").is_some(),
            "error field should be populated on network failure: {out}"
        );
    }

    #[test]
    fn system_prompts_wire_name_and_role_keyword_through() {
        // Basic construction sanity. Each factory produces an
        // LlmSpecialist whose `system_prompt` mentions both the
        // specialist's own name and a role-identifying keyword — so
        // a typo in factory wiring can't silently ship a blank prompt.
        let h = LlmSpecialist::herald("http://x", "m");
        assert_eq!(h.name(), Name::Herald);
        assert!(h.system_prompt().contains("Herald"));
        assert!(h.system_prompt().to_lowercase().contains("discord"));

        let s = LlmSpecialist::sentinel("http://x", "m");
        assert_eq!(s.name(), Name::Sentinel);
        assert!(s.system_prompt().contains("Sentinel"));
        assert!(s.system_prompt().to_lowercase().contains("bug"));

        let m = LlmSpecialist::magistrate("http://x", "m");
        assert_eq!(m.name(), Name::Magistrate);
        assert!(m.system_prompt().contains("Magistrate"));
        assert!(m.system_prompt().to_lowercase().contains("feature"));

        let q = LlmSpecialist::quartermaster("http://x", "m");
        assert_eq!(q.name(), Name::Quartermaster);
        assert!(q.system_prompt().contains("Quartermaster"));
        assert!(q.system_prompt().to_lowercase().contains("github"));
    }

    #[test]
    fn extract_user_content_prefers_content_field_then_falls_back_to_json() {
        // Discord messages carry `content`; GitHub payloads may not.
        let disc = json!({"content": "hello", "author": "alice"});
        assert_eq!(extract_user_content(&disc), "hello");

        let gh = json!({"event": "pull_request", "action": "opened", "number": 7});
        let got = extract_user_content(&gh);
        assert!(got.contains("pull_request"));
        assert!(got.contains("opened"));
    }

    #[test]
    fn extract_content_parses_openai_shape() {
        let v = json!({"choices":[{"message":{"content":"ok"}}]});
        assert_eq!(extract_content(&v).as_deref(), Some("ok"));

        let missing = json!({"choices":[]});
        assert!(extract_content(&missing).is_none());
    }

    // ---- retrieval-augmented prompt tests ----
    //
    // The retrieval layer is opt-in via `with_retrieval(Arc<WikiIndex>)`;
    // until it's attached the bare-Herald path must stay byte-identical.
    // These tests cover the three behavioural states: (1) retrieval off
    // → bare prompt, (2) retrieval on with hits → chunks injected,
    // (3) retrieval on with no hits → bare prompt fallback.

    #[test]
    fn build_system_prompt_bare_herald_matches_system_prompt() {
        // Regression: when no retrieval index is attached,
        // `build_system_prompt` must return the base prompt verbatim.
        // This guarantees Sentinel/Magistrate/Quartermaster (which never
        // call `with_retrieval`) keep their exact pre-change behaviour.
        let h = LlmSpecialist::herald("http://x", "m");
        let got = h.build_system_prompt("anything user types here");
        assert_eq!(
            got,
            h.system_prompt(),
            "bare Herald must return the unmodified system_prompt"
        );
    }

    #[test]
    fn build_system_prompt_injects_retrieval_chunks_when_hits_present() {
        // Fixture: tempdir wiki with one markdown file that clearly
        // matches a query term. Index should find it, and the Herald
        // system prompt should grow to include the formatted chunk
        // header + file path, and the original prompt tail stays intact.
        use std::fs;
        use std::sync::Arc;
        let tmp = tempfile::tempdir().expect("tempdir");
        let wiki_path = tmp.path();
        fs::write(
            wiki_path.join("gfx1151.md"),
            "# gfx1151 troubleshooting\n\nThe amdgpu OPTC hang is kernel 7.x specific.\n",
        )
        .unwrap();
        let idx = WikiIndex::load(wiki_path).expect("index loads");
        assert!(!idx.is_empty(), "fixture must produce at least one chunk");

        let h = LlmSpecialist::herald("http://x", "m").with_retrieval(Arc::new(idx));
        let prompt = h.build_system_prompt("amdgpu OPTC hang");
        assert!(
            prompt.starts_with("RELEVANT DOCS"),
            "retrieval block should lead the system prompt, got: {prompt:.120}"
        );
        assert!(
            prompt.contains("gfx1151.md"),
            "prompt should cite the fixture file, got: {prompt}"
        );
        // Role rules (tail of HERALD_PROMPT) must still be present.
        assert!(
            prompt.contains("Herald"),
            "base Herald role text must survive injection"
        );
        // And must be STRICTLY longer than the bare prompt.
        assert!(
            prompt.len() > h.system_prompt().len(),
            "augmented prompt must be longer than base"
        );
    }

    #[test]
    fn build_system_prompt_empty_hits_falls_back_to_base_prompt() {
        // If retrieval is wired but the query matches nothing in the
        // index, the specialist must NOT inject an empty
        // "RELEVANT DOCS (top 0):" header — it should silently return
        // the base prompt so the LLM isn't primed with "here are docs"
        // followed by nothing.
        use std::fs;
        use std::sync::Arc;
        let tmp = tempfile::tempdir().expect("tempdir");
        let wiki_path = tmp.path();
        fs::write(
            wiki_path.join("unrelated.md"),
            "# Unrelated\n\nThis file is about cooking recipes only.\n",
        )
        .unwrap();
        let idx = WikiIndex::load(wiki_path).expect("index loads");

        let h = LlmSpecialist::herald("http://x", "m").with_retrieval(Arc::new(idx));
        // Query uses terms that don't appear in the fixture at all.
        let prompt = h.build_system_prompt("xyzzy_never_appears_anywhere_in_corpus");
        assert_eq!(
            prompt,
            h.system_prompt(),
            "empty hits must fall back to bare prompt, got augmented: {prompt}"
        );
    }
}
