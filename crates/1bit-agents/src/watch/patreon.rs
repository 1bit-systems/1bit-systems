//! Patreon watcher plumbing used by the `1bit-watch-patreon` binary.
//!
//! Pure, offline logic: Patreon API response parsing, snapshot diff,
//! state-file I/O. The binary layers `reqwest` + pagination + `Registry`
//! dispatch + Discord posting on top; everything here is unit-testable
//! with no network, no process env, no clock.
//!
//! The Patreon v2 `members` payload looks like:
//!
//! ```json
//! {
//!   "data": [
//!     { "id": "abc", "type": "member",
//!       "attributes": {
//!         "full_name": "Jane",
//!         "email": "jane@example.com",
//!         "patron_status": "active_patron",
//!         "currently_entitled_amount_cents": 500
//!       }
//!     },
//!     ...
//!   ],
//!   "links": { "next": "https://.../members?page%5Bcursor%5D=eyJ..." },
//!   "meta":  { "pagination": { "cursors": { "next": "eyJ..." } } }
//! }
//! ```
//!
//! We diff against a snapshot file written on the previous run. The
//! snapshot is the flattened [`Member`] list — not the raw API payload —
//! so its shape is stable across Patreon API drifts.
//!
//! First-run behaviour: when the state file does not exist, we treat the
//! prior snapshot as empty but STILL write the current snapshot out.
//! The binary skips the Discord announcement on a first-run seed to
//! avoid announcing every existing patron. Callers distinguish via
//! [`DiffOutcome::first_run`].

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Default Patreon campaign id. Kept as a `&str` so the parser + binary
/// use the same constant and tests can reference it without duplication.
/// Sourced from memory `project_patreon_live.md` (2026-04-22).
pub const DEFAULT_CAMPAIGN_ID: &str = "15895969";

/// Default Discord channel name the welcome post targets. The channel
/// ID itself must come from `HALO_DISCORD_ANNOUNCE_CHANNEL` because
/// Discord's API needs a numeric ID, not a name.
pub const DEFAULT_ANNOUNCE_CHANNEL: &str = "announcements";

/// Flattened member record. What we persist, what we diff, what we hand
/// off to Quartermaster. Matches the `fields[member]=...` subset we
/// request in the pagination loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Member {
    // `id` lives at the top level of a Patreon member entry, not inside
    // `attributes`. We default to empty here so `parse_members_page`
    // can deserialize the `attributes` subtree into `Member` and then
    // assign the real id after. Snapshots always have `id` populated
    // by the time they hit disk.
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub full_name: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    /// Patreon's `patron_status` literal — one of `active_patron`,
    /// `declined_patron`, `former_patron`, or occasionally `null` for
    /// free/following members. We store the raw string so the diff
    /// code can't silently collapse future states into a catch-all.
    #[serde(default)]
    pub patron_status: Option<String>,
    #[serde(default)]
    pub currently_entitled_amount_cents: Option<i64>,
}

impl Member {
    /// True if Patreon considers this member an active, paying patron.
    /// Declined / former / missing statuses all return `false`.
    pub fn is_active(&self) -> bool {
        matches!(self.patron_status.as_deref(), Some("active_patron"))
    }
}

/// On-disk snapshot. Versioned so we can change the member record shape
/// later without silently mis-parsing old files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Format version. Bump when `Member` gains a non-optional field.
    pub version: u32,
    pub members: Vec<Member>,
}

impl Snapshot {
    pub const CURRENT_VERSION: u32 = 1;

    pub fn new(members: Vec<Member>) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            members,
        }
    }
}

/// Result of comparing a freshly-fetched member list against the prior
/// snapshot.
///
/// `first_run` is `true` when the prior snapshot was absent — callers
/// should typically seed the state file but skip the welcome post,
/// since announcing every pre-existing patron on day one is rude.
#[derive(Debug, Clone)]
pub struct DiffOutcome {
    /// Members newly-active since the previous snapshot.
    pub new_patrons: Vec<Member>,
    /// True iff the diff ran against an empty prior (no state file).
    pub first_run: bool,
}

/// Compute the diff between a previous snapshot and the current member
/// list. A "new patron" is a member who is currently `active_patron`
/// AND whose id was not in the previous snapshot (at all — regardless
/// of that prior record's status).
///
/// The converse is deliberate: a prior `active_patron` who is now
/// `declined_patron` does NOT show up. Transitions OUT of active are
/// not announcements; they're churn, handled elsewhere.
///
/// Stability: the result preserves the order of `current`, so the
/// binary's logging + Discord post order matches the API's natural
/// order (Patreon sorts newest-first, which reads sensibly).
pub fn diff_new_patrons(previous: Option<&Snapshot>, current: &[Member]) -> DiffOutcome {
    let first_run = previous.is_none();
    let known: HashSet<&str> = previous
        .map(|s| s.members.iter().map(|m| m.id.as_str()).collect())
        .unwrap_or_default();
    let new_patrons = current
        .iter()
        .filter(|m| m.is_active() && !known.contains(m.id.as_str()))
        .cloned()
        .collect();
    DiffOutcome {
        new_patrons,
        first_run,
    }
}

/// Parse one page of `GET /campaigns/{id}/members` into (members, next_cursor).
///
/// `next_cursor` is taken from `links.next` if present, else from
/// `meta.pagination.cursors.next`. We return the raw cursor string
/// (the opaque token Patreon hands back) rather than the full URL —
/// the caller re-constructs the URL with the documented
/// `page[cursor]=...` param. Using the raw cursor avoids parsing and
/// round-tripping the `links.next` URL, which occasionally carries
/// host changes (www vs api) between Patreon infra generations.
pub fn parse_members_page(page: &serde_json::Value) -> Result<(Vec<Member>, Option<String>)> {
    let data = page
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow::anyhow!("members page missing `data` array"))?;

    let mut out = Vec::with_capacity(data.len());
    for entry in data {
        let id = entry
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("member entry missing `id`"))?
            .to_string();
        let attrs = entry.get("attributes").cloned().unwrap_or_default();
        // Build via serde so optional-field handling matches Member's derive.
        let mut member: Member =
            serde_json::from_value(attrs).context("member attributes failed to deserialize")?;
        member.id = id;
        out.push(member);
    }

    // Prefer the explicit cursor in meta.pagination.cursors.next —
    // that's the value Patreon's docs recommend threading back through.
    // Fall back to parsing ?page[cursor]= out of links.next for older
    // response shapes that only populated the HATEOAS link.
    let next = page
        .pointer("/meta/pagination/cursors/next")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            page.pointer("/links/next")
                .and_then(|v| v.as_str())
                .and_then(extract_cursor_from_link)
        });

    Ok((out, next))
}

/// Pull the `page[cursor]=...` value out of a `links.next` URL.
/// Returns `None` if the URL has no cursor param (last page) or is
/// malformed enough to break our lightweight parser. We deliberately
/// avoid a full URL-parser dep; the param shape is stable.
fn extract_cursor_from_link(url: &str) -> Option<String> {
    // The cursor param can be URL-encoded (`page%5Bcursor%5D=`) or
    // raw (`page[cursor]=`) depending on the infra generation that
    // serves the response. Handle both.
    const ENCODED: &str = "page%5Bcursor%5D=";
    const RAW: &str = "page[cursor]=";
    let marker_start = url
        .find(ENCODED)
        .map(|i| i + ENCODED.len())
        .or_else(|| url.find(RAW).map(|i| i + RAW.len()))?;
    let tail = &url[marker_start..];
    let end = tail.find('&').unwrap_or(tail.len());
    let raw = &tail[..end];
    if raw.is_empty() {
        None
    } else {
        Some(percent_decode(raw))
    }
}

/// Minimal percent decoder — just enough for the cursor. Patreon's
/// cursor is base64-y so the only likely encoded bytes are `=` and
/// `+`. We decode every `%XX` triple we see; anything we can't decode
/// is passed through verbatim (garbage-in survives garbage-out).
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
            if let Some(h) = hex {
                if let Ok(b) = u8::from_str_radix(h, 16) {
                    out.push(b);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|e| {
        // Invalid UTF-8 after decode — fall back to original bytes.
        String::from_utf8_lossy(&e.into_bytes()).into_owned()
    })
}

/// Resolve the snapshot path: env override first, then
/// `~/.local/state/1bit-halo/patreon-members.json`. Returns a
/// `PathBuf` even when the directory doesn't exist yet — the caller
/// creates it lazily at write time.
pub fn state_path_from_env() -> PathBuf {
    if let Ok(v) = std::env::var("HALO_PATREON_STATE_FILE") {
        if !v.trim().is_empty() {
            return PathBuf::from(v);
        }
    }
    let base = dirs::state_dir()
        .or_else(dirs::data_local_dir)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join("1bit-halo").join("patreon-members.json")
}

/// Read a snapshot from disk. Returns `Ok(None)` if the file doesn't
/// exist (first-run path); bubbles any other error (perms, corrupt
/// JSON, wrong schema version).
pub fn read_snapshot(path: &Path) -> Result<Option<Snapshot>> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let snap: Snapshot = serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing snapshot at {}", path.display()))?;
            Ok(Some(snap))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => {
            Err(anyhow::Error::new(e).context(format!("reading snapshot at {}", path.display())))
        }
    }
}

/// Write a snapshot atomically: serialise to `<path>.tmp`, `sync_all`,
/// then `rename`. A crash mid-write leaves the prior snapshot intact,
/// which is the correct failure mode — partial state would lose track
/// of existing patrons and re-announce them on the next poll.
pub fn write_snapshot(path: &Path, snap: &Snapshot) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(snap).context("serialising snapshot")?;
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp)
            .with_context(|| format!("opening {} for write", tmp.display()))?;
        f.write_all(&bytes)
            .with_context(|| format!("writing {}", tmp.display()))?;
        f.sync_all().ok(); // best-effort durability; dev-box ENOSPC is fine to swallow
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_member(id: &str, name: &str, status: Option<&str>) -> Member {
        Member {
            id: id.to_string(),
            full_name: Some(name.to_string()),
            email: Some(format!("{id}@example.invalid")),
            patron_status: status.map(|s| s.to_string()),
            currently_entitled_amount_cents: Some(500),
        }
    }

    #[test]
    fn diff_identifies_new_active_patrons() {
        // Prior snapshot has one member (B); current has B and a new
        // active A. Diff should surface only A.
        let prev = Snapshot::new(vec![mk_member("B", "Bee", Some("active_patron"))]);
        let current = vec![
            mk_member("A", "Alice", Some("active_patron")),
            mk_member("B", "Bee", Some("active_patron")),
        ];
        let out = diff_new_patrons(Some(&prev), &current);
        assert!(!out.first_run);
        assert_eq!(out.new_patrons.len(), 1);
        assert_eq!(out.new_patrons[0].id, "A");
    }

    #[test]
    fn diff_ignores_declined_or_former() {
        // Member A was active last run, now declined. This is not a
        // new-patron event; diff should be empty, not surface A as
        // "new" just because the status key changed.
        let prev = Snapshot::new(vec![mk_member("A", "Alice", Some("active_patron"))]);
        let current = vec![mk_member("A", "Alice", Some("declined_patron"))];
        let out = diff_new_patrons(Some(&prev), &current);
        assert!(
            out.new_patrons.is_empty(),
            "declined transitions must not announce: {:?}",
            out.new_patrons
        );

        // And a former_patron likewise — even one that wasn't in
        // the prior snapshot — is not "new". `is_active` gates on
        // `active_patron` specifically.
        let out2 = diff_new_patrons(Some(&prev), &[mk_member("C", "Cee", Some("former_patron"))]);
        assert!(out2.new_patrons.is_empty());
    }

    #[test]
    fn state_file_round_trips() {
        // Write a snapshot, read it back, assert equality. Uses a
        // tempdir so we don't race with the real state location.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sub").join("snap.json");
        let snap = Snapshot::new(vec![
            mk_member("A", "Alice", Some("active_patron")),
            mk_member("B", "Bee", Some("declined_patron")),
        ]);
        write_snapshot(&path, &snap).unwrap();
        assert!(path.exists(), "write should have created the file");
        let got = read_snapshot(&path).unwrap().unwrap();
        assert_eq!(got.version, Snapshot::CURRENT_VERSION);
        assert_eq!(got.members, snap.members);
    }

    #[test]
    fn empty_previous_snapshot_is_handled() {
        // No prior file → read_snapshot returns None → diff runs
        // against an empty set. EVERY active patron comes out as
        // "new". `first_run` is `true` so the binary can skip the
        // Discord post on the seed run (documented behaviour, see
        // module docs).
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("does-not-exist.json");
        let prior = read_snapshot(&missing).unwrap();
        assert!(prior.is_none());

        let current = vec![
            mk_member("A", "Alice", Some("active_patron")),
            mk_member("B", "Bee", Some("declined_patron")),
            mk_member("C", "Cee", Some("active_patron")),
        ];
        let out = diff_new_patrons(prior.as_ref(), &current);
        assert!(out.first_run, "missing file must produce first_run=true");
        // Active A and C, but not declined B.
        assert_eq!(out.new_patrons.len(), 2);
        let ids: Vec<&str> = out.new_patrons.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"A") && ids.contains(&"C"));
    }

    #[test]
    fn parse_members_page_extracts_attributes_and_cursor() {
        // The Patreon shape: `id` outside `attributes`, the rest
        // under `attributes`. Cursor lives at meta.pagination.cursors.next.
        let page = serde_json::json!({
            "data": [
                { "id": "m1", "type": "member", "attributes": {
                    "full_name": "First Patron",
                    "email": "first@example.com",
                    "patron_status": "active_patron",
                    "currently_entitled_amount_cents": 1000
                }},
                { "id": "m2", "type": "member", "attributes": {
                    "full_name": "Declined",
                    "patron_status": "declined_patron"
                }}
            ],
            "meta": { "pagination": { "cursors": { "next": "eyJwYWdlIjoyfQ==" } } }
        });
        let (members, next) = parse_members_page(&page).unwrap();
        assert_eq!(members.len(), 2);
        assert_eq!(members[0].id, "m1");
        assert_eq!(members[0].full_name.as_deref(), Some("First Patron"));
        assert!(members[0].is_active());
        assert!(!members[1].is_active());
        assert_eq!(next.as_deref(), Some("eyJwYWdlIjoyfQ=="));
    }

    #[test]
    fn parse_members_page_falls_back_to_links_next() {
        // Older shape: only links.next is populated, meta cursors are
        // absent. Extract from the URL-encoded ?page[cursor]= param.
        let page = serde_json::json!({
            "data": [
                { "id": "m1", "attributes": { "patron_status": "active_patron" } }
            ],
            "links": {
                "next": "https://www.patreon.com/api/oauth2/v2/members?fields%5Bmember%5D=email&page%5Bcursor%5D=ZnV0dXJl&extra=x"
            }
        });
        let (_members, next) = parse_members_page(&page).unwrap();
        assert_eq!(next.as_deref(), Some("ZnV0dXJl"));
    }

    #[test]
    fn parse_members_page_last_page_has_no_cursor() {
        let page = serde_json::json!({
            "data": [{ "id": "m1", "attributes": { "patron_status": "active_patron" } }],
            "meta": { "pagination": { "cursors": { "next": null } } }
        });
        let (_members, next) = parse_members_page(&page).unwrap();
        assert!(next.is_none(), "null cursor must come through as None");
    }
}
