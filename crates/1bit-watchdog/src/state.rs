// Persistent state for 1bit-watchdog.
//
// Per-watch-entry we record:
//   last_seen_sha     — what we saw this poll
//   first_seen_at     — when we first observed a new SHA different from last_merged
//   last_merged_sha   — the SHA we propagated most recently
//   last_merged_at    — timestamp of that merge
//
// On each poll we compare the freshly-fetched SHA against last_merged_sha.
// If different and first_seen_at is unset, we set it to now. If it's set
// and (now - first_seen_at) >= soak_hours, we trigger on_merge and record
// a new last_merged.

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntryState {
    pub last_seen_sha:   Option<String>,
    pub first_seen_at:   Option<DateTime<Utc>>,
    pub last_merged_sha: Option<String>,
    pub last_merged_at:  Option<DateTime<Utc>>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct State {
    pub entries: BTreeMap<String, EntryState>,
}

#[derive(Debug)]
pub enum Transition {
    NoChange,
    SeenNew,
    Soaking { remaining_hours: i64 },
    SoakComplete,
}

pub fn default_path() -> PathBuf {
    let xdg = std::env::var("XDG_STATE_HOME")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home).join(".local/state")
        });
    xdg.join("1bit-watchdog").join("state.json")
}

impl State {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let s: State = serde_json::from_str(&raw).context("parsing state.json")?;
        Ok(s)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let raw = serde_json::to_string_pretty(self)?;
        fs::write(path, raw).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    pub fn entries(&self) -> &BTreeMap<String, EntryState> {
        &self.entries
    }

    pub fn reset(&mut self, id: &str) {
        if let Some(e) = self.entries.get_mut(id) {
            e.first_seen_at = None;
            e.last_seen_sha = None;
        }
    }

    pub fn mark_merged(&mut self, id: &str, now: DateTime<Utc>) {
        let e = self.entries.entry(id.to_string()).or_default();
        e.last_merged_sha = e.last_seen_sha.clone();
        e.last_merged_at = Some(now);
        e.first_seen_at = None;
    }

    pub fn observe(&mut self, id: &str, latest: &str, soak_hours: u32) -> Transition {
        let now = Utc::now();
        let e = self.entries.entry(id.to_string()).or_default();
        e.last_seen_sha = Some(latest.to_string());

        match &e.last_merged_sha {
            Some(merged) if merged == latest => {
                e.first_seen_at = None;
                Transition::NoChange
            }
            _ => match e.first_seen_at {
                None => {
                    e.first_seen_at = Some(now);
                    Transition::SeenNew
                }
                Some(t0) => {
                    let dwell = now - t0;
                    let target = Duration::hours(soak_hours as i64);
                    if dwell >= target {
                        Transition::SoakComplete
                    } else {
                        Transition::Soaking {
                            remaining_hours: (target - dwell).num_hours().max(0),
                        }
                    }
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn no_change_when_sha_matches_merged() {
        let mut s = State::default();
        s.entries.insert(
            "x".into(),
            EntryState {
                last_merged_sha: Some("abc".into()),
                ..Default::default()
            },
        );
        assert!(matches!(s.observe("x", "abc", 24), Transition::NoChange));
    }

    #[test]
    fn seen_new_on_first_divergence() {
        let mut s = State::default();
        assert!(matches!(s.observe("x", "abc", 24), Transition::SeenNew));
    }

    #[test]
    fn soaking_before_dwell_elapses() {
        let mut s = State::default();
        s.observe("x", "abc", 24);
        // Second observe of same sha within dwell window returns Soaking
        let t = s.observe("x", "abc", 24);
        assert!(matches!(t, Transition::Soaking { .. }));
    }

    #[test]
    fn soak_complete_after_window() {
        let mut s = State::default();
        s.entries.insert(
            "x".into(),
            EntryState {
                first_seen_at: Some(Utc::now() - Duration::hours(25)),
                last_seen_sha: Some("abc".into()),
                ..Default::default()
            },
        );
        assert!(matches!(s.observe("x", "abc", 24), Transition::SoakComplete));
    }

    #[test]
    fn reset_clears_dwell_clock() {
        let mut s = State::default();
        s.observe("x", "abc", 24);
        s.reset("x");
        assert!(matches!(s.observe("x", "abc", 24), Transition::SeenNew));
    }

    #[test]
    fn mark_merged_records_sha() {
        let mut s = State::default();
        s.observe("x", "abc", 24);
        s.mark_merged("x", Utc::now());
        let e = s.entries.get("x").unwrap();
        assert_eq!(e.last_merged_sha.as_deref(), Some("abc"));
        assert!(e.first_seen_at.is_none());
    }
}
