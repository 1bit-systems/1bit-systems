// packages.toml watch-section parser.
//
// Reads ONLY the [watch.*] tables from the workspace manifest; ignores
// [component.*] and [model.*] so this crate stays narrow.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WatchKind {
    Github,
    Huggingface,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WatchEntryRaw {
    pub kind: WatchKind,
    pub repo: String,
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default = "default_soak")]
    pub soak_hours: u32,
    #[serde(default)]
    pub merge_into: Option<String>,
    #[serde(default)]
    pub on_merge: Vec<Vec<String>>,
    #[serde(default)]
    pub on_bump: Vec<Vec<String>>,
    #[serde(default)]
    pub notify: String,
}

fn default_soak() -> u32 {
    24
}

#[derive(Debug, Clone)]
pub struct WatchEntry {
    pub id: String,
    pub kind: WatchKind,
    pub repo: String,
    pub branch: Option<String>,
    pub soak_hours: u32,
    pub on_merge: Vec<Vec<String>>,
    pub on_bump: Vec<Vec<String>>,
    pub notify: String,
}

#[derive(Debug, Deserialize)]
struct ManifestRaw {
    #[serde(default)]
    watch: BTreeMap<String, WatchEntryRaw>,
}

pub struct Manifest {
    pub watch: BTreeMap<String, WatchEntry>,
}

impl Manifest {
    pub fn load(path: &str) -> Result<Self> {
        let raw = fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        Self::from_toml(&raw)
    }

    pub fn from_toml(raw: &str) -> Result<Self> {
        // We deserialize the whole doc as `toml::Value` and pluck only the
        // `watch` table; this avoids failing on the `component` / `model`
        // sections that don't match our schema.
        let val: toml::Value = raw.parse().context("parsing toml")?;
        let watch_val = val
            .get("watch")
            .cloned()
            .unwrap_or(toml::Value::Table(Default::default()));
        let raw_entries: BTreeMap<String, WatchEntryRaw> = watch_val.try_into()?;
        let watch = raw_entries
            .into_iter()
            .map(|(id, r)| {
                (
                    id.clone(),
                    WatchEntry {
                        id,
                        kind: r.kind,
                        repo: r.repo,
                        branch: r.branch,
                        soak_hours: r.soak_hours,
                        on_merge: r.on_merge,
                        on_bump: r.on_bump,
                        notify: r.notify,
                    },
                )
            })
            .collect();
        Ok(Self { watch })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_github_entry() {
        let toml = r#"
            [watch.qwen3-tts]
            kind       = "github"
            repo       = "khimaros/qwen3-tts.cpp"
            branch     = "main"
            soak_hours = 24
            on_merge   = [["cmake", "--build", "."]]
            notify     = "discord:halo-updates"
        "#;
        let m = Manifest::from_toml(toml).unwrap();
        assert_eq!(m.watch.len(), 1);
        let e = m.watch.get("qwen3-tts").unwrap();
        assert_eq!(e.repo, "khimaros/qwen3-tts.cpp");
        assert_eq!(e.soak_hours, 24);
        assert_eq!(e.on_merge.len(), 1);
    }

    #[test]
    fn parses_huggingface_entry() {
        let toml = r#"
            [watch.wan]
            kind    = "huggingface"
            repo    = "Wan-AI/Wan2.2-TI2V-5B"
            on_bump = [["ssh", "runpod", "requant"]]
            notify  = "discord:halo-updates"
        "#;
        let m = Manifest::from_toml(toml).unwrap();
        let e = m.watch.get("wan").unwrap();
        assert_eq!(e.soak_hours, 24); // default applied
        assert!(matches!(e.kind, WatchKind::Huggingface));
    }

    #[test]
    fn ignores_other_sections() {
        let toml = r#"
            [component.core]
            description = "core"
            [model.foo]
            description = "m"
            [watch.x]
            kind = "github"
            repo = "a/b"
            notify = ""
        "#;
        let m = Manifest::from_toml(toml).unwrap();
        assert_eq!(m.watch.len(), 1);
    }

    #[test]
    fn empty_watch_section_is_ok() {
        let m = Manifest::from_toml("[component.foo]\ndescription = \"x\"\n").unwrap();
        assert!(m.watch.is_empty());
    }
}
