// TOML profile table loader.
//
// Maps on-disk `/etc/halo-power/profiles.toml` into a `Profile` map. Every
// field is Option<u32> so users can omit anything they don't want clamped —
// RyzenAdj just leaves those knobs alone.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Profile {
    /// Sustained Thermal / Average Power (mW). Long-run socket budget.
    pub stapm_limit: Option<u32>,
    /// PPT Fast (mW). Short-burst PPT.
    pub fast_limit: Option<u32>,
    /// PPT Slow (mW). Sits between STAPM and fast.
    pub slow_limit: Option<u32>,
    /// Tctl target (°C). Strix Halo safe ceiling is 95 °C.
    pub tctl_temp: Option<u32>,
    /// VRM EDC TDC (mA).
    pub vrm_current: Option<u32>,
    /// VRM EDC max (mA).
    pub vrmmax_current: Option<u32>,
    /// SoC VRM TDC (mA).
    pub vrmsoc_current: Option<u32>,
    /// SoC VRM EDC max (mA).
    pub vrmsocmax_current: Option<u32>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Profiles {
    #[serde(flatten)]
    map: BTreeMap<String, Profile>,
}

impl Profiles {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let s = fs::read_to_string(p)
            .with_context(|| format!("reading profiles from {}", p.display()))?;
        let me: Profiles =
            toml::from_str(&s).with_context(|| format!("parsing profiles from {}", p.display()))?;
        Ok(me)
    }

    pub fn get(&self, name: &str) -> Option<&Profile> {
        self.map.get(name)
    }

    pub fn names(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_inline_table() {
        let src = r#"
[balanced]
stapm_limit = 55000
fast_limit  = 80000
slow_limit  = 70000
tctl_temp   = 90
"#;
        let p: Profiles = toml::from_str(src).unwrap();
        let b = p.get("balanced").unwrap();
        assert_eq!(b.stapm_limit, Some(55000));
        assert_eq!(b.fast_limit, Some(80000));
        assert_eq!(b.tctl_temp, Some(90));
        assert_eq!(b.vrm_current, None);
    }

    #[test]
    fn missing_profile_is_none() {
        let src = "[quiet]\nstapm_limit = 30000\n";
        let p: Profiles = toml::from_str(src).unwrap();
        assert!(p.get("nope").is_none());
        assert_eq!(p.names(), vec!["quiet".to_string()]);
    }
}
