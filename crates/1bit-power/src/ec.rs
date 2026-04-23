// Embedded-controller backend for Sixunited AXB35-02 board.
//
// Drives fans + APU power-mode + CPU temp via the `ec_su_axb35` kernel
// module's sysfs surface at /sys/class/ec_su_axb35/. Complementary to
// ryzen.rs (MSR layer): EC sets fan policy and board-level power mode;
// ryzenadj sets APU stapm/fast/slow limits.
//
// Writes require root (sysfs files are 0644 + root-owned); read path
// works as the invoking user. The CLI documents that `fan` and `board`
// subcommands must be run with sudo.
//
// Layered so the sysfs root is injectable — tests point it at a
// tempdir that mimics the real /sys layout.

use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

pub const SYSFS_ROOT: &str = "/sys/class/ec_su_axb35";

#[derive(Debug, Clone)]
pub struct EcBackend {
    root: PathBuf,
}

impl EcBackend {
    pub fn new() -> Self {
        Self {
            root: PathBuf::from(SYSFS_ROOT),
        }
    }

    #[allow(dead_code)]
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn available(&self) -> bool {
        self.root.join("apu").join("power_mode").is_file()
    }

    fn read_trim(&self, rel: &[&str]) -> Result<String> {
        let mut p = self.root.clone();
        for part in rel {
            p.push(part);
        }
        let raw = fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        Ok(raw.trim().to_string())
    }

    fn write_str(&self, rel: &[&str], val: &str) -> Result<()> {
        let mut p = self.root.clone();
        for part in rel {
            p.push(part);
        }
        fs::write(&p, val).with_context(|| format!("write {} to {}", val, p.display()))
    }

    pub fn temp_c(&self) -> Result<i32> {
        let s = self.read_trim(&["temp1", "temp"])?;
        s.parse::<i32>()
            .with_context(|| format!("parse temp `{s}`"))
    }

    pub fn power_mode(&self) -> Result<String> {
        self.read_trim(&["apu", "power_mode"])
    }

    pub fn set_power_mode(&self, mode: &str) -> Result<()> {
        const VALID: &[&str] = &["quiet", "balanced", "performance"];
        if !VALID.contains(&mode) {
            bail!("power_mode must be one of {VALID:?}, got `{mode}`");
        }
        self.write_str(&["apu", "power_mode"], mode)
    }

    pub fn fan(&self, id: u8) -> Result<FanSnapshot> {
        validate_fan_id(id)?;
        let name = format!("fan{id}");
        let rpm: u32 = self.read_trim(&[&name, "rpm"])?.parse().unwrap_or(0);
        let mode = self.read_trim(&[&name, "mode"])?;
        let level: u8 = self.read_trim(&[&name, "level"])?.parse().unwrap_or(0);
        let rampup = parse_curve(&self.read_trim(&[&name, "rampup_curve"])?);
        let rampdown = parse_curve(&self.read_trim(&[&name, "rampdown_curve"])?);
        Ok(FanSnapshot {
            id,
            rpm,
            mode,
            level,
            rampup,
            rampdown,
        })
    }

    pub fn set_fan_mode(&self, id: u8, mode: &str) -> Result<()> {
        validate_fan_id(id)?;
        const VALID: &[&str] = &["auto", "fixed", "curve"];
        if !VALID.contains(&mode) {
            bail!("fan mode must be one of {VALID:?}, got `{mode}`");
        }
        self.write_str(&[&format!("fan{id}"), "mode"], mode)
    }

    pub fn set_fan_level(&self, id: u8, level: u8) -> Result<()> {
        validate_fan_id(id)?;
        if level > 5 {
            bail!("fan level must be 0..=5, got {level}");
        }
        self.write_str(&[&format!("fan{id}"), "level"], &level.to_string())
    }

    pub fn set_fan_curve(&self, id: u8, direction: CurveDir, curve: &[u8; 5]) -> Result<()> {
        validate_fan_id(id)?;
        let file = match direction {
            CurveDir::Rampup => "rampup_curve",
            CurveDir::Rampdown => "rampdown_curve",
        };
        let csv = curve
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        self.write_str(&[&format!("fan{id}"), file], &csv)
    }

    pub fn snapshot(&self) -> Result<EcSnapshot> {
        Ok(EcSnapshot {
            temp_c: self.temp_c().ok(),
            power_mode: self.power_mode().ok(),
            fans: (1..=3).filter_map(|i| self.fan(i).ok()).collect(),
        })
    }
}

impl Default for EcBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CurveDir {
    Rampup,
    Rampdown,
}

#[derive(Debug, Clone, Serialize)]
pub struct FanSnapshot {
    pub id: u8,
    pub rpm: u32,
    pub mode: String,
    pub level: u8,
    pub rampup: Vec<u8>,
    pub rampdown: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EcSnapshot {
    pub temp_c: Option<i32>,
    pub power_mode: Option<String>,
    pub fans: Vec<FanSnapshot>,
}

fn validate_fan_id(id: u8) -> Result<()> {
    if !(1..=3).contains(&id) {
        bail!("fan id must be 1..=3, got {id}");
    }
    Ok(())
}

fn parse_curve(s: &str) -> Vec<u8> {
    s.split(',')
        .filter_map(|t| t.trim().parse::<u8>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn fake_sysfs() -> (tempfile::TempDir, EcBackend) {
        let dir = tempdir().unwrap();
        let root = dir.path().to_path_buf();
        for name in ["fan1", "fan2", "fan3"] {
            let fan = root.join(name);
            fs::create_dir_all(&fan).unwrap();
            fs::write(fan.join("rpm"), "0\n").unwrap();
            fs::write(fan.join("mode"), "curve\n").unwrap();
            fs::write(fan.join("level"), "0\n").unwrap();
            fs::write(fan.join("rampup_curve"), "60,70,83,95,97\n").unwrap();
            fs::write(fan.join("rampdown_curve"), "40,50,80,94,96\n").unwrap();
        }
        let temp = root.join("temp1");
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("temp"), "42\n").unwrap();
        let apu = root.join("apu");
        fs::create_dir_all(&apu).unwrap();
        fs::write(apu.join("power_mode"), "balanced\n").unwrap();
        let be = EcBackend::with_root(&root);
        (dir, be)
    }

    #[test]
    fn available_true_when_apu_present() {
        let (_d, be) = fake_sysfs();
        assert!(be.available());
    }

    #[test]
    fn reads_temp_and_mode() {
        let (_d, be) = fake_sysfs();
        assert_eq!(be.temp_c().unwrap(), 42);
        assert_eq!(be.power_mode().unwrap(), "balanced");
    }

    #[test]
    fn fan_snapshot_parses_curve() {
        let (_d, be) = fake_sysfs();
        let f = be.fan(1).unwrap();
        assert_eq!(f.id, 1);
        assert_eq!(f.mode, "curve");
        assert_eq!(f.rampup, vec![60, 70, 83, 95, 97]);
    }

    #[test]
    fn set_power_mode_round_trip() {
        let (_d, be) = fake_sysfs();
        be.set_power_mode("performance").unwrap();
        assert_eq!(be.power_mode().unwrap(), "performance");
    }

    #[test]
    fn rejects_bad_power_mode() {
        let (_d, be) = fake_sysfs();
        assert!(be.set_power_mode("turbo").is_err());
    }

    #[test]
    fn rejects_bad_fan_id() {
        let (_d, be) = fake_sysfs();
        assert!(be.fan(0).is_err());
        assert!(be.fan(4).is_err());
    }

    #[test]
    fn set_fan_level_clamps() {
        let (_d, be) = fake_sysfs();
        be.set_fan_level(1, 3).unwrap();
        assert!(be.set_fan_level(1, 6).is_err());
    }

    #[test]
    fn set_fan_curve_writes_csv() {
        let (_d, be) = fake_sysfs();
        be.set_fan_curve(2, CurveDir::Rampup, &[30, 40, 60, 80, 95])
            .unwrap();
        let f = be.fan(2).unwrap();
        assert_eq!(f.rampup, vec![30, 40, 60, 80, 95]);
    }

    #[test]
    fn snapshot_collects_all_fans() {
        let (_d, be) = fake_sysfs();
        let snap = be.snapshot().unwrap();
        assert_eq!(snap.fans.len(), 3);
        assert_eq!(snap.temp_c, Some(42));
        assert_eq!(snap.power_mode.as_deref(), Some("balanced"));
    }
}
