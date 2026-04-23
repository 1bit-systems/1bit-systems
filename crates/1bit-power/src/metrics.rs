// One-line JSON metrics sampler for `halo-power log`.
//
// Reads `/sys/class/hwmon` entries (temp_input, power1_average, freq) and
// emits a single serde_json line. Deliberately tolerant: anything missing
// just becomes `null`. A systemd timer fires this every 30 s and pipes to
// journald where Loki/`halo logs` can scoop it up.

use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize)]
pub struct Sample {
    pub ts_unix: u64,
    pub host: String,
    pub tctl_c:      Option<f32>,
    pub edge_c:      Option<f32>,
    pub pkg_power_w: Option<f32>,
    pub ec_temp_c:   Option<i32>,
    pub ec_power_mode: Option<String>,
    pub ec_fan1_rpm: Option<u32>,
    pub ec_fan2_rpm: Option<u32>,
    pub ec_fan3_rpm: Option<u32>,
}

pub fn sample() -> Result<String> {
    let ec = crate::ec::EcBackend::new();
    let (temp, mode, rpm1, rpm2, rpm3) = if ec.available() {
        (
            ec.temp_c().ok(),
            ec.power_mode().ok(),
            ec.fan(1).ok().map(|f| f.rpm),
            ec.fan(2).ok().map(|f| f.rpm),
            ec.fan(3).ok().map(|f| f.rpm),
        )
    } else {
        (None, None, None, None, None)
    };
    let s = Sample {
        ts_unix: SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        host: hostname(),
        tctl_c:      read_hwmon("k10temp", "temp1_input").map(|v| v / 1000.0),
        edge_c:      read_hwmon("amdgpu",  "temp1_input").map(|v| v / 1000.0),
        pkg_power_w: read_hwmon("amdgpu",  "power1_average").map(|v| v / 1_000_000.0),
        ec_temp_c: temp,
        ec_power_mode: mode,
        ec_fan1_rpm: rpm1,
        ec_fan2_rpm: rpm2,
        ec_fan3_rpm: rpm3,
    };
    Ok(serde_json::to_string(&s)?)
}

fn hostname() -> String {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

/// Walk /sys/class/hwmon/hwmon*/name and, if it matches `want`, read `file`
/// from that directory as a decimal integer.
fn read_hwmon(want: &str, file: &str) -> Option<f32> {
    let root = Path::new("/sys/class/hwmon");
    for entry in fs::read_dir(root).ok()?.flatten() {
        let name = fs::read_to_string(entry.path().join("name")).ok()?;
        if name.trim() == want {
            let raw = fs::read_to_string(entry.path().join(file)).ok()?;
            return raw.trim().parse::<f32>().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_is_valid_json() {
        // Missing hwmon files on a non-Strix box should still yield a
        // parseable line — everything just becomes null.
        let line = sample().unwrap();
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert!(v.get("ts_unix").is_some());
        assert!(v.get("host").is_some());
    }
}
