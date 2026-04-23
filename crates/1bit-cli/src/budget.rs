//! `halo budget` — unified-memory + service RSS audit for the
//! 8-concurrent-models target.
//!
//! Strix Halo has no real VRAM: the 512 MB `mem_info_vram_*` file is
//! the tiny stolen BAR; the actual model residency sits in GTT, which
//! is backed by the 128 GB LPDDR5x system pool. So the question
//! "how much room for model N+1?" reduces to: GTT free + system free,
//! minus the headroom the Wayland compositor needs to stay alive.
//!
//! This subcommand prints:
//! * GTT total / used / free (gfx1151)
//! * /proc/meminfo MemTotal / MemAvailable
//! * per-halo-service RSS (ps -C matching `halo-*` / `1bit-*`)
//! * a budget row computing free-for-next-model with a 4 GB compositor
//!   reserve.

use std::path::Path;

use anyhow::{Context, Result};

/// Keep 4 GB aside for KWin / Plasma / Chrome display buffers. Empirical,
/// not load-bearing — adjust once we have a real workload profile.
const COMPOSITOR_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Primary AMD GPU sysfs root on strix. Card index can drift; stick to
/// card1 for now and refactor when we get a second GPU.
const DRM_ROOT: &str = "/sys/class/drm/card1/device";

pub async fn run() -> Result<()> {
    let snap = Snapshot::read()?;
    println!("{}", snap.render());
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub gtt_total: u64,
    pub gtt_used: u64,
    pub vram_total: u64,
    pub vram_used: u64,
    pub mem_total: u64,
    pub mem_available: u64,
    pub services: Vec<ServiceRss>,
}

#[derive(Debug, Clone)]
pub struct ServiceRss {
    pub name: String,
    pub rss_kib: u64,
}

impl Snapshot {
    pub fn read() -> Result<Self> {
        let gtt_total = read_u64(Path::new(DRM_ROOT).join("mem_info_gtt_total"))?;
        let gtt_used = read_u64(Path::new(DRM_ROOT).join("mem_info_gtt_used"))?;
        let vram_total = read_u64(Path::new(DRM_ROOT).join("mem_info_vram_total"))?;
        let vram_used = read_u64(Path::new(DRM_ROOT).join("mem_info_vram_used"))?;

        let meminfo = std::fs::read_to_string("/proc/meminfo").context("read /proc/meminfo")?;
        let mem_total = parse_meminfo_kib(&meminfo, "MemTotal").unwrap_or(0) * 1024;
        let mem_available = parse_meminfo_kib(&meminfo, "MemAvailable").unwrap_or(0) * 1024;

        let services = scan_halo_services();

        Ok(Self {
            gtt_total,
            gtt_used,
            vram_total,
            vram_used,
            mem_total,
            mem_available,
            services,
        })
    }

    pub fn gtt_free(&self) -> u64 {
        self.gtt_total.saturating_sub(self.gtt_used)
    }

    pub fn budget_for_next_model(&self) -> u64 {
        let gtt_free = self.gtt_free();
        let ram_free = self.mem_available.saturating_sub(COMPOSITOR_RESERVE_BYTES);
        gtt_free.min(ram_free)
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "GTT   {used:>7} / {total:>7}  ({free} free)\n",
            used = fmt_bytes(self.gtt_used),
            total = fmt_bytes(self.gtt_total),
            free = fmt_bytes(self.gtt_free()),
        ));
        s.push_str(&format!(
            "VRAM  {used:>7} / {total:>7}  (stolen BAR — not the model pool)\n",
            used = fmt_bytes(self.vram_used),
            total = fmt_bytes(self.vram_total),
        ));
        s.push_str(&format!(
            "RAM   {avail:>7} / {total:>7}  available\n",
            avail = fmt_bytes(self.mem_available),
            total = fmt_bytes(self.mem_total),
        ));
        s.push_str(&format!(
            "next model budget  ≈ {} (min(GTT free, RAM avail − 4 GB reserve))\n",
            fmt_bytes(self.budget_for_next_model())
        ));
        s.push_str("---- halo services ----\n");
        for svc in &self.services {
            s.push_str(&format!(
                "  {:28} {:>7}\n",
                svc.name,
                fmt_bytes(svc.rss_kib * 1024)
            ));
        }
        s
    }
}

fn read_u64(p: impl AsRef<Path>) -> Result<u64> {
    let s = std::fs::read_to_string(p.as_ref())
        .with_context(|| format!("read {}", p.as_ref().display()))?;
    s.trim()
        .parse::<u64>()
        .with_context(|| format!("parse u64 from {}", p.as_ref().display()))
}

/// Parse a `Key:     N kB` row out of /proc/meminfo. Returns kibibytes.
pub fn parse_meminfo_kib(meminfo: &str, key: &str) -> Option<u64> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim_start_matches(':').trim();
            let first = rest.split_whitespace().next()?;
            return first.parse::<u64>().ok();
        }
    }
    None
}

/// Format bytes with a two-digit scale suffix. We do this by hand to
/// avoid pulling in a humansize crate for a single CLI string.
pub fn fmt_bytes(n: u64) -> String {
    const K: u64 = 1024;
    const M: u64 = K * 1024;
    const G: u64 = M * 1024;
    if n >= G {
        format!("{:.1} GB", n as f64 / G as f64)
    } else if n >= M {
        format!("{:.1} MB", n as f64 / M as f64)
    } else if n >= K {
        format!("{:.1} KB", n as f64 / K as f64)
    } else {
        format!("{n} B")
    }
}

/// Scan `/proc` for processes whose `comm` looks like a halo service.
/// Returns one row per pid — a single service may have multiple children
/// (caller can aggregate if needed).
fn scan_halo_services() -> Vec<ServiceRss> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return out;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = match name.to_str() {
            Some(n) => n,
            None => continue,
        };
        if !name.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let pid_dir = entry.path();
        let comm = std::fs::read_to_string(pid_dir.join("comm"))
            .unwrap_or_default()
            .trim()
            .to_string();
        if !looks_like_halo_service(&comm) {
            continue;
        }
        let status = std::fs::read_to_string(pid_dir.join("status")).unwrap_or_default();
        let rss_kib = parse_meminfo_kib(&status, "VmRSS").unwrap_or(0);
        out.push(ServiceRss {
            name: comm,
            rss_kib,
        });
    }
    out.sort_by_key(|s| std::cmp::Reverse(s.rss_kib));
    out
}

/// Coarse gate: process names we consider halo-owned. Picks up the
/// native binaries (`bitnet_decode`, `whisper-server`, `halo-server-real`)
/// as well as our own cargo binaries (`1bit-*`, `halo-*`).
pub fn looks_like_halo_service(comm: &str) -> bool {
    const HEURISTICS: &[&str] = &[
        "bitnet_decode",
        "whisper-server",
        "sd-server",
        "kokoro",
        "lemond",
        "halo-",
        "1bit-",
        "halo-server",
        "halo-landing",
    ];
    HEURISTICS.iter().any(|h| comm.contains(h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_meminfo_returns_kib_for_known_key() {
        let real = "MemTotal:       131000000 kB\nMemAvailable:   120000000 kB\n";
        assert_eq!(parse_meminfo_kib(real, "MemTotal"), Some(131_000_000));
        assert_eq!(parse_meminfo_kib(real, "MemAvailable"), Some(120_000_000));
        assert_eq!(parse_meminfo_kib(real, "Nope"), None);
    }

    #[test]
    fn fmt_bytes_picks_reasonable_unit() {
        assert_eq!(fmt_bytes(0), "0 B");
        assert_eq!(fmt_bytes(1023), "1023 B");
        assert!(fmt_bytes(1024 * 1024).starts_with("1.0 MB"));
        assert!(fmt_bytes(3 * 1024 * 1024 * 1024).starts_with("3.0 GB"));
    }

    #[test]
    fn halo_service_heuristic_covers_known_names() {
        assert!(looks_like_halo_service("bitnet_decode"));
        assert!(looks_like_halo_service("whisper-server"));
        assert!(looks_like_halo_service("halo-landing"));
        assert!(looks_like_halo_service("1bit-mcp"));
        assert!(!looks_like_halo_service("chrome"));
        assert!(!looks_like_halo_service("kwin_wayland"));
    }

    #[test]
    fn snapshot_budget_floors_at_zero_when_ram_exhausted() {
        let snap = Snapshot {
            gtt_total: 64 * 1024 * 1024 * 1024,
            gtt_used: 60 * 1024 * 1024 * 1024,
            vram_total: 0,
            vram_used: 0,
            mem_total: 128 * 1024 * 1024 * 1024,
            mem_available: 1024 * 1024 * 1024, // 1 GB, less than the 4 GB reserve
            services: Vec::new(),
        };
        assert_eq!(snap.budget_for_next_model(), 0);
    }
}
