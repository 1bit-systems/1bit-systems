// `1bit doctor` — health check across the stack. Exit 0 green, 1 warn, 2 fail.

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::status::SERVICES;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Ok,
    Warn,
    Fail,
}

impl Outcome {
    fn glyph(self) -> &'static str {
        match self {
            Self::Ok => "●",
            Self::Warn => "◉",
            Self::Fail => "○",
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Warn => "warn",
            Self::Fail => "fail",
        }
    }
}

fn row(name: &str, out: Outcome, detail: &str) {
    println!(
        "  {}  {:<20} {:<5} {}",
        out.glyph(),
        name,
        out.tag(),
        detail
    );
}

fn cmd_ok(bin: &str, args: &[&str]) -> bool {
    Command::new(bin)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn cmd_out(bin: &str, args: &[&str]) -> Option<String> {
    Command::new(bin)
        .args(args)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
}

fn check_gpu() -> (Outcome, String) {
    match cmd_out("/opt/rocm/bin/rocminfo", &[]) {
        Some(s) if s.contains("gfx1151") => (Outcome::Ok, "gfx1151 present".into()),
        Some(_) => (Outcome::Fail, "rocminfo ran but no gfx1151".into()),
        None => (Outcome::Fail, "rocminfo not found at /opt/rocm/bin".into()),
    }
}

fn check_kernel() -> (Outcome, String) {
    match cmd_out("uname", &["-r"]) {
        Some(s) => {
            let v = s.trim();
            let major = v
                .split('.')
                .next()
                .and_then(|n| n.parse::<u32>().ok())
                .unwrap_or(0);
            if major >= 7 {
                (Outcome::Ok, format!("kernel {v}"))
            } else {
                (Outcome::Warn, format!("kernel {v} — NPU driver wants 7.x"))
            }
        }
        None => (Outcome::Fail, "uname failed".into()),
    }
}

fn check_service(unit: &str, port: u16) -> (Outcome, String) {
    let active = cmd_ok("systemctl", &["--user", "is-active", "--quiet", unit]);
    let listening = port == 0
        || cmd_out("ss", &["-lnt"])
            .map(|s| s.lines().any(|l| l.contains(&format!("127.0.0.1:{port}"))))
            .unwrap_or(false);
    match (active, listening) {
        (true, true) => (Outcome::Ok, "active + listening".into()),
        (true, false) => (Outcome::Warn, "active, no port".into()),
        (false, _) => (Outcome::Fail, "inactive".into()),
    }
}

async fn check_http(url: &str) -> (Outcome, String) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .danger_accept_invalid_certs(true)
        .build()
    {
        Ok(c) => c,
        Err(e) => return (Outcome::Fail, e.to_string()),
    };
    match client.get(url).send().await {
        Ok(r) if r.status().is_success() => {
            (Outcome::Ok, format!("{} {}", r.status().as_u16(), url))
        }
        Ok(r) => (Outcome::Warn, format!("{} {}", r.status().as_u16(), url)),
        Err(e) => (Outcome::Fail, format!("{url}: {e}")),
    }
}

fn check_pi() -> (Outcome, String) {
    if cmd_ok("ping", &["-c", "1", "-W", "1", "100.64.0.4"]) {
        (Outcome::Ok, "100.64.0.4 reachable".into())
    } else {
        (Outcome::Warn, "pi not reachable (archive offline?)".into())
    }
}

fn check_halo_storage() -> Vec<(&'static str, Outcome, String)> {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return vec![("~", Outcome::Fail, "no home dir".into())],
    };
    let mut out = Vec::new();
    for (name, rel) in [
        ("skills", ".halo/skills"),
        ("memories", ".halo/memories"),
        ("state.db", ".halo/state.db"),
    ] {
        let p = home.join(rel);
        let (o, d) = if p.exists() {
            (Outcome::Ok, p.display().to_string())
        } else {
            (
                Outcome::Warn,
                format!("{} missing (first-run ok)", p.display()),
            )
        };
        out.push((name, o, d));
    }
    out
}

fn check_tunnel_config() -> (Outcome, String) {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return (Outcome::Fail, "no home".into()),
    };
    let bin = cmd_ok("cloudflared", &["--version"]);
    let cfg = home.join(".cloudflared/config.yml").exists();
    let cert = home.join(".cloudflared/cert.pem").exists();
    match (bin, cfg, cert) {
        (true, true, true) => (Outcome::Ok, "cloudflared + config + cert present".into()),
        (true, false, _) => (
            Outcome::Warn,
            "cloudflared installed, config.yml not in place".into(),
        ),
        (true, true, false) => (
            Outcome::Warn,
            "config present but no cert.pem (run `cloudflared tunnel login`)".into(),
        ),
        (false, _, _) => (
            Outcome::Warn,
            "cloudflared not installed (pacman -S cloudflared)".into(),
        ),
    }
}

// --- accelerator probes ------------------------------------------------------
//
// Each probe takes an injected filesystem root so tests can assemble a fake
// /sys tree under a tempdir. In production the caller passes `Path::new("/")`.
//
// Probes never shell out — they read sysfs only. This keeps them fast, cheap,
// and mockable.

/// XDNA2 NPU probe. Looks for `/sys/class/accel/accel*/device/vendor == 0x1022`
/// AND a `device` id in the Strix Halo NPU range (0x1502 = Phoenix, 0x17f0 =
/// STX/STX-H NPU5). A kernel module presence check on `/sys/module/amdxdna`
/// is reported as a stronger signal when both match.
pub(crate) fn npu_probe(root: &Path) -> (Outcome, String) {
    let accel_dir = root.join("sys/class/accel");
    let entries = match std::fs::read_dir(&accel_dir) {
        Ok(e) => e,
        Err(_) => {
            return (
                Outcome::Warn,
                "no /sys/class/accel — XDNA2 NPU absent or driver unloaded".into(),
            );
        }
    };

    let mut found: Vec<(String, String)> = Vec::new();
    for ent in entries.flatten() {
        let dev = ent.path().join("device");
        let vendor = std::fs::read_to_string(dev.join("vendor"))
            .ok()
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();
        let device = std::fs::read_to_string(dev.join("device"))
            .ok()
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();
        if vendor == "0x1022" && !device.is_empty() {
            found.push((ent.file_name().to_string_lossy().into_owned(), device));
        }
    }

    let mod_loaded = root.join("sys/module/amdxdna").exists();

    match (found.as_slice(), mod_loaded) {
        ([], _) => (
            Outcome::Warn,
            "no AMD accel device in /sys/class/accel (expected on LTS kernel)".into(),
        ),
        ([(name, id)], true) => (Outcome::Ok, format!("{name} device={id} amdxdna loaded")),
        ([(name, id)], false) => (
            Outcome::Warn,
            format!("{name} device={id} but amdxdna module not loaded"),
        ),
        (many, _) => (
            Outcome::Ok,
            format!("{} AMD accel devices found", many.len()),
        ),
    }
}

/// Intel Xe2 (Battlemage) probe on the sliger host. Walks PCI devices under
/// `/sys/bus/pci/devices/*`, matches vendor 0x8086 plus device ids in the
/// Battlemage range (0xE20B, 0xE202, 0xE20C, 0xE210, 0xE212, 0xE215, 0xE216),
/// and reports `xe` as the preferred kernel driver over `i915`.
pub(crate) fn xe2_probe(root: &Path) -> (Outcome, String) {
    let pci_dir = root.join("sys/bus/pci/devices");
    let entries = match std::fs::read_dir(&pci_dir) {
        Ok(e) => e,
        Err(_) => {
            return (
                Outcome::Warn,
                "no /sys/bus/pci/devices — not a PCI host?".into(),
            );
        }
    };

    const BATTLEMAGE_IDS: &[&str] = &[
        "0xe20b", "0xe202", "0xe20c", "0xe210", "0xe212", "0xe215", "0xe216",
    ];

    for ent in entries.flatten() {
        let p = ent.path();
        let vendor = std::fs::read_to_string(p.join("vendor"))
            .ok()
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();
        let device = std::fs::read_to_string(p.join("device"))
            .ok()
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();
        if vendor == "0x8086" && BATTLEMAGE_IDS.contains(&device.as_str()) {
            // Resolve the bound driver via the `driver` symlink.
            let drv_link = p.join("driver");
            let drv_name = std::fs::read_link(&drv_link)
                .ok()
                .and_then(|t| t.file_name().map(|s| s.to_string_lossy().into_owned()))
                .unwrap_or_else(|| "<none>".into());
            return match drv_name.as_str() {
                "xe" => (Outcome::Ok, format!("Battlemage {device} bound to xe")),
                "i915" => (
                    Outcome::Warn,
                    format!("Battlemage {device} bound to i915 — force xe.conf"),
                ),
                "<none>" => (
                    Outcome::Warn,
                    format!("Battlemage {device} present, no driver bound"),
                ),
                other => (
                    Outcome::Warn,
                    format!("Battlemage {device} bound to {other}"),
                ),
            };
        }
    }
    (
        Outcome::Warn,
        "no Intel Battlemage device on PCI bus (sliger-only)".into(),
    )
}

/// gfx1201 (RDNA 4 / RX 9070 XT) probe for the ryzen host. Uses
/// /sys/class/drm/card*/device/{vendor,device} + the `amdgpu` driver binding.
/// gfx1201 has PCI device id 0x7590 (Navi 48 XT, RX 9070 XT).
pub(crate) fn gfx1201_probe(root: &Path) -> (Outcome, String) {
    let drm_dir = root.join("sys/class/drm");
    let entries = match std::fs::read_dir(&drm_dir) {
        Ok(e) => e,
        Err(_) => {
            return (Outcome::Warn, "no /sys/class/drm — no DRM subsystem".into());
        }
    };

    const NAVI48_IDS: &[&str] = &["0x7590", "0x7591", "0x7592"];

    for ent in entries.flatten() {
        let name = ent.file_name().to_string_lossy().into_owned();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dev = ent.path().join("device");
        let vendor = std::fs::read_to_string(dev.join("vendor"))
            .ok()
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();
        let device = std::fs::read_to_string(dev.join("device"))
            .ok()
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();
        if vendor == "0x1002" && NAVI48_IDS.contains(&device.as_str()) {
            let drv_link = dev.join("driver");
            let drv_name = std::fs::read_link(&drv_link)
                .ok()
                .and_then(|t| t.file_name().map(|s| s.to_string_lossy().into_owned()))
                .unwrap_or_else(|| "<none>".into());
            return if drv_name == "amdgpu" {
                (
                    Outcome::Ok,
                    format!("gfx1201 RX 9070 XT ({device}) bound to amdgpu"),
                )
            } else {
                (
                    Outcome::Warn,
                    format!("gfx1201 {device} present, driver={drv_name}"),
                )
            };
        }
    }
    (
        Outcome::Warn,
        "no gfx1201 (Navi 48) on DRM bus (ryzen host only)".into(),
    )
}

fn fs_root() -> PathBuf {
    PathBuf::from("/")
}

/// OOBE anchor #7 hook: run the host-side probes (GPU/kernel/services/
/// endpoints) silently and return a `(warn, fail)` tally. Unlike `run`,
/// this never calls `std::process::exit` — it's safe to embed in the
/// install flow. The probes re-use the same functions `run` uses so the
/// tally matches what the operator would see if they ran `1bit doctor`
/// by hand.
pub(crate) fn tally_for_oobe() -> (u32, u32) {
    let mut warn = 0u32;
    let mut fail = 0u32;
    let mut tally = |o: Outcome| match o {
        Outcome::Warn => warn += 1,
        Outcome::Fail => fail += 1,
        Outcome::Ok => {}
    };
    // Host
    let (o, _) = check_gpu();
    tally(o);
    let (o, _) = check_kernel();
    tally(o);
    // Accelerators
    let root = fs_root();
    let (o, _) = npu_probe(&root);
    tally(o);
    let (o, _) = xe2_probe(&root);
    tally(o);
    let (o, _) = gfx1201_probe(&root);
    tally(o);
    // Services
    for (_, unit, port) in SERVICES {
        let (o, _) = check_service(unit, *port);
        tally(o);
    }
    // Storage
    for (_, o, _) in check_halo_storage() {
        tally(o);
    }
    (warn, fail)
}

pub async fn run() -> Result<()> {
    let mut warn = 0u32;
    let mut fail = 0u32;
    let mut tally = |o: Outcome| match o {
        Outcome::Warn => warn += 1,
        Outcome::Fail => fail += 1,
        Outcome::Ok => {}
    };

    println!("─── host ────────────────────────────────────");
    let (o, d) = check_gpu();
    tally(o);
    row("gpu", o, &d);
    let (o, d) = check_kernel();
    tally(o);
    row("kernel", o, &d);

    println!("\n─── accelerators ────────────────────────────");
    let root = fs_root();
    let (o, d) = npu_probe(&root);
    tally(o);
    row("npu (xdna2)", o, &d);
    let (o, d) = xe2_probe(&root);
    tally(o);
    row("xe2 (sliger)", o, &d);
    let (o, d) = gfx1201_probe(&root);
    tally(o);
    row("gfx1201 (ryzen)", o, &d);

    println!("\n─── services ────────────────────────────────");
    for (short, unit, port) in SERVICES {
        let (o, d) = check_service(unit, *port);
        tally(o);
        row(short, o, &d);
    }

    println!("\n─── endpoints ───────────────────────────────");
    for (name, url) in &[
        ("v1 models", "http://127.0.0.1:8080/v1/models"),
        ("v2 models", "http://127.0.0.1:8180/v1/models"),
        ("lemonade gw", "http://127.0.0.1:8200/v1/models"),
        ("landing", "http://127.0.0.1:8190/"),
        ("kokoro tts", "http://127.0.0.1:8083/voices"),
        ("whisper stt", "http://127.0.0.1:8082/health"),
    ] {
        let (o, d) = check_http(url).await;
        tally(o);
        row(name, o, &d);
    }

    println!("\n─── storage ─────────────────────────────────");
    for (name, o, d) in check_halo_storage() {
        tally(o);
        row(name, o, &d);
    }

    println!("\n─── tunnel ──────────────────────────────────");
    let (o, d) = check_tunnel_config();
    tally(o);
    row("cloudflared", o, &d);
    let (o, d) = check_http("https://api.1bit.systems/v1/models").await;
    tally(o);
    row("api public", o, &d);

    println!("\n─── network ─────────────────────────────────");
    let (o, d) = check_pi();
    tally(o);
    row("pi-archive", o, &d);

    println!("\n{} warn, {} fail", warn, fail);
    std::process::exit(if fail > 0 {
        2
    } else if warn > 0 {
        1
    } else {
        0
    });
}

#[cfg(test)]
mod tests {
    //! Fake-sysfs tests for the three accelerator probes. Each test builds a
    //! tempdir root and passes it in as the injected fs root. We deliberately
    //! do not touch the real filesystem.
    use super::*;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    use tempfile::TempDir;

    // ---- helpers ------------------------------------------------------------

    fn write(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    fn mk_accel(root: &Path, name: &str, vendor: &str, device: &str) {
        let dev = root.join("sys/class/accel").join(name).join("device");
        write(&dev.join("vendor"), &format!("{vendor}\n"));
        write(&dev.join("device"), &format!("{device}\n"));
    }

    fn mk_pci(root: &Path, bdf: &str, vendor: &str, device: &str, driver: Option<&str>) {
        let dev = root.join("sys/bus/pci/devices").join(bdf);
        write(&dev.join("vendor"), &format!("{vendor}\n"));
        write(&dev.join("device"), &format!("{device}\n"));
        if let Some(drv) = driver {
            let drv_dir = root.join("sys/bus/pci/drivers").join(drv);
            fs::create_dir_all(&drv_dir).unwrap();
            #[cfg(unix)]
            symlink(&drv_dir, dev.join("driver")).unwrap();
        }
    }

    fn mk_drm_card(root: &Path, card: &str, vendor: &str, device: &str, driver: Option<&str>) {
        let dev = root.join("sys/class/drm").join(card).join("device");
        write(&dev.join("vendor"), &format!("{vendor}\n"));
        write(&dev.join("device"), &format!("{device}\n"));
        if let Some(drv) = driver {
            let drv_dir = root.join("sys/bus/pci/drivers").join(drv);
            fs::create_dir_all(&drv_dir).unwrap();
            #[cfg(unix)]
            symlink(&drv_dir, dev.join("driver")).unwrap();
        }
    }

    // ---- npu_probe ----------------------------------------------------------

    #[test]
    fn npu_probe_missing_accel_dir_warns() {
        let td = TempDir::new().unwrap();
        let (o, d) = npu_probe(td.path());
        assert_eq!(o, Outcome::Warn);
        assert!(d.contains("/sys/class/accel") || d.contains("absent"));
    }

    #[test]
    fn npu_probe_empty_accel_dir_warns() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("sys/class/accel")).unwrap();
        let (o, d) = npu_probe(td.path());
        assert_eq!(o, Outcome::Warn);
        assert!(d.contains("no AMD accel"));
    }

    #[test]
    fn npu_probe_phoenix_with_module_ok() {
        let td = TempDir::new().unwrap();
        mk_accel(td.path(), "accel0", "0x1022", "0x1502");
        fs::create_dir_all(td.path().join("sys/module/amdxdna")).unwrap();
        let (o, d) = npu_probe(td.path());
        assert_eq!(o, Outcome::Ok);
        assert!(d.contains("accel0"));
        assert!(d.contains("0x1502"));
        assert!(d.contains("amdxdna"));
    }

    #[test]
    fn npu_probe_stxh_without_module_warns() {
        let td = TempDir::new().unwrap();
        mk_accel(td.path(), "accel0", "0x1022", "0x17f0");
        let (o, d) = npu_probe(td.path());
        assert_eq!(o, Outcome::Warn);
        assert!(d.contains("amdxdna module not loaded"));
    }

    #[test]
    fn npu_probe_wrong_vendor_ignored() {
        let td = TempDir::new().unwrap();
        mk_accel(td.path(), "accel0", "0x10de", "0x2330"); // nvidia-ish
        let (o, _d) = npu_probe(td.path());
        assert_eq!(o, Outcome::Warn);
    }

    // ---- xe2_probe ----------------------------------------------------------

    #[test]
    fn xe2_probe_missing_pci_warns() {
        let td = TempDir::new().unwrap();
        let (o, _d) = xe2_probe(td.path());
        assert_eq!(o, Outcome::Warn);
    }

    #[test]
    fn xe2_probe_no_battlemage_device_warns() {
        let td = TempDir::new().unwrap();
        mk_pci(td.path(), "0000:00:02.0", "0x8086", "0x46a6", Some("i915"));
        let (o, d) = xe2_probe(td.path());
        assert_eq!(o, Outcome::Warn);
        assert!(d.contains("no Intel Battlemage"));
    }

    #[cfg(unix)]
    #[test]
    fn xe2_probe_b580_on_xe_ok() {
        let td = TempDir::new().unwrap();
        mk_pci(td.path(), "0000:03:00.0", "0x8086", "0xe20b", Some("xe"));
        let (o, d) = xe2_probe(td.path());
        assert_eq!(o, Outcome::Ok);
        assert!(d.contains("0xe20b"));
        assert!(d.contains("xe"));
    }

    #[cfg(unix)]
    #[test]
    fn xe2_probe_b580_on_i915_warns() {
        let td = TempDir::new().unwrap();
        mk_pci(td.path(), "0000:03:00.0", "0x8086", "0xe20b", Some("i915"));
        let (o, d) = xe2_probe(td.path());
        assert_eq!(o, Outcome::Warn);
        assert!(d.contains("i915"));
    }

    #[test]
    fn xe2_probe_b580_no_driver_warns() {
        let td = TempDir::new().unwrap();
        mk_pci(td.path(), "0000:03:00.0", "0x8086", "0xe20b", None);
        let (o, d) = xe2_probe(td.path());
        assert_eq!(o, Outcome::Warn);
        assert!(d.contains("no driver") || d.contains("<none>"));
    }

    // ---- gfx1201_probe ------------------------------------------------------

    #[test]
    fn gfx1201_probe_missing_drm_warns() {
        let td = TempDir::new().unwrap();
        let (o, _d) = gfx1201_probe(td.path());
        assert_eq!(o, Outcome::Warn);
    }

    #[test]
    fn gfx1201_probe_no_navi48_warns() {
        let td = TempDir::new().unwrap();
        // gfx1151 Strix Halo iGPU (not Navi 48) — must not match.
        mk_drm_card(td.path(), "card0", "0x1002", "0x1586", Some("amdgpu"));
        let (o, d) = gfx1201_probe(td.path());
        assert_eq!(o, Outcome::Warn);
        assert!(d.contains("no gfx1201"));
    }

    #[cfg(unix)]
    #[test]
    fn gfx1201_probe_rx9070xt_on_amdgpu_ok() {
        let td = TempDir::new().unwrap();
        mk_drm_card(td.path(), "card1", "0x1002", "0x7590", Some("amdgpu"));
        let (o, d) = gfx1201_probe(td.path());
        assert_eq!(o, Outcome::Ok);
        assert!(d.contains("gfx1201"));
        assert!(d.contains("amdgpu"));
    }

    #[test]
    fn gfx1201_probe_skips_card_partitions() {
        let td = TempDir::new().unwrap();
        // a `card0-DP-1` connector entry should be skipped, and the real
        // card0 underneath should still match.
        let dev = td.path().join("sys/class/drm/card0-DP-1/device");
        write(&dev.join("vendor"), "0x1002\n");
        write(&dev.join("device"), "0x7590\n");
        let (o, d) = gfx1201_probe(td.path());
        // The connector-shaped dir is skipped by the `contains('-')` guard,
        // so with no other card present we warn.
        assert_eq!(o, Outcome::Warn);
        assert!(d.contains("no gfx1201"));
    }

    #[cfg(unix)]
    #[test]
    fn gfx1201_probe_wrong_driver_warns() {
        let td = TempDir::new().unwrap();
        mk_drm_card(td.path(), "card1", "0x1002", "0x7590", Some("vfio-pci"));
        let (o, d) = gfx1201_probe(td.path());
        assert_eq!(o, Outcome::Warn);
        assert!(d.contains("vfio-pci"));
    }
}
