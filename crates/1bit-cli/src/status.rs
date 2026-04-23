// `1bit status` — one-line-per-service state + port listener check.

use anyhow::Result;
use std::process::Command;

pub const SERVICES: &[(&str, &str, u16)] = &[
    ("bitnet", "1bit-halo-bitnet.service", 8080), // gen-1 C++ bitnet_decode
    ("strix", "strix-server.service", 8180),      // gen-2 Rust 1bit-server
    ("sd", "1bit-halo-sd.service", 8081),
    ("whisper", "1bit-halo-whisper.service", 8082),
    ("kokoro", "1bit-halo-kokoro.service", 8083),
    ("lemonade", "1bit-halo-lemonade.service", 8000), // gen-1 lemonade daemon
    ("strix-lm", "strix-lemonade.service", 8200),     // gen-2 1bit-lemonade OpenAI gateway
    ("landing", "strix-landing.service", 8190),       // 1bit-landing marketing + /metrics
    ("burnin", "strix-burnin.service", 0),            // shadow-burnin v1 vs v2
    ("tunnel", "strix-cloudflared.service", 0),       // CF tunnel → api.1bit.systems
    ("agent", "1bit-halo-agent.service", 0),
];

pub const TIMERS: &[(&str, &str)] = &[
    ("anvil", "1bit-halo-anvil.timer"),
    ("gh-trio", "1bit-halo-gh-trio.timer"),
    ("memory-sync", "1bit-halo-memory-sync.timer"),
    ("archive", "1bit-halo-archive.timer"),
];

fn systemctl_user_active(unit: &str) -> bool {
    Command::new("systemctl")
        .args(["--user", "is-active", "--quiet", unit])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn port_listening(port: u16) -> bool {
    if port == 0 {
        return true;
    } // agent has no port
    Command::new("ss")
        .args(["-lnt"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.lines().any(|l| l.contains(&format!("127.0.0.1:{port}"))))
        .unwrap_or(false)
}

pub async fn run() -> Result<()> {
    println!("─── services (user systemd) ─────────────────");
    for (short, unit, port) in SERVICES {
        let active = systemctl_user_active(unit);
        let listening = port_listening(*port);
        let dot = if active && listening {
            "●"
        } else if active {
            "◉"
        } else {
            "○"
        };
        let port_s = if *port == 0 {
            "     ".into()
        } else {
            format!(":{port}")
        };
        println!(
            "  {dot}  {:<10} {:<25} {:<5} {}",
            short,
            unit,
            port_s,
            if active {
                if listening {
                    "active"
                } else {
                    "active (no port)"
                }
            } else {
                "inactive"
            }
        );
    }
    println!();
    println!("─── timers ──────────────────────────────────");
    for (short, unit) in TIMERS {
        let active = systemctl_user_active(unit);
        let dot = if active { "●" } else { "○" };
        println!("  {dot}  {:<12} {}", short, unit);
    }
    Ok(())
}
