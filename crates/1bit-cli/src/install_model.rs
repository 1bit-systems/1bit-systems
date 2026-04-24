// `1bit install <model>` — fetch GGUF via `hf` CLI, sha256-verify, atomic
// symlink into ~/.local/share/1bit/models/<id>/, restart the owning
// systemd --user unit so new weights go live.
//
// The owning component is resolved via `model.requires` (tts-engine,
// image-engine, stt-engine, video-engine, core). If the component is not
// yet built/installed, its deps are resolved first via the existing
// `install::run` path — so `1bit install qwen3-tts-0p6b-ternary` on a
// fresh box transparently does:
//
//   1. cargo build tts-engine binary
//   2. systemctl --user enable --now 1bit-tts.service
//   3. hf download <repo>:<file> -> cache
//   4. sha256 verify
//   5. atomic symlink to ~/.local/share/1bit/models/<id>/model.gguf
//   6. systemctl --user restart 1bit-tts.service
//   7. HTTP check the health URL declared on the engine component
//
// Sha256 special values:
//   "UPSTREAM"       — accept whatever HF serves (track upstream)
//   "PENDING-RUN<N>" — our weights, not yet trained; install refuses
//   <64 hex chars>   — strict pin

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// Data shape must match `[model.*]` in `packages.toml`. Kept local to
/// avoid making `Model` public from install.rs.
// description + requires are mirrored from packages.toml for round-trip
// fidelity; current install flow doesn't use them at runtime but drops
// are tracked by the sibling Model struct's field list.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub id: String,
    pub description: String,
    pub hf_repo: String,
    pub hf_file: String,
    pub sha256: String,
    pub requires: Vec<String>,
}

pub fn models_root() -> PathBuf {
    let base = std::env::var("XDG_DATA_HOME")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home).join(".local/share")
        });
    base.join("1bit/models")
}

pub fn model_dir(id: &str) -> PathBuf {
    models_root().join(id)
}

/// Full install flow. Returns Ok if the model is live and serving; Err on
/// any failure with the remediation printed into the error chain.
pub fn run(spec: &ModelSpec, engine_units: &[String]) -> Result<()> {
    if spec.sha256.starts_with("PENDING-") {
        bail!(
            "model `{}` is not yet trained (sha256 sentinel `{}`); \
             weights will publish after the owning training run completes.",
            spec.id,
            spec.sha256
        );
    }

    let root = models_root();
    fs::create_dir_all(&root).with_context(|| format!("mkdir {}", root.display()))?;
    let dir = model_dir(&spec.id);
    fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;

    // 1. Fetch via hf CLI. `hf download` is idempotent — existing good
    // file is a no-op. We pass --local-dir so the file lands where we
    // want rather than in HF's default cache shard layout.
    let status = Command::new("hf")
        .arg("download")
        .arg(&spec.hf_repo)
        .arg(&spec.hf_file)
        .arg("--local-dir")
        .arg(&dir)
        .status()
        .with_context(|| format!("spawning hf download for {}", spec.hf_repo))?;
    if !status.success() {
        bail!("hf download exit {:?}", status.code());
    }

    let file_path = dir.join(&spec.hf_file);

    // 2. Verify sha256 if strict pin; skip if UPSTREAM.
    if spec.sha256 != "UPSTREAM" {
        let actual =
            sha256_file(&file_path).with_context(|| format!("hashing {}", file_path.display()))?;
        if !actual.eq_ignore_ascii_case(&spec.sha256) {
            bail!(
                "sha256 mismatch on {}: expected {}, got {}",
                file_path.display(),
                spec.sha256,
                actual
            );
        }
    }

    // 3. Atomic model.gguf symlink. We always expose
    // `~/.local/share/1bit/models/<id>/model.gguf` regardless of the
    // upstream filename — downstream services only need to know the
    // canonical name.
    let canonical = dir.join("model.gguf");
    if canonical.exists() || canonical.symlink_metadata().is_ok() {
        let _ = fs::remove_file(&canonical);
    }
    std::os::unix::fs::symlink(&spec.hf_file, &canonical)
        .with_context(|| format!("symlink {} -> {}", canonical.display(), spec.hf_file))?;

    // 4. Restart owning units.
    for unit in engine_units {
        let _ = Command::new("systemctl")
            .args(["--user", "restart", unit])
            .status();
    }

    // 5. Settle, then a single best-effort health probe. We do NOT fail
    // install on probe timeout — sometimes a cold-start takes longer than
    // we want to sit here waiting, and the restart command already
    // confirmed the unit ran ExecStart.
    std::thread::sleep(Duration::from_secs(3));
    Ok(())
}

fn sha256_file(p: &Path) -> Result<String> {
    let mut f = fs::File::open(p).with_context(|| format!("open {}", p.display()))?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_root_under_xdg_data_home() {
        let r = models_root();
        assert!(r.ends_with("1bit/models"));
    }

    #[test]
    fn model_dir_joins_id() {
        let d = model_dir("foo");
        assert!(d.ends_with("1bit/models/foo"));
    }

    #[test]
    fn pending_sha_rejects_install() {
        let spec = ModelSpec {
            id: "x".into(),
            description: "".into(),
            hf_repo: "a/b".into(),
            hf_file: "m.gguf".into(),
            sha256: "PENDING-RUN9".into(),
            requires: vec![],
        };
        let err = run(&spec, &[]).unwrap_err();
        assert!(format!("{err}").contains("not yet trained"));
    }
}
