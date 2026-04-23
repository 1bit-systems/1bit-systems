// RyzenAdj backend abstraction.
//
// Two implementations:
//
//   * ShelloutBackend — invokes `/usr/bin/ryzenadj`. Default. Zero build-time
//     dependencies. One fork/exec per `apply_profile`; fine for a unit that
//     runs on boot and on explicit user request.
//
//   * LibBackend (feature `libryzenadj`) — dlopens `libryzenadj.so` and
//     calls the C API directly. Stubbed today. Wire the real vtable once
//     we measure shellout overhead and decide it's worth the extra moving
//     part.

use crate::profiles::Profile;
use anyhow::{Context, Result, bail};
use std::process::Command;
use thiserror::Error;
use tracing::{debug, info};

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("ryzenadj exit status {0}")]
    BadStatus(i32),
    #[error("unknown knob `{0}`")]
    UnknownKey(String),
}

pub trait PowerBackend {
    fn name(&self) -> &'static str;
    fn apply_profile(&self, p: &Profile) -> Result<()>;
    fn set_one(&self, key: &str, value: u32) -> Result<()>;
}

pub struct ShelloutBackend {
    dry_run: bool,
    path: &'static str,
}

impl ShelloutBackend {
    pub fn new(dry_run: bool) -> Self {
        Self {
            dry_run,
            path: "/usr/bin/ryzenadj",
        }
    }

    fn run(&self, args: &[String]) -> Result<()> {
        debug!(?args, "ryzenadj invocation");
        if self.dry_run {
            info!(cmd = %format!("{} {}", self.path, args.join(" ")), "dry-run");
            return Ok(());
        }
        let status = Command::new(self.path)
            .args(args)
            .status()
            .with_context(|| format!("spawning {}", self.path))?;
        if !status.success() {
            bail!(BackendError::BadStatus(status.code().unwrap_or(-1)));
        }
        Ok(())
    }
}

impl PowerBackend for ShelloutBackend {
    fn name(&self) -> &'static str {
        "shellout"
    }

    fn apply_profile(&self, p: &Profile) -> Result<()> {
        let mut args: Vec<String> = Vec::with_capacity(8);
        fn push(args: &mut Vec<String>, flag: &str, v: Option<u32>) {
            if let Some(v) = v {
                args.push(format!("--{flag}={v}"));
            }
        }
        push(&mut args, "stapm-limit", p.stapm_limit);
        push(&mut args, "fast-limit", p.fast_limit);
        push(&mut args, "slow-limit", p.slow_limit);
        push(&mut args, "tctl-temp", p.tctl_temp);
        push(&mut args, "vrm-current", p.vrm_current);
        push(&mut args, "vrmmax-current", p.vrmmax_current);
        push(&mut args, "vrmsoc-current", p.vrmsoc_current);
        push(&mut args, "vrmsocmax-current", p.vrmsocmax_current);
        if args.is_empty() {
            return Ok(());
        }
        self.run(&args)
    }

    fn set_one(&self, key: &str, value: u32) -> Result<()> {
        const KNOBS: &[&str] = &[
            "stapm-limit",
            "fast-limit",
            "slow-limit",
            "tctl-temp",
            "vrm-current",
            "vrmmax-current",
            "vrmsoc-current",
            "vrmsocmax-current",
        ];
        if !KNOBS.contains(&key) {
            bail!(BackendError::UnknownKey(key.to_string()));
        }
        self.run(&[format!("--{key}={value}")])
    }
}

#[cfg(feature = "libryzenadj")]
pub struct LibBackend {
    _lib: libloading::Library,
}

#[cfg(feature = "libryzenadj")]
impl LibBackend {
    pub fn open() -> Result<Self> {
        // Stub: in real wiring we'd resolve set_stapm_limit, set_fast_limit,
        // etc. and call them on a `ryzen_access` handle. Deliberately TODO
        // until we profile shellout overhead.
        let lib = unsafe { libloading::Library::new("libryzenadj.so") }
            .context("dlopen libryzenadj.so")?;
        Ok(Self { _lib: lib })
    }
}

#[cfg(feature = "libryzenadj")]
impl PowerBackend for LibBackend {
    fn name(&self) -> &'static str {
        "libryzenadj"
    }
    fn apply_profile(&self, _p: &Profile) -> Result<()> {
        bail!("libryzenadj backend not yet wired; use default shellout");
    }
    fn set_one(&self, _key: &str, _value: u32) -> Result<()> {
        bail!("libryzenadj backend not yet wired; use default shellout");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_run_apply_does_not_spawn() {
        let b = ShelloutBackend::new(true);
        let p = Profile {
            stapm_limit: Some(55_000),
            ..Default::default()
        };
        b.apply_profile(&p).expect("dry-run should be infallible");
    }

    #[test]
    fn rejects_unknown_knob() {
        let b = ShelloutBackend::new(true);
        let err = b.set_one("not-a-knob", 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("unknown knob"), "got: {s}");
    }
}
