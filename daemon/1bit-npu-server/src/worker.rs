//! NPU Worker subprocess — communicates with `npu_engine_universal --worker`
//! via a binary protocol over stdin/stdout.
//!
//! Protocol:
//!   Write: [op:u32][layer:u32][batch:u32][in_dim:u32][data:f32 x batch*in_dim]
//!   Read:  [status:u32][out_dim:u32][data:f32 x out_dim]

use anyhow::{Context, Result};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use tracing::info;

/// GEMM operation codes matching the C++ worker.
#[derive(Debug, Clone, Copy)]
pub enum GemmOp {
    Qkv = 1,
    O = 2,
    GateUp = 3,
    Down = 5,
}

/// Manages the NPU worker subprocess lifecycle and GEMM operations.
pub struct NpuWorker {
    child: Option<Child>,
    engine_bin: String,
    model_path: String,
    model_tag: String,
    xclbin_dir: String,
}

impl NpuWorker {
    pub fn new(engine_bin: String, model_path: String, model_tag: String, xclbin_dir: String) -> Self {
        Self {
            child: None,
            engine_bin,
            model_path,
            model_tag,
            xclbin_dir,
        }
    }

    /// Start the NPU worker subprocess and wait for "READY" on stdout.
    pub fn start(&mut self) -> Result<()> {
        let engine_path = Path::new(&self.engine_bin);
        if !engine_path.exists() || !engine_path.is_file() {
            anyhow::bail!("NPU engine not found or not executable: {}", self.engine_bin);
        }
        let model_path = Path::new(&self.model_path);
        if !model_path.exists() {
            anyhow::bail!("Model not found: {}", self.model_path);
        }

        // Set environment
        let mut proc_env: Vec<(String, String)> = std::env::vars().collect();
        proc_env.push(("NPU_XCLBIN_DIR".to_string(), self.xclbin_dir.clone()));

        info!(
            "Starting NPU worker: {} {} --model-tag {} --worker",
            self.engine_bin, self.model_path, self.model_tag
        );

        let mut child = Command::new(&self.engine_bin)
            .arg(&self.model_path)
            .arg("--model-tag")
            .arg(&self.model_tag)
            .arg("--worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .envs(proc_env)
            .spawn()
            .with_context(|| format!("Failed to spawn NPU worker: {}", self.engine_bin))?;

        // Wait for "READY" on stdout (with 60s timeout)
        let stdout = child.stdout.as_mut()
            .context("Failed to capture worker stdout")?;

        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(60);
        let mut buf = Vec::new();
        let mut ready = false;

        while start.elapsed() < timeout {
            let mut byte = [0u8; 1];
            match stdout.read(&mut byte) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    buf.push(byte[0]);
                    // Check for "READY\n" at the end
                    if buf.len() >= 6 && &buf[buf.len() - 6..] == b"READY\n" {
                        ready = true;
                        break;
                    }
                    // Also check for "READY\n" anywhere in the buffer
                    if buf.windows(6).any(|w| w == b"READY\n") {
                        ready = true;
                        break;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // WouldBlock doesn't happen with piped stdout normally
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                Err(e) => {
                    anyhow::bail!("Error reading worker stdout: {e}");
                }
            }
        }

        if !ready {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("NPU worker did not signal READY within 60s");
        }

        info!("NPU worker ready (PID {})", child.id());
        self.child = Some(child);
        Ok(())
    }

    /// Run a GEMM operation on the NPU.
    ///
    /// `data` is a flat f32 array of length `batch * in_dim`.
    /// Returns flat f32 array of length out_dim.
    pub fn gemm(&mut self, op: GemmOp, layer: u32, batch: u32, in_dim: u32, data: &[f32]) -> Result<Vec<f32>> {
        let child = self.child.as_mut()
            .context("NPU worker not started")?;

        let stdin = child.stdin.as_mut()
            .context("Worker stdin not available")?;
        let stdout = child.stdout.as_mut()
            .context("Worker stdout not available")?;

        // Write header: 4 x u32 LE (op, layer, batch, in_dim)
        let header = [
            (op as u32).to_le_bytes(),
            layer.to_le_bytes(),
            batch.to_le_bytes(),
            in_dim.to_le_bytes(),
        ];
        for h in &header {
            stdin.write_all(h)?;
        }

        // Write data as raw f32 bytes
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        stdin.write_all(data_bytes)?;
        stdin.flush()?;

        // Read response: 2 x u32 LE (status, out_dim)
        let mut resp_header = [0u8; 8];
        stdout.read_exact(&mut resp_header)
            .context("Failed to read worker response header")?;

        let status = u32::from_le_bytes(resp_header[0..4].try_into().unwrap());
        let out_dim = u32::from_le_bytes(resp_header[4..8].try_into().unwrap());

        if status != 0 {
            anyhow::bail!("GEMM failed: op={:?} layer={} status={}", op, layer, status);
        }

        // Read output data
        let out_bytes = out_dim as usize * 4;
        let mut out_raw = vec![0u8; out_bytes];
        stdout.read_exact(&mut out_raw)
            .context("Failed to read worker output data")?;

        let out: Vec<f32> = out_raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        Ok(out)
    }

    /// Stop the NPU worker.
    #[allow(dead_code)]
    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            // Send shutdown signal (op=0 means quit)
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(&0u32.to_le_bytes());
                let _ = stdin.flush();
            }
            let _ = child.kill();
            let _ = child.wait();
            info!("NPU worker stopped");
        }
    }

    /// Check if the worker is running.
    #[allow(dead_code)]
    pub fn is_running(&mut self) -> bool {
        self.child.as_mut().map_or(false, |c| {
            c.try_wait().map(|s| s.is_none()).unwrap_or(false)
        })
    }
}
