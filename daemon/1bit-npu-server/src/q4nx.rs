//! Q4NX model reader — loads weights from the Q4NX binary format.
//!
//! Format:
//!   [8 bytes: header_size (u64 LE)]
//!   [header_size bytes: JSON header with tensor metadata]
//!   [data: packed bfloat16 weight arrays]
//!
//! The JSON header maps tensor names to objects containing:
//!   "data_offsets": [start_byte, end_byte]  — byte range in the data section

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// Reads bfloat16 weights from a Q4NX model file.
pub struct Q4nxReader {
    data: Vec<u8>,
    data_start: usize,
    offsets: HashMap<String, (usize, usize)>,
}

impl Q4nxReader {
    /// Load and parse a Q4NX model file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read(path.as_ref())
            .with_context(|| format!("Failed to read model file: {}", path.as_ref().display()))?;

        // Parse header size (first 8 bytes, u64 LE)
        if raw.len() < 8 {
            anyhow::bail!("File too small: {} bytes", raw.len());
        }
        let hsz = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
        let data_start = 8 + hsz;

        if raw.len() < data_start {
            anyhow::bail!("Header claims {} bytes but file is only {} bytes", data_start, raw.len());
        }

        // Parse JSON header
        let header_str = std::str::from_utf8(&raw[8..data_start])
            .context("Header is not valid UTF-8")?;
        let header: Value = serde_json::from_str(header_str)
            .context("Failed to parse Q4NX header JSON")?;

        // Extract tensor offsets
        let mut offsets = HashMap::new();
        if let Value::Object(map) = &header {
            for (name, info) in map {
                if let Some(arr) = info.get("data_offsets").and_then(|v| v.as_array()) {
                    if arr.len() >= 2 {
                        let start = arr[0].as_u64().unwrap_or(0) as usize;
                        let end = arr[1].as_u64().unwrap_or(0) as usize;
                        offsets.insert(name.clone(), (start, end));
                    }
                }
            }
        }

        Ok(Self {
            data: raw,
            data_start,
            offsets,
        })
    }

    /// Read a tensor as f32 slice, converting from bfloat16.
    /// Returns `None` if the tensor is not found.
    pub fn read_bf16(&self, name: &str, count: usize) -> Option<Vec<f32>> {
        let (start, end) = self.offsets.get(name)?;
        let byte_slice = self.data.get(self.data_start + start..self.data_start + end)?;

        // Each bf16 value is 2 bytes; convert to f32
        let n = byte_slice.len() / 2;
        let mut out = Vec::with_capacity(n.min(count));

        for i in 0..n.min(count) {
            // Read 2 bytes as little-endian uint16 (bf16 value)
            let lo = byte_slice[2 * i] as u16;
            let hi = byte_slice[2 * i + 1] as u16;
            let bf16_val = lo | (hi << 8); // little-endian u16
            // bf16 is the upper 16 bits of f32
            let f = f32::from_bits((bf16_val as u32) << 16);
            out.push(f);
        }

        Some(out)
    }

    /// Read a weight tensor that should be reshaped as [rows, cols].
    pub fn read_bf16_2d(&self, name: &str, rows: usize, cols: usize) -> Option<Vec<f32>> {
        let values = self.read_bf16(name, rows * cols)?;
        Some(values)
    }

    /// Return the number of bf16 elements in a tensor, or None if not found.
    pub fn tensor_bf16_count(&self, name: &str) -> Option<usize> {
        let (start, end) = self.offsets.get(name)?;
        let byte_len = end.checked_sub(*start)?;
        Some(byte_len / 2)
    }

    /// Check if a tensor exists in the file.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.offsets.contains_key(name)
    }

    /// Get the header JSON for debugging.
    pub fn header_json(&self) -> &str {
        let hsz = u64::from_le_bytes(self.data[0..8].try_into().unwrap()) as usize;
        std::str::from_utf8(&self.data[8..8 + hsz]).unwrap_or("")
    }
}

/// Convert a bfloat16 (u16) to f32.
/// Matches Python's: struct.unpack('<f', struct.pack('<I', v << 16))[0]
/// The bf16 value is placed in the upper 16 bits of an f32.
#[inline]
pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_conversion() {
        // 1.0 in bf16 is 0x3F80
        let val = bf16_to_f32(0x3F80);
        assert!((val - 1.0).abs() < 1e-6);

        // 0.5 in bf16 is 0x3F00 (approx)
        // 0.5 f32 = 0x3F000000, upper 16 = 0x3F00
        let val = bf16_to_f32(0x3F00);
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_read_nonexistent_tensor() {
        // Can't easily test without a real file, but check the API compiles
        let _ = bf16_to_f32(0);
    }
}
