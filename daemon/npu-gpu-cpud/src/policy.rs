//! Policy engine — model size estimation and device selection.
//!
//! Policy (model_size → device):
//!   < 2B params  → NPU (lowest power)
//!   >= 2B params → GPU (fastest compute; also the >8B fallback until a
//!                  real CPU backend exists — see #147)

use std::fmt;

/// Available devices for routing inference requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Npu,
    Gpu,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Npu => write!(f, "npu"),
            Device::Gpu => write!(f, "gpu"),
        }
    }
}

/// Estimate model size in billions of parameters from its name.
///
/// Matches known patterns like "0.6b", "7b", "70b" etc.
/// Falls back to 7B for unknown models.
pub fn estimate_model_size(model_name: &str) -> f64 {
    let lower = model_name.to_lowercase();

    // Check patterns from largest to smallest for correctness
    // (e.g., "70b" must match before "7b")
    let patterns: &[(&str, f64)] = &[
        ("70b", 70.0),
        ("34b", 34.0),
        ("20b", 20.0),
        ("14b", 14.0),
        ("13b", 13.0),
        ("9b", 9.0),
        ("8b", 8.0),
        ("7b", 7.0),
        ("4b", 4.0),
        ("3b", 3.0),
        ("2.6b", 2.6),
        ("2b", 2.0),
        ("1.7b", 1.7),
        ("1.5b", 1.5),
        ("1.2b", 1.2),
        ("1b", 1.0),
        ("0.8b", 0.8),
        ("0.6b", 0.6),
        ("0.5b", 0.5),
    ];

    for (pattern, size) in patterns {
        if lower.contains(pattern) {
            return *size;
        }
    }

    // Default fallback
    7.0
}

/// Select device based on model size.
///
/// NOTE: there is no CPU backend implemented. Models >8B are
/// routed to GPU rather than a nonexistent CPU path (#147).
pub fn select_device(model_size_b: f64) -> Device {
    if model_size_b < 2.0 {
        Device::Npu
    } else {
        Device::Gpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_model_size() {
        assert_eq!(estimate_model_size("Qwen3-0.6B-NPU2"), 0.6);
        assert_eq!(estimate_model_size("Qwen3-8B"), 8.0);
        assert_eq!(estimate_model_size("Llama-3.1-70B"), 70.0);
        assert_eq!(estimate_model_size("unknown-model"), 7.0);
        assert_eq!(estimate_model_size("Gemma4-2B-IT"), 2.0);
        assert_eq!(estimate_model_size("tiny-0.5b"), 0.5);
        assert_eq!(estimate_model_size("zaya-1.5b"), 1.5);
    }

    #[test]
    fn test_select_device() {
        assert_eq!(select_device(0.6), Device::Npu);
        assert_eq!(select_device(1.9), Device::Npu);
        assert_eq!(select_device(2.0), Device::Gpu);
        assert_eq!(select_device(8.0), Device::Gpu);
        assert_eq!(select_device(70.0), Device::Gpu);
    }
}
