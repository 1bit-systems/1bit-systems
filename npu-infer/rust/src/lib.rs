//! Rust bindings to the NPU inference engine (XDNA2 NPU on Strix Halo).
//!
//! This crate links against the C++ NPU engine via `ffi_bridge.cpp` and exposes
//! a safe Rust API for loading models, running inference, and decoding tokens.
//!
//! # Example
//! ```no_run
//! use npu_engine_sys::NpuEngine;
//!
//! let mut engine = NpuEngine::new("/path/to/model.q4nx").unwrap();
//! let output = engine.generate(&[151643], 16).unwrap();
//! println!("Generated {} tokens: {:?}", output.len(), &output);
//! ```

use std::ffi::CString;
use std::ptr;

// ── Raw FFI declarations ────────────────────────────────────────

#[repr(C)]
struct NpuEngineHandle {
    _data: [u8; 0],
    _marker: std::marker::PhantomData<(*mut u8, std::marker::PhantomPinned)>,
}

extern "C" {
    fn npu_engine_create() -> *mut NpuEngineHandle;
    fn npu_engine_destroy(h: *mut NpuEngineHandle) -> i32;
    fn npu_engine_init(h: *mut NpuEngineHandle, model_path: *const i8) -> i32;
    fn npu_engine_is_initialized(h: *mut NpuEngineHandle) -> i32;
    fn npu_engine_last_error(h: *mut NpuEngineHandle) -> *const i8;
    fn npu_engine_generate(
        h: *mut NpuEngineHandle,
        input_tokens: *const i32,
        num_input_tokens: i32,
        output_tokens: *mut i32,
        max_output_tokens: i32,
    ) -> i32;
    fn npu_engine_get_config(
        h: *mut NpuEngineHandle,
        out_vocab_size: *mut i32,
        out_hidden_size: *mut i32,
        out_num_layers: *mut i32,
    ) -> i32;
}

// ── Safe Rust wrapper ───────────────────────────────────────────

/// Safe wrapper around the NPU inference engine.
pub struct NpuEngine {
    handle: *mut NpuEngineHandle,
}

unsafe impl Send for NpuEngine {}
unsafe impl Sync for NpuEngine {}

impl NpuEngine {
    /// Create a new NPU engine and initialize it with the given model.
    pub fn new(model_path: &str) -> Result<Self, String> {
        let handle = unsafe { npu_engine_create() };
        if handle.is_null() {
            return Err("Failed to create engine handle".into());
        }

        let cpath = CString::new(model_path)
            .map_err(|_| "Model path contains null byte".to_string())?;

        let ret = unsafe { npu_engine_init(handle, cpath.as_ptr()) };
        if ret != 0 {
            let err = unsafe {
                let p = npu_engine_last_error(handle);
                if p.is_null() {
                    "unknown error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            unsafe { npu_engine_destroy(handle); }
            return Err(err);
        }

        Ok(NpuEngine { handle })
    }

    /// Check if the engine is initialized.
    pub fn is_initialized(&self) -> bool {
        unsafe { npu_engine_is_initialized(self.handle) != 0 }
    }

    /// Run inference on a prompt, generating output tokens.
    ///
    /// `input_tokens` — token IDs for the prompt.
    /// `max_output_tokens` — maximum number of tokens to generate.
    ///
    /// Returns the generated token IDs.
    pub fn generate(&self, input_tokens: &[i32], max_output_tokens: usize) -> Result<Vec<i32>, String> {
        if input_tokens.is_empty() {
            return Err("Empty input".into());
        }
        if max_output_tokens == 0 {
            return Ok(vec![]);
        }

        // Allocate output buffer (cap at 4096 for safety)
        let max_out = max_output_tokens.min(4096);
        let mut output = vec![0i32; max_out];

        let ret = unsafe {
            npu_engine_generate(
                self.handle,
                input_tokens.as_ptr(),
                input_tokens.len() as i32,
                output.as_mut_ptr(),
                max_out as i32,
            )
        };

        if ret < 0 {
            let err = unsafe {
                let p = npu_engine_last_error(self.handle);
                if p.is_null() {
                    "generate failed".to_string()
                } else {
                    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            return Err(err);
        }

        output.truncate(ret as usize);
        Ok(output)
    }

    /// Get model configuration.
    pub fn config(&self) -> Result<ModelConfig, String> {
        let mut vocab = 0i32;
        let mut hidden = 0i32;
        let mut layers = 0i32;
        unsafe {
            npu_engine_get_config(self.handle, &mut vocab, &mut hidden, &mut layers);
        }
        Ok(ModelConfig {
            vocab_size: vocab as usize,
            hidden_size: hidden as usize,
            num_layers: layers as usize,
        })
    }
}

impl Drop for NpuEngine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { npu_engine_destroy(self.handle); }
            self.handle = ptr::null_mut();
        }
    }
}

/// Model configuration parameters.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_not_initialized() {
        // Just test the create/destroy lifecycle
        let handle = unsafe { npu_engine_create() };
        assert!(!handle.is_null());
        assert_eq!(unsafe { npu_engine_is_initialized(handle) }, 0);
        unsafe { npu_engine_destroy(handle); }
    }

    #[test]
    fn test_engine_init_fails_bad_path() {
        let result = NpuEngine::new("/nonexistent/model.q4nx");
        assert!(result.is_err());
    }
}
