//! Token-at-a-time autoregressive decode loop for the OGA graph.
//!
//! # Graph contract (TriLM 3.9B int4 int-4 MatMulNBits, 30 layers)
//!
//! Inputs (62 total):
//!
//! | Name                             | Shape                             | Dtype |
//! |----------------------------------|-----------------------------------|-------|
//! | `input_ids`                      | `[B, seq]`                        | i64   |
//! | `attention_mask`                 | `[B, total_seq]`                  | i64   |
//! | `past_key_values.{0..L-1}.key`   | `[B, kv_heads, past_seq, head]`   | f32   |
//! | `past_key_values.{0..L-1}.value` | `[B, kv_heads, past_seq, head]`   | f32   |
//!
//! Outputs (61 total):
//!
//! | Name                             | Shape                             | Dtype |
//! |----------------------------------|-----------------------------------|-------|
//! | `logits`                         | `[B, seq, vocab]`                 | f32   |
//! | `present.{0..L-1}.key`           | `[B, kv_heads, total_seq, head]`  | f32   |
//! | `present.{0..L-1}.value`         | `[B, kv_heads, total_seq, head]`  | f32   |
//!
//! Prefill: `input_ids = [1, P]`, all `past` tensors empty (`past_seq = 0`).
//! Decode step: `input_ids = [1, 1]`, `past` = `present` from previous call,
//! `attention_mask` grows monotonically.
//!
//! # Scope of this module
//!
//! * Greedy + temperature+top-k sampling (no beam, no nucleus, no
//!   repetition penalty — the halo stack sits on top of a sampler we
//!   already trust in `1bit-core`; we do NOT reinvent it).
//! * Single-request path, no batching. The server serializes behind a
//!   mutex same as the HIP lane — one decode at a time.
//! * No streaming surface. A streaming variant plugs in later; start
//!   with a blocking `generate` that's easy to reason about.

use std::time::Instant;

use ndarray::{Array2, Array4, ArrayViewD, Ix4, IxDyn};
use ort::value::Value;

use crate::error::OnnxError;
use crate::session::OnnxSession;

/// Request shape for [`OnnxSession::generate`].
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Raw prompt string. Tokenized with the crate's HF tokenizer — caller
    /// is responsible for any chat template they need.
    pub prompt: String,
    /// Maximum number of tokens to generate AFTER the prompt. Hard capped
    /// by the graph's exported `context_length`.
    pub max_new_tokens: usize,
    /// `0.0` for pure greedy (deterministic). Values >0 enable
    /// temperature-scaled top-k sampling.
    pub temperature: f32,
    /// Optional top-k truncation. `None` = full vocab.
    pub top_k: Option<u32>,
    /// RNG seed for reproducible sampling runs. Ignored when greedy.
    pub seed: Option<u64>,
}

impl GenerateRequest {
    /// Convenience constructor for a deterministic greedy run.
    pub fn greedy(prompt: impl Into<String>, max_new_tokens: usize) -> Self {
        Self {
            prompt: prompt.into(),
            max_new_tokens,
            temperature: 0.0,
            top_k: None,
            seed: None,
        }
    }
}

/// Response shape for [`OnnxSession::generate`].
#[derive(Debug, Clone)]
pub struct GenerateResponse {
    /// The generated continuation, decoded back to a `String`.
    pub text: String,
    /// Number of tokens consumed from the prompt (post-tokenization).
    pub prompt_tokens: usize,
    /// Number of tokens actually generated (may be less than `max_new_tokens`
    /// if EOS fired).
    pub completion_tokens: usize,
    /// Wall clock of the full generate call (prefill + decode).
    pub wall_ms: u128,
    /// Tok/s computed from `completion_tokens / (wall_ms / 1000)`.
    /// Zero if `wall_ms == 0` (can happen in degenerate tests).
    pub tokens_per_second: f32,
}

impl OnnxSession {
    /// Run `req` against the loaded graph.
    ///
    /// Errors:
    ///
    /// * [`OnnxError::SessionInit`] if the session was loaded via
    ///   [`OnnxSession::load_config_only`] (no ORT session to run).
    /// * [`OnnxError::SessionInit`] wrapping the underlying ORT error if a
    ///   forward pass returns an error. We don't invent a new variant —
    ///   session-time failures all route through the same surface.
    /// * [`OnnxError::TokenizerLoad`] wrapping encode/decode errors from
    ///   the `tokenizers` crate.
    pub fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse, OnnxError> {
        // Borrow the immutable pieces up front so we don't fight the
        // borrow checker once the mutable session handle is in flight.
        let layers = self.config.model.decoder.num_hidden_layers;
        let kv_heads = self.config.model.decoder.num_key_value_heads;
        let head_size = self.config.model.decoder.head_size;
        let eos = self.config.model.eos_token_id as i64;
        let ctx_cap = self.config.model.context_length;
        let encoded = self
            .tokenizer
            .encode(req.prompt.as_str(), true)
            .map_err(|e| OnnxError::TokenizerLoad(e.to_string()))?;

        let session = self.session_mut().ok_or_else(|| {
            OnnxError::SessionInit(
                "session is config-only; call load() not load_config_only()".into(),
            )
        })?;

        let t0 = Instant::now();

        let mut tokens: Vec<i64> = encoded.get_ids().iter().map(|&t| t as i64).collect();
        let prompt_tokens = tokens.len();

        // Empty KV for prefill. past_seq == 0, same dtype (f32) as the
        // present.* outputs we'll swap in on subsequent steps.
        let mut past_k: Vec<Array4<f32>> = (0..layers)
            .map(|_| Array4::<f32>::zeros((1, kv_heads, 0, head_size)))
            .collect();
        let mut past_v: Vec<Array4<f32>> = (0..layers)
            .map(|_| Array4::<f32>::zeros((1, kv_heads, 0, head_size)))
            .collect();

        let mut step_ids: Array2<i64> =
            Array2::from_shape_vec((1, tokens.len()), tokens.clone()).expect("prefill ids shape");

        let mut rng_state = req.seed.unwrap_or(0x5_eedb_17c0_ffee_u64);

        let mut completion_tokens = 0usize;
        let mut new_tokens: Vec<i64> = Vec::with_capacity(req.max_new_tokens);

        for _step in 0..req.max_new_tokens {
            let total_seq = tokens.len();
            if total_seq >= ctx_cap {
                break;
            }

            let attn_mask: Array2<i64> = Array2::from_elem((1, total_seq), 1i64);

            let mut inputs: Vec<(String, Value)> = Vec::with_capacity(2 + 2 * layers);
            inputs.push(("input_ids".into(), to_value_i64(&step_ids)?));
            inputs.push(("attention_mask".into(), to_value_i64(&attn_mask)?));
            for layer in 0..layers {
                inputs.push((
                    format!("past_key_values.{layer}.key"),
                    to_value_f32(&past_k[layer])?,
                ));
                inputs.push((
                    format!("past_key_values.{layer}.value"),
                    to_value_f32(&past_v[layer])?,
                ));
            }

            let outputs = session
                .run(inputs)
                .map_err(|e| OnnxError::SessionInit(format!("forward pass failed: {e}")))?;

            // Logits: [1, seq, vocab]. We only want the last position's row.
            let logits_val = outputs
                .get("logits")
                .ok_or_else(|| OnnxError::SessionInit("output 'logits' missing".into()))?;
            let logits_array = logits_val
                .try_extract_array::<f32>()
                .map_err(|e| OnnxError::SessionInit(format!("extract logits: {e}")))?;
            let logits_slice = last_row_logits(&logits_array)?;

            let next_token = sample_next(logits_slice, req.temperature, req.top_k, &mut rng_state);
            new_tokens.push(next_token);
            tokens.push(next_token);
            completion_tokens += 1;

            if next_token == eos {
                break;
            }

            // Step input for next iteration is just the one new token.
            step_ids = Array2::from_shape_vec((1, 1), vec![next_token]).expect("decode ids shape");

            // Rotate present.* into past.* for the next step.
            for layer in 0..layers {
                past_k[layer] = extract_kv(&outputs, &format!("present.{layer}.key"))?;
                past_v[layer] = extract_kv(&outputs, &format!("present.{layer}.value"))?;
            }
        }

        let text = self
            .tokenizer
            .decode(
                &new_tokens.iter().map(|&t| t as u32).collect::<Vec<_>>(),
                /* skip_special_tokens */ true,
            )
            .map_err(|e| OnnxError::TokenizerLoad(e.to_string()))?;

        let wall_ms = t0.elapsed().as_millis();
        let tokens_per_second = if wall_ms > 0 {
            (completion_tokens as f32) * 1000.0 / (wall_ms as f32)
        } else {
            0.0
        };

        Ok(GenerateResponse {
            text,
            prompt_tokens,
            completion_tokens,
            wall_ms,
            tokens_per_second,
        })
    }
}

/// Convert an `Array2<i64>` into an owned `ort::Value`. Copies — ORT
/// requires an owned buffer for non-DML lanes.
fn to_value_i64(arr: &Array2<i64>) -> Result<Value, OnnxError> {
    Value::from_array(arr.clone())
        .map(Value::from)
        .map_err(|e| OnnxError::SessionInit(format!("pack i64: {e}")))
}

/// Convert an `Array4<f32>` into an owned `ort::Value`.
fn to_value_f32(arr: &Array4<f32>) -> Result<Value, OnnxError> {
    Value::from_array(arr.clone())
        .map(Value::from)
        .map_err(|e| OnnxError::SessionInit(format!("pack f32: {e}")))
}

/// Pull the last row (`[vocab]`) out of a `[1, seq, vocab]` logits tensor.
///
/// We read only the tail because decode sampling is over the last
/// prompt/step position; everything else is wasted by the caller.
fn last_row_logits(arr: &ArrayViewD<'_, f32>) -> Result<Vec<f32>, OnnxError> {
    let shape = arr.shape();
    if shape.len() != 3 {
        return Err(OnnxError::SessionInit(format!(
            "logits must be rank-3, got {:?}",
            shape
        )));
    }
    let seq = shape[1];
    let vocab = shape[2];
    if seq == 0 {
        return Err(OnnxError::SessionInit("logits seq dim is 0".into()));
    }
    let last = seq - 1;
    let mut out = Vec::with_capacity(vocab);
    for v in 0..vocab {
        out.push(arr[IxDyn(&[0, last, v])]);
    }
    Ok(out)
}

/// Pull a KV tensor out of the outputs map and drop it into an
/// `Array4<f32>` matching the `[B, kv_heads, total_seq, head]` contract.
fn extract_kv(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Array4<f32>, OnnxError> {
    let val = outputs
        .get(name)
        .ok_or_else(|| OnnxError::SessionInit(format!("output '{name}' missing")))?;
    let arr = val
        .try_extract_array::<f32>()
        .map_err(|e| OnnxError::SessionInit(format!("extract {name}: {e}")))?;
    let shape = arr.shape();
    if shape.len() != 4 {
        return Err(OnnxError::SessionInit(format!(
            "KV tensor {name} must be rank-4, got {:?}",
            shape
        )));
    }
    arr.to_owned()
        .into_dimensionality::<Ix4>()
        .map_err(|e| OnnxError::SessionInit(format!("reshape {name}: {e}")))
}

/// Greedy argmax when `temperature == 0.0`, otherwise temperature-scaled
/// softmax + top-k cutoff + categorical sample.
///
/// `rng_state` is a mutable xorshift64 state — deterministic per-request
/// when the caller supplies a seed.
fn sample_next(mut logits: Vec<f32>, temperature: f32, top_k: Option<u32>, rng: &mut u64) -> i64 {
    if temperature <= 0.0 {
        return argmax(&logits) as i64;
    }

    // Temperature scale in-place.
    let inv_t = 1.0f32 / temperature;
    for v in logits.iter_mut() {
        *v *= inv_t;
    }

    // Top-k cutoff: find the k-th largest logit, mask anything below to -inf.
    if let Some(k) = top_k {
        let k = (k as usize).min(logits.len()).max(1);
        if k < logits.len() {
            let mut sorted = logits.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let cutoff = sorted[k - 1];
            for v in logits.iter_mut() {
                if *v < cutoff {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Softmax.
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum == 0.0 || !sum.is_finite() {
        return argmax(&logits) as i64;
    }
    for v in logits.iter_mut() {
        *v /= sum;
    }

    // Categorical sample via inverse CDF.
    let r = next_f32(rng);
    let mut acc = 0.0f32;
    for (i, &p) in logits.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i as i64;
        }
    }
    (logits.len() - 1) as i64
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// xorshift64 → uniform f32 in [0, 1).
fn next_f32(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = if x == 0 { 0x1234_5678_9abc_def0 } else { x };
    (x >> 40) as f32 / ((1u64 << 24) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_global_max() {
        assert_eq!(argmax(&[0.1, 0.9, 0.4, -1.0]), 1);
    }

    #[test]
    fn greedy_constructor_is_deterministic() {
        let r = GenerateRequest::greedy("hello", 16);
        assert_eq!(r.temperature, 0.0);
        assert_eq!(r.top_k, None);
        assert_eq!(r.seed, None);
        assert_eq!(r.max_new_tokens, 16);
    }

    #[test]
    fn sample_next_greedy_matches_argmax() {
        let mut rng = 1u64;
        let logits = vec![0.1, 0.9, 0.4, -1.0];
        assert_eq!(sample_next(logits, 0.0, None, &mut rng), 1);
    }

    #[test]
    fn sample_next_deterministic_with_seed() {
        let logits = vec![0.1, 0.9, 0.4, -1.0];
        let mut rng_a = 42u64;
        let mut rng_b = 42u64;
        let a = sample_next(logits.clone(), 1.0, Some(2), &mut rng_a);
        let b = sample_next(logits, 1.0, Some(2), &mut rng_b);
        assert_eq!(a, b, "same seed must produce same token");
    }
}
