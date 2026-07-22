//! Full transformer inference loop.
//!
//! CPU handles: RoPE, attention, RMSNorm, SiLU, KV cache, sampling
//! NPU handles: QKV, O, GU, D GEMM projections (via NpuWorker)
//!
//! Architecture matches the Python `Inference` class exactly.

use crate::q4nx::Q4nxReader;
use crate::worker::{GemmOp, NpuWorker};
use anyhow::Result;
use std::f32::consts::PI;
use tracing::info;

pub const EPS: f32 = 1e-6;
pub const EOS_ID: u32 = 2; // matches Python

/// Model configuration constants (Qwen3-0.6B architecture).
pub struct ModelConfig {
    pub h: usize,      // hidden size
    pub nc: usize,     // num layers
    pub nh: usize,     // num heads
    pub nkv: usize,    // num kv heads
    pub hd: usize,     // head dim
    pub im: usize,     // intermediate size (ffn hidden)
    pub nv: usize,     // vocab size
    pub max_seq: usize, // max sequence length
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            h: 1024,
            nc: 28,
            nh: 16,
            nkv: 8,
            hd: 128,
            im: 3072,
            nv: 151936,
            max_seq: 4096,
        }
    }
}

impl ModelConfig {
    pub fn qd(&self) -> usize {
        self.nh * self.hd
    }
    pub fn kd(&self) -> usize {
        self.nkv * self.hd
    }
    pub fn gqa(&self) -> usize {
        self.nh / self.nkv
    }
}

/// Full transformer inference engine.
pub struct Inference {
    pub cfg: ModelConfig,
    worker: NpuWorker,

    // Weights loaded from Q4NX
    embed: Vec<f32>,           // [nv, h]
    final_norm: Vec<f32>,       // [h]
    lm_emb: Vec<f32>,           // [nv, h] — might be same as embed
    input_norms: Vec<Vec<f32>>, // [nc][h]
    post_attn_norms: Vec<Vec<f32>>, // [nc][h]

    // KV cache
    k_cache: Vec<Vec<Vec<f32>>>, // [nc][max_seq][nkv*hd]
    v_cache: Vec<Vec<Vec<f32>>>, // [nc][max_seq][nkv*hd]
    pos: usize,

    // Precomputed RoPE
    cos_c: Vec<f32>, // [max_seq, hd]
    sin_c: Vec<f32>, // [max_seq, hd]
}

impl Inference {
    /// Load weights and initialize the inference engine.
    pub fn new(worker: NpuWorker, model: &Q4nxReader, cfg: ModelConfig) -> Result<Self> {
        let nv = cfg.nv;
        let h = cfg.h;
        let nc = cfg.nc;
        let hd = cfg.hd;
        let nkv = cfg.nkv;

        // Load weights
        let embed = model.read_bf16("model.embed_tokens.weight", nv * h)
            .unwrap_or_else(|| vec![0.0; nv * h]);
        let final_norm = model.read_bf16("model.norm.weight", h)
            .unwrap_or_else(|| vec![0.0; h]);

        let lm_emb = if model.has_tensor("lm_head.weight") {
            model.read_bf16("lm_head.weight", nv * h)
                .unwrap_or_else(|| embed.clone())
        } else {
            embed.clone()
        };

        let mut input_norms = Vec::with_capacity(nc);
        let mut post_attn_norms = Vec::with_capacity(nc);
        for i in 0..nc {
            let in_n = model.read_bf16(&format!("model.layers.{i}.input_layernorm.weight"), h)
                .unwrap_or_else(|| vec![0.0; h]);
            let pa_n = model.read_bf16(&format!("model.layers.{i}.post_attention_layernorm.weight"), h)
                .unwrap_or_else(|| vec![0.0; h]);
            input_norms.push(in_n);
            post_attn_norms.push(pa_n);
        }

        // Precompute KV cache
        let max_seq = cfg.max_seq;
        let kv_slot = nkv * hd;
        let k_cache = vec![vec![vec![0.0f32; kv_slot]; max_seq]; nc];
        let v_cache = vec![vec![vec![0.0f32; kv_slot]; max_seq]; nc];

        // Precompute RoPE
        let (cos_c, sin_c) = Self::build_rope(max_seq, hd);

        info!(
            "Inference ready: {} layers, {} heads, H={}, {} params",
            nc, cfg.nh, h, format_params(nc, h, cfg.im, nv)
        );

        Ok(Self {
            cfg,
            worker,
            embed,
            final_norm,
            lm_emb,
            input_norms,
            post_attn_norms,
            k_cache,
            v_cache,
            pos: 0,
            cos_c,
            sin_c,
        })
    }

    /// Precompute RoPE cos/sin tables.
    fn build_rope(max_seq: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
        let hd2 = hd / 2;
        let mut cos_c = vec![0.0f32; max_seq * hd];
        let mut sin_c = vec![0.0f32; max_seq * hd];

        for p in 0..max_seq {
            for d in 0..hd2 {
                let theta = 1_000_000.0_f32.powf(-2.0 * d as f32 / hd as f32);
                let angle = p as f32 * theta;
                let c = angle.cos();
                let s = angle.sin();
                cos_c[p * hd + d] = c;
                sin_c[p * hd + d] = s;
                cos_c[p * hd + d + hd2] = c;
                sin_c[p * hd + d + hd2] = s;
            }
        }

        (cos_c, sin_c)
    }

    /// Apply rotary position embedding to query or key.
    /// x is [nh or nkv, hd], modifies in-place.
    fn rope(&self, x: &mut [f32], n_heads: usize, hd: usize, pos: usize) {
        let hd2 = hd / 2;
        let cos_row = &self.cos_c[pos * hd..(pos + 1) * hd];
        let sin_row = &self.sin_c[pos * hd..(pos + 1) * hd];

        for h in 0..n_heads {
            let base = h * hd;
            // first half
            for d in 0..hd2 {
                let a = x[base + d];
                let b = x[base + d + hd2];
                x[base + d] = a * cos_row[d] - b * sin_row[d];
                x[base + d + hd2] = a * sin_row[d] + b * cos_row[d];
            }
        }
    }

    /// RMS normalization.
    fn rmsnorm(x: &[f32], w: &[f32]) -> Vec<f32> {
        let n = x.len();
        let mut ss = 0.0f32;
        for &v in x {
            ss += v * v;
        }
        ss = ss / n as f32;
        let rms = 1.0 / (ss + EPS).sqrt();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(x[i] * rms * w[i]);
        }
        out
    }

    /// CPU attention: Q @ K^T → softmax → @ V.
    /// q is [nh, hd], returns [nh, hd] flat.
    fn attn(&self, q: &[f32], layer: usize, cl: usize) -> Vec<f32> {
        let nh = self.cfg.nh;
        let _nkv = self.cfg.nkv;
        let hd = self.cfg.hd;
        let gqa = self.cfg.gqa();

        let mut out = vec![0.0f32; nh * hd];

        for h in 0..nh {
            let kvh = h / gqa; // which KV head this query head maps to

            // Score: q[h] @ k_cache[0..cl, kvh]
            let mut scores = vec![0.0f32; cl];
            for t in 0..cl {
                let k_slice = &self.k_cache[layer][t];
                let k_start = kvh * hd;
                let mut dot = 0.0;
                for d in 0..hd {
                    dot += q[h * hd + d] * k_slice[k_start + d];
                }
                scores[t] = dot / (hd as f32).sqrt();
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = ((*s) - max_s).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / (sum_exp + 1e-10);
            for s in &mut scores {
                *s *= inv_sum;
            }

            // Weighted sum of V
            for d in 0..hd {
                let mut val = 0.0;
                for t in 0..cl {
                    val += scores[t] * self.v_cache[layer][t][kvh * hd + d];
                }
                out[h * hd + d] = val;
            }
        }

        out
    }

    /// SiLU activation: x * sigmoid(x)
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Process one token, update KV cache, return logits.
    fn forward(&mut self, token: u32) -> Vec<f32> {
        let cfg = &self.cfg;
        let h = cfg.h;

        // Embedding lookup
        let token_idx = token as usize;
        let mut hidden = if token_idx * h < self.embed.len() {
            self.embed[token_idx * h..(token_idx + 1) * h].to_vec()
        } else {
            vec![0.0; h]
        };

        for l in 0..cfg.nc {
            let residual = hidden.clone();

            // RMSNorm
            hidden = Self::rmsnorm(&hidden, &self.input_norms[l]);

            // QKV on NPU
            let qkv = self.worker
                .gemm(GemmOp::Qkv, l as u32, 1, h as u32, &hidden)
                .expect("QKV GEMM failed");

            let qd = cfg.qd();
            let kd = cfg.kd();
            let q = &qkv[..qd];
            let k = &qkv[qd..qd + kd];
            let v = &qkv[qd + kd..];

            // Copy to mutable arrays for RoPE
            let mut q_mut = q.to_vec();
            let mut k_mut = k.to_vec();

            // RoPE
            self.rope(&mut q_mut, cfg.nh, cfg.hd, self.pos);
            self.rope(&mut k_mut, cfg.nkv, cfg.hd, self.pos);

            // Store KV
            self.k_cache[l][self.pos].copy_from_slice(&k_mut);
            self.v_cache[l][self.pos].copy_from_slice(v);
            let cl = self.pos + 1;

            // CPU Attention
            let attn_out = self.attn(&q_mut, l, cl);

            // O projection on NPU
            let oo = self.worker
                .gemm(GemmOp::O, l as u32, 1, qd as u32, &attn_out)
                .expect("O GEMM failed");

            // Residual add
            for i in 0..h {
                hidden[i] = residual[i] + oo[i];
            }

            // FFN
            let ffn_residual = hidden.clone();

            hidden = Self::rmsnorm(&hidden, &self.post_attn_norms[l]);

            // Gate+Up projection on NPU
            let gu = self.worker
                .gemm(GemmOp::GateUp, l as u32, 1, h as u32, &hidden)
                .expect("GU GEMM failed");

            let im = cfg.im;
            let gate = &gu[..im];
            let up = &gu[im..];

            // SiLU(gate) * up
            let mut activated = Vec::with_capacity(im);
            for i in 0..im {
                activated.push(Self::silu(gate[i]) * up[i]);
            }

            // Down projection on NPU
            let dw = self.worker
                .gemm(GemmOp::Down, l as u32, 1, im as u32, &activated)
                .expect("Down GEMM failed");

            // Residual add
            for i in 0..h {
                hidden[i] = ffn_residual[i] + dw[i];
            }
        }

        self.pos += 1;

        // Final RMSNorm
        hidden = Self::rmsnorm(&hidden, &self.final_norm);

        // LM head: hidden @ lm_head^T
        let nv = cfg.nv;
        let mut logits = vec![0.0f32; nv];
        for v in 0..nv {
            let mut dot = 0.0;
            for d in 0..h {
                dot += hidden[d] * self.lm_emb[v * h + d];
            }
            logits[v] = dot;
        }

        logits
    }

    /// Generate tokens from input tokens. Uses greedy sampling.
    pub fn generate(&mut self, input_tokens: &[u32], max_new: usize) -> Vec<u32> {
        self.pos = 0;

        // Clamp to remaining KV cache
        let remaining = self.cfg.max_seq.saturating_sub(input_tokens.len());
        if remaining == 0 {
            return vec![];
        }
        let max_new = max_new.min(remaining);

        // Initialize KV cache with input tokens (prefill)
        // We need to run forward for each input token to fill the KV cache.
        // Python version just calls forward(last) per token, so input tokens
        // before the last are just used for KV cache fill.
        let mut result: Vec<u32> = Vec::new();
        let mut last = *input_tokens.first().unwrap_or(&0);

        // Prefill: process all but the last token for KV cache buildup
        // Actually the Python version does it differently — it just calls
        // forward(last) repeatedly and the KV cache is built incrementally.
        // But for prefill, we need to process each token in sequence.
        for &t in input_tokens {
            let _logits = self.forward(t);
            last = t;
        }

        // Generation loop
        for _ in 0..max_new {
            let logits = self.forward(last);

            // Greedy sampling: argmax
            let next_tok = logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            result.push(next_tok);
            if next_tok == EOS_ID {
                break;
            }
            last = next_tok;
        }

        result
    }
}

fn format_params(nc: usize, h: usize, im: usize, nv: usize) -> String {
    // Rough param count: embed + (QKV + O + GU + D) * layers + norm + lm_head
    let embed = nv * h;
    let layer = 4 * h * im + 3 * h * h; // approx
    let total = embed + layer * nc + h + nv * h;
    format!("{:.1}B", total as f64 / 1e9)
}
