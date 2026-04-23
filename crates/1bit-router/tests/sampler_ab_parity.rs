//! CPU-lane sampler A/B parity — live-server integration test.
//!
//! Validates the 2026-04-20 sampler offload: one 64-token decode run
//! twice through [`Router`], once with `HALO_SAMPLER=inline` and once
//! with `HALO_SAMPLER=cpu`, must produce **byte-identical** output
//! under a fixed seed + `temperature=0`. This is the A/B regression
//! guard that underpins flipping the default from `Inline` to `Cpu`.
//!
//! Why `temperature=0`? Because the GPU returns the argmax directly
//! via `forward_token`'s return value, and both paths short-circuit
//! there: the sampler is skipped, so divergence would indicate a KV
//! cache bug or a routing bug — not a sampler bug. That's intentional
//! — it's the strongest possible parity claim (both paths must return
//! literally the same forward-pass outputs). A second optional
//! `temperature=1.0` pass exercises the full sampler offload; both
//! samplers are seeded identically from the same `SamplerConfig`, so
//! they must also agree draw-for-draw.
//!
//! # Run
//!
//! ```bash
//! cargo test -p onebit-router --release --features hip \
//!     --test sampler_ab_parity -- --ignored --nocapture
//! ```
//!
//! Needs ROCm + the halo-1bit-2b.h1b model on disk. Default CI run
//! skips (the `#[ignore]` attr + hip feature gate).

#![cfg(feature = "hip")]

use std::path::Path;

use onebit_core::sampler::SamplerConfig;
use onebit_router::{BackendKind, Router, RouterConfig, RouterRequest, SamplerMode, detect};

fn default_model() -> String {
    std::env::var("HOME")
        .map(|h| format!("{h}/1bit systems/models/halo-1bit-2b.h1b"))
        .unwrap_or_else(|_| "halo-1bit-2b.h1b".into())
}

fn default_htok(model_path: &str) -> String {
    model_path.replace(".h1b", ".htok")
}

/// Build a router with an explicit [`SamplerMode`]. Bypasses
/// [`RouterConfig::from_env`] because env mutation across test binaries
/// is racy even with a mutex — the in-memory config is cleaner.
fn router_with_mode(model_path: &Path, mode: SamplerMode) -> Router {
    let htok = default_htok(&model_path.to_string_lossy());
    let cfg = RouterConfig {
        backend: Default::default(),
        sampler_mode: mode,
    };
    Router::new_with_config(
        model_path,
        Path::new(&htok),
        "halo-1bit-2b".into(),
        4096,
        cfg,
    )
    .expect("router init")
}

/// 64-token decode with `temperature=0` (greedy) must match byte-for-byte
/// across `SamplerMode::Inline` and `SamplerMode::Cpu`. This is the
/// cutover gate for flipping the default.
#[test]
#[ignore = "requires ROCm + halo-1bit-2b.h1b on disk; run with --ignored"]
fn inline_and_cpu_agree_greedy_64_tokens() {
    let model_path = std::env::var("HALO_SMOKE_MODEL").unwrap_or_else(|_| default_model());
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: model not found at {model_path}");
        return;
    }
    assert_eq!(detect(), BackendKind::Hip, "expected HIP detection");

    let prompt = "The capital of France is";
    let sampler = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        rep_penalty: 1.0,
        rep_last_n: 0,
        seed: 0xCAFE_BABE,
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let inline_router = router_with_mode(Path::new(&model_path), SamplerMode::Inline);
    assert_eq!(inline_router.sampler_mode(), SamplerMode::Inline);
    let inline_resp = rt.block_on(async {
        inline_router
            .generate(RouterRequest {
                prompt: prompt.to_string(),
                max_new_tokens: 64,
                sampler,
                stop: Vec::new(),
            })
            .await
            .expect("inline generate")
    });
    drop(inline_router);

    let cpu_router = router_with_mode(Path::new(&model_path), SamplerMode::Cpu);
    assert_eq!(cpu_router.sampler_mode(), SamplerMode::Cpu);
    let cpu_resp = rt.block_on(async {
        cpu_router
            .generate(RouterRequest {
                prompt: prompt.to_string(),
                max_new_tokens: 64,
                sampler,
                stop: Vec::new(),
            })
            .await
            .expect("cpu generate")
    });
    drop(cpu_router);

    println!("inline_text = {:?}", inline_resp.text);
    println!("cpu_text    = {:?}", cpu_resp.text);
    println!(
        "inline_toks = {} / cpu_toks = {}",
        inline_resp.completion_tokens, cpu_resp.completion_tokens
    );

    assert_eq!(
        inline_resp.text, cpu_resp.text,
        "greedy decode must be byte-identical across sampler modes"
    );
    assert_eq!(
        inline_resp.completion_tokens, cpu_resp.completion_tokens,
        "completion token counts must match"
    );
    assert_eq!(
        inline_resp.prompt_tokens, cpu_resp.prompt_tokens,
        "prompt token counts must match"
    );
    assert!(
        inline_resp.completion_tokens >= 1,
        "expected at least one decoded token"
    );
}

/// Temperature=1.0 / top_k=50 / seed-pinned 32-token decode must also
/// agree byte-for-byte across modes — the full sampler path runs on
/// both sides and the RNG is seeded identically, so divergence here
/// would indicate the bounded-channel handoff corrupted scratch or
/// RNG state. Separate test so the primary greedy gate can pass even
/// if the stochastic-path test is flaky for unrelated reasons.
#[test]
#[ignore = "requires ROCm + halo-1bit-2b.h1b on disk; run with --ignored"]
fn inline_and_cpu_agree_stochastic_32_tokens() {
    let model_path = std::env::var("HALO_SMOKE_MODEL").unwrap_or_else(|_| default_model());
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: model not found at {model_path}");
        return;
    }
    assert_eq!(detect(), BackendKind::Hip, "expected HIP detection");

    let sampler = SamplerConfig {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.95,
        rep_penalty: 1.0,
        rep_last_n: 0,
        seed: 0xDEAD_BEEF,
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let prompt = "The capital of France is";

    let inline_router = router_with_mode(Path::new(&model_path), SamplerMode::Inline);
    let inline_resp = rt.block_on(async {
        inline_router
            .generate(RouterRequest {
                prompt: prompt.to_string(),
                max_new_tokens: 32,
                sampler,
                stop: Vec::new(),
            })
            .await
            .expect("inline generate")
    });
    drop(inline_router);

    let cpu_router = router_with_mode(Path::new(&model_path), SamplerMode::Cpu);
    let cpu_resp = rt.block_on(async {
        cpu_router
            .generate(RouterRequest {
                prompt: prompt.to_string(),
                max_new_tokens: 32,
                sampler,
                stop: Vec::new(),
            })
            .await
            .expect("cpu generate")
    });
    drop(cpu_router);

    assert_eq!(
        inline_resp.text, cpu_resp.text,
        "stochastic decode must be byte-identical under fixed seed across sampler modes"
    );
}
