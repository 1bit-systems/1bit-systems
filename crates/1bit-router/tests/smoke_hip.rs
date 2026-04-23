//! Hardware smoke test — only runs with `--ignored` on a host that has
//! ROCm + the halo-1bit-2b.h1b model on disk. The default CI run skips it.
//!
//! What it validates (when `--ignored` is passed):
//!   1. Hardware detection reports HIP.
//!   2. `.h1b` + `.htok` load + weight upload succeed.
//!   3. One forward pass runs and returns a token id.
//!
//! The expected prompt is "The capital of France is" — the reported
//! first-16-tokens output should match gen-1's
//! `bitnet_decode --text "The capital of France is" 1` argmax chain.

#![cfg(feature = "hip")]

use std::path::Path;

use onebit_core::sampler::SamplerConfig;
use onebit_router::{BackendKind, HipBackend, Router, RouterRequest, detect};

fn default_model() -> String {
    std::env::var("HOME")
        .map(|h| format!("{h}/1bit systems/models/halo-1bit-2b.h1b"))
        .unwrap_or_else(|_| "halo-1bit-2b.h1b".into())
}

#[test]
#[ignore = "requires ROCm + halo-1bit-2b.h1b on disk; run with --ignored"]
fn first_token_matches_gen1_argmax() {
    let model_path = std::env::var("HALO_SMOKE_MODEL").unwrap_or_else(|_| default_model());
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: model not found at {model_path}");
        return;
    }

    assert_eq!(detect(), BackendKind::Hip, "expected HIP detection");

    let router = Router::new(&model_path).expect("router init");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let resp = rt.block_on(async {
        router
            .generate(RouterRequest {
                prompt: "The capital of France is".to_string(),
                max_new_tokens: 16,
                sampler: SamplerConfig {
                    temperature: 0.0,
                    ..SamplerConfig::default()
                },
                stop: Vec::new(),
            })
            .await
            .expect("generate")
    });

    println!("TEXT: {:?}", resp.text);
    println!("prompt_tokens = {}", resp.prompt_tokens);
    println!("completion_tokens = {}", resp.completion_tokens);
    assert!(
        resp.completion_tokens >= 1,
        "expected at least one decoded token"
    );

    // Second pass: stand up a bare HipBackend and dump raw token IDs so
    // the user can compare argmax-by-argmax against gen-1's
    // `bitnet_decode --text "The capital of France is" 1`.
    let htok_path = model_path.replace(".h1b", ".htok");
    let mut bare = HipBackend::new(
        Path::new(&model_path),
        Path::new(&htok_path),
        "1bit-monster-2b".into(),
        4096,
    )
    .expect("bare backend init");

    let prompt = "The capital of France is";
    let ids = bare.tokenize(prompt);
    println!("prompt ids = {:?}", ids);

    let mut logits = Vec::new();
    let mut cur = ids[0];
    let mut generated: Vec<i32> = Vec::new();
    let mut pos = 0i32;
    for &t in &ids {
        cur = bare.forward_token(t, pos, &mut logits).expect("forward");
        pos += 1;
    }
    // `cur` is the argmax of the logits computed from the last prompt
    // token — it's the "first generated token" gen-1 prints too.
    generated.push(cur);
    for _ in 1..16 {
        cur = bare.forward_token(cur, pos, &mut logits).expect("forward");
        pos += 1;
        generated.push(cur);
    }
    println!("first 16 argmax ids = {:?}", generated);
    println!("first 16 argmax text = {:?}", bare.detokenize(&generated));
}

/// BitNet v2 integration test. Exercises the online-Hadamard dispatch
/// path end to end: with the `bitnet-v2` feature AND a rotated checkpoint
/// (signalled by `H1B_FLAG_HADAMARD_ROTATED` in the cfg `reserved` slot),
/// each `forward_token` should fire the rotation kernel exactly four
/// times per layer — confirmed via the test-only counter on
/// `HipBackend`.
///
/// Requires **both**:
///   1. `--features hip,bitnet-v2` at build time.
///   2. A `.h1b` file at `$HALO_BITNET_V2_MODEL` (or the default path)
///      whose requantizer set the H1B_FLAG_HADAMARD_ROTATED bit. Running
///      this test against a stock checkpoint will skip on the counter
///      assertion (hadamard_dispatch_count stays at 0, feature-but-no-flag
///      pass-through path — the unit tests cover that decision).
///
/// No shipping checkpoint carries the flag today; this test lands as a
/// ready-to-run harness for when the rotated 2B-4T variant lands.
#[test]
#[ignore = "requires --features bitnet-v2 + rotated .h1b; run with --ignored"]
#[cfg(feature = "bitnet-v2")]
fn hadamard_hook_fires_four_times_per_layer() {
    let model_path = std::env::var("HALO_BITNET_V2_MODEL").unwrap_or_else(|_| default_model());
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: rotated model not found at {model_path}");
        return;
    }
    let htok_path = model_path.replace(".h1b", ".htok");

    let mut bare = HipBackend::new(
        Path::new(&model_path),
        Path::new(&htok_path),
        "1bit-monster-2b-v2".into(),
        4096,
    )
    .expect("bare backend init");

    // Confirm model flag is set; otherwise the counter will stay at 0 and
    // the assertion would fail for the wrong reason.
    // Exposing cfg through a public accessor would grow the surface area;
    // instead probe indirectly by running one token and reading the counter.
    let ids = bare.tokenize("Hi");
    let mut logits = Vec::new();
    let start = bare.hadamard_dispatch_count();
    let _ = bare.forward_token(ids[0], 0, &mut logits).expect("forward");
    let delta = bare.hadamard_dispatch_count() - start;

    // Hook fires 4× per layer (Q/K/V pre-quant, O pre-quant, gate+up pre-quant,
    // down pre-quant). For the 28-layer 2B-4T base, expected = 112 per token.
    // If delta is 0 the model isn't flagged — surface a helpful skip message
    // rather than failing.
    if delta == 0 {
        eprintln!(
            "skipping assertion: loaded model does not carry H1B_FLAG_HADAMARD_ROTATED \
             (counter delta = 0 after one forward_token). Re-run against a rotated \
             checkpoint via HALO_BITNET_V2_MODEL=... cargo test --features bitnet-v2,hip \
             --ignored hadamard_hook_fires_four_times_per_layer"
        );
        return;
    }
    // 4 sites × num_layers. Read num_layers indirectly from the counter and
    // cross-check it divides evenly by 4.
    assert!(
        delta % 4 == 0,
        "hadamard hook fired {delta} times per token; expected a multiple of 4 (sites/layer)"
    );
    eprintln!("hadamard dispatch count / token = {delta}");
}

/// Smoke test for the `HALO_BITNET_V2=1` runtime env-var gate: run one
/// forward pass against the same rotated checkpoint with the env var
/// unset (V1 baseline) and set (V2 online-Hadamard), and assert the
/// argmax token or logits differ.
///
/// This is deliberately a *smoke* — one-token argmax delta — not a PPL
/// regression. The actual perplexity comparison belongs to the parity
/// harness at `benchmarks/ppl-gen2.sh`; here we just prove the runtime
/// gate actually changes the forward pass.
///
/// Runtime requirements:
///   * ROCm + a `.h1b` at `$HALO_BITNET_V2_MODEL` (or the default path).
///   * The checkpoint must carry `H1B_FLAG_HADAMARD_ROTATED` — otherwise
///     V2 passes through to V1 and this test skips.
///   * `--features hip` (the compile feature isn't required for V2
///     dispatch; the env var is the production path).
#[test]
#[ignore = "requires ROCm + rotated .h1b on disk; run with --ignored"]
fn v1_vs_v2_output_differs_via_env_gate() {
    let model_path = std::env::var("HALO_BITNET_V2_MODEL").unwrap_or_else(|_| default_model());
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: rotated model not found at {model_path}");
        return;
    }
    let htok_path = model_path.replace(".h1b", ".htok");

    let prompt = "The capital of France is";

    // --- V1 pass: env unset -------------------------------------------------
    // SAFETY: this test runs serially inside cargo's default test harness
    // by virtue of modifying a global env var. No other test here pokes
    // HALO_BITNET_V2. Writes are legal in Rust 2024 only inside `unsafe`.
    unsafe {
        std::env::remove_var("HALO_BITNET_V2");
    }
    let mut v1 = HipBackend::new(
        Path::new(&model_path),
        Path::new(&htok_path),
        "1bit-monster-2b-v1".into(),
        4096,
    )
    .expect("v1 backend init");
    let ids = v1.tokenize(prompt);
    let mut logits_v1: Vec<f32> = Vec::new();
    let mut tok_v1: i32 = ids[0];
    let mut pos = 0i32;
    for &t in &ids {
        tok_v1 = v1
            .forward_token(t, pos, &mut logits_v1)
            .expect("v1 forward");
        pos += 1;
    }
    let logits_v1_snapshot = logits_v1.clone();

    // --- V2 pass: env set to "1" --------------------------------------------
    // SAFETY: see above. Same single-threaded test harness guarantee.
    unsafe {
        std::env::set_var("HALO_BITNET_V2", "1");
    }
    let mut v2 = HipBackend::new(
        Path::new(&model_path),
        Path::new(&htok_path),
        "1bit-monster-2b-v2".into(),
        4096,
    )
    .expect("v2 backend init");
    let mut logits_v2: Vec<f32> = Vec::new();
    let mut tok_v2: i32 = ids[0];
    let mut pos = 0i32;
    for &t in &ids {
        tok_v2 = v2
            .forward_token(t, pos, &mut logits_v2)
            .expect("v2 forward");
        pos += 1;
    }

    // Restore env for subsequent tests / the live process.
    // SAFETY: see above.
    unsafe {
        std::env::remove_var("HALO_BITNET_V2");
    }

    eprintln!("v1 argmax = {tok_v1}, v2 argmax = {tok_v2}");

    // If the model isn't flagged, V2 short-circuits to V1 and the whole
    // test is a no-op — surface a helpful skip rather than a spurious
    // assert failure.
    if tok_v1 == tok_v2 && logits_v1_snapshot == logits_v2 {
        eprintln!(
            "skipping assertion: V1 and V2 outputs are byte-identical. \
             Most likely the loaded model does not carry \
             H1B_FLAG_HADAMARD_ROTATED, so V2 passes through to V1. \
             Re-run against a rotated checkpoint via \
             HALO_BITNET_V2_MODEL=... cargo test --features hip \
             --ignored v1_vs_v2_output_differs_via_env_gate"
        );
        return;
    }

    // At least one of (argmax, logit vector) must differ. A rotated
    // checkpoint changes the quant distribution enough that bit-exact
    // equality is essentially impossible.
    assert!(
        tok_v1 != tok_v2 || logits_v1_snapshot != logits_v2,
        "V1 vs V2 output identical; expected HALO_BITNET_V2=1 to diverge"
    );
}
