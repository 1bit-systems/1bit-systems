//! Parity test for the dedicated greedy fast path
//! ([`HipBackend::forward_token_greedy`]).
//!
//! What this validates: for a fixed prompt, running N tokens through
//! `forward_token_greedy` produces the *same* token id sequence as
//! running the same N tokens through `forward_token` + the standing
//! host-argmax reconcile. The two paths share the on-device
//! `forward_pass_to_device_argmax` helper, so parity comes down to:
//!
//!   1. `forward_token_greedy` returns the device argmax verbatim.
//!   2. `forward_token` (called with an empty `logits_out` and
//!      `HALO_SKIP_LOGITS_COPY` at its default) also returns the device
//!      argmax verbatim via the existing 0.1.0 skip-copy fast path.
//!
//! Any divergence here means the greedy fast path is returning a
//! different token than the skip-copy fast path — which would be a
//! routing or kernel-state-leak bug.
//!
//! Also exercises the `forward_token` + populated `logits_out` +
//! `host_argmax_lowest_index` reconcile path: on a non-tied max the
//! reconcile path must return the same token as the device-argmax
//! path. Ties are the only case the two legitimately diverge, and
//! short prompts don't exercise them — pick a prompt whose argmax is
//! stably "unique" on a BitNet-2B-4T checkpoint.
//!
//! # Run
//!
//! ```bash
//! cargo test -p onebit-router --release --features hip \
//!     --test forward_token_greedy_parity -- --ignored --nocapture
//! ```
//!
//! Needs ROCm + the halo-1bit-2b.h1b model on disk; default CI run
//! skips (the `#[ignore]` attr + hip feature gate).

#![cfg(feature = "hip")]

use std::path::Path;

use onebit_router::HipBackend;

fn default_model() -> String {
    std::env::var("HOME")
        .map(|h| format!("{h}/1bit systems/models/halo-1bit-2b.h1b"))
        .unwrap_or_else(|_| "halo-1bit-2b.h1b".into())
}

fn default_htok(model_path: &str) -> String {
    model_path.replace(".h1b", ".htok")
}

#[test]
#[ignore = "requires ROCm + halo-1bit-2b.h1b on disk; run with --ignored"]
fn forward_token_greedy_matches_forward_token() {
    let model_path = std::env::var("HALO_PARITY_MODEL").unwrap_or_else(|_| default_model());
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: model not found at {model_path}");
        return;
    }
    let htok_path = default_htok(&model_path);

    // Two separate backends so the KV caches don't cross-contaminate.
    let mut greedy_be = HipBackend::new(
        Path::new(&model_path),
        Path::new(&htok_path),
        "halo-1bit-2b-greedy".into(),
        4096,
    )
    .expect("greedy backend init");
    let mut full_be = HipBackend::new(
        Path::new(&model_path),
        Path::new(&htok_path),
        "halo-1bit-2b-full".into(),
        4096,
    )
    .expect("full backend init");

    // Same prompt to both backends. Short enough that short-circuit
    // KV-cache behaviour is stable; long enough to exercise a few
    // decode steps past the prefill boundary.
    let prompt = "The capital of France is";
    let ids = greedy_be.tokenize(prompt);
    assert!(!ids.is_empty(), "tokenizer produced empty id sequence");

    // Sanity: both backends tokenize identically.
    assert_eq!(
        ids,
        full_be.tokenize(prompt),
        "tokenizers diverged across backend instances — should never happen"
    );

    // --- Greedy path (no logits buffer, no host reconcile) ---
    let mut greedy_ids: Vec<i32> = Vec::new();
    let mut pos = 0i32;
    let mut cur = ids[0];
    for &t in &ids {
        cur = greedy_be
            .forward_token_greedy(t, pos)
            .expect("greedy forward");
        pos += 1;
    }
    greedy_ids.push(cur);
    let decode_steps: usize = 8;
    for _ in 1..decode_steps {
        cur = greedy_be
            .forward_token_greedy(cur, pos)
            .expect("greedy forward");
        pos += 1;
        greedy_ids.push(cur);
    }

    // --- Full path (populated logits_out → host reconcile active) ---
    let mut full_ids: Vec<i32> = Vec::new();
    // Pre-size so `logits_out` is NOT empty at entry, forcing the
    // non-skip branch (the one that actually copies + reconciles).
    let vocab = full_be.config().vocab_size as usize;
    let mut logits = vec![0.0f32; vocab];
    let mut pos = 0i32;
    let mut cur = ids[0];
    for &t in &ids {
        cur = full_be
            .forward_token(t, pos, &mut logits)
            .expect("full forward");
        pos += 1;
    }
    full_ids.push(cur);
    for _ in 1..decode_steps {
        cur = full_be
            .forward_token(cur, pos, &mut logits)
            .expect("full forward");
        pos += 1;
        full_ids.push(cur);
    }

    println!("greedy_ids = {greedy_ids:?}");
    println!("full_ids   = {full_ids:?}");

    // The only legitimate divergence source is an argmax tie; on the
    // default BitNet-2B-4T / "capital of France" prompt the argmax is
    // unambiguous, so both paths must return the exact same id
    // sequence.
    assert_eq!(
        greedy_ids, full_ids,
        "forward_token_greedy diverged from forward_token+host-argmax"
    );
}
