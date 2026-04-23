//! stream_smoke.rs — black-box integration tests for `WhisperStream`.
//!
//! All three tests below run under the default `stub` feature and never
//! touch libwhisper. The real-feature path has its own link-level test
//! in `tests/real_link.rs`; this file is only about the Rust-side
//! threading + channel plumbing the streaming module owns.

#![cfg(all(feature = "stub", not(feature = "real-whisper")))]

use onebit_whisper::{WhisperEngine, WhisperError, WhisperStream};
use std::thread;
use std::time::Duration;

/// 1. Under the stub feature, `WhisperStream::new` surfaces the same
///    `UnsupportedStub` error as the engine it wraps — no worker is
///    spawned, no background thread leaks.
#[test]
fn constructor_under_stub_returns_unsupported() {
    match WhisperStream::new("/nonexistent/model.bin") {
        Ok(_) => panic!("stub stream must not construct successfully"),
        Err(WhisperError::UnsupportedStub) => { /* expected */ }
        Err(other) => panic!("expected UnsupportedStub, got {other:?}"),
    }
}

/// 2. Feed + try_recv lifecycle: using the test-only stub-engine
///    constructor we can drive a live `WhisperStream` through
///    `feed`/`try_recv` and confirm the channel plumbing works end-to-
///    end even though the inner engine is a no-op. `try_recv` must
///    always return `None` because a stub engine never emits partials.
#[test]
fn feed_and_try_recv_lifecycle_under_stub() {
    let engine = WhisperEngine::new_stub_for_tests();
    let mut stream = WhisperStream::from_engine(engine);

    // Feed ~1 s of silence at 16 kHz in 125 ms chunks (2000 samples).
    // That's 8 chunks — more than the 500 ms step boundary, so the
    // worker should have fired at least one tick.
    let chunk = vec![0i16; 2_000];
    for _ in 0..8 {
        assert!(
            stream.feed(&chunk),
            "backpressure should not trigger on idle worker"
        );
    }

    // Let the worker breathe so it definitely gets past its step
    // threshold. 200 ms is well above the 100 ms WORKER_RECV_TIMEOUT.
    thread::sleep(Duration::from_millis(200));

    // Stub engine never produces partials. `try_recv` must return None
    // and `drain` must return an empty Vec.
    assert!(stream.try_recv().is_none());
    assert!(stream.drain().is_empty());
}

/// 3. Thread-drop cleanup: drop a `WhisperStream` while the worker is
///    in the middle of a pressure loop. The `Drop` impl must return
///    promptly (no deadlock, no hung join) and the process must stay
///    responsive. We can't directly assert "no thread leak" from a
///    unit test, but we *can* assert that many create/drop cycles
///    under load complete without hanging — a leak would manifest as
///    either a hang (bad shutdown signalling) or FD / memory
///    exhaustion after enough iterations.
#[test]
fn drop_during_pressure_is_clean() {
    for _ in 0..32 {
        let engine = WhisperEngine::new_stub_for_tests();
        let mut stream = WhisperStream::from_engine(engine);

        // Pressure: shove 50 chunks (~6 s of audio) in without waiting.
        // Some will be accepted, some may drop on backpressure — both
        // are fine. The point is the worker is actively draining the
        // PCM channel when we hit Drop.
        let chunk = vec![1i16; 1_600]; // 100 ms
        for _ in 0..50 {
            let _ = stream.feed(&chunk);
        }

        // Deliberately do NOT sleep, join, or otherwise coordinate. The
        // test asserts nothing about the worker's progress — only that
        // the following `drop(stream)` returns in bounded time.
        drop(stream);
    }
}
