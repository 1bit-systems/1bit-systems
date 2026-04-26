# Halo-Whisper Streaming Plan

Status: scaffold landed 2026-04-20. `crates/1bit-whisper/` exists at
HEAD. Voice input is still batch-POST in `1bit-voice` today; this doc is
the design for the streaming rewrite that will let `1bit-voice` start
reacting before the user stops talking.

## 2026-04-20 UPDATE: crate scaffold landed

- Crate directory: `crates/1bit-whisper/`, package `onebit-whisper`,
  library ident `onebit_whisper`, binary `1bit-whisper` (the brand name
  wins at the filesystem + bin level; the Cargo package name takes the
  leading-digit-free form because Cargo rejects `1bit-whisper` as a
  package identifier).
- Backed by Arch/AUR `whisper.cpp` 1.8.3: `/usr/bin/whisper-server`,
  `/usr/bin/whisper-cli`, `/usr/include/whisper.h`,
  `/usr/lib/libwhisper.so.1.8.3`. No vendored whisper.cpp source tree —
  we link against the system `.so` the same way `1bit-hip` links
  `librocm_cpp.so`.
- Feature flags:
  - `stub` (default) — no C shim built, no libwhisper link. Every
    engine method returns `WhisperError::UnsupportedStub`. Keeps CI +
    laptops without libwhisper green.
  - `real-whisper` — `build.rs` compiles `cpp/shim.cpp` into
    `libonebit_whisper_shim.a` via `cc = "1.2"` and emits
    `cargo:rustc-link-lib=whisper`. Swaps the stub for the
    libwhisper-backed implementation.
- FFI surface in `cpp/shim.h` (four C-linkage functions over an opaque
  `WhisperCtx*`): `onebit_whisper_init / _free / _feed / _drain`. The
  shim owns a 500 ms scheduler that fires `whisper_full` over a 5 s
  sliding window; Rust just pumps PCM in and pulls text out.
- Safe Rust surface: `WhisperEngine::new(path) / feed(&[i16]) /
  drain_partials() -> Vec<Partial>` where
  `Partial { text, start_ms, end_ms }` derives Serialize + Deserialize.
- Tests (stub feature, default): 3 unit + 1 integration noop = 4.
  Covers UnsupportedStub error path, Partial serde roundtrip,
  WhisperError::Display informativeness, and a stable integration
  test count across feature configurations.
- Tests (real-whisper feature): 2 unit + 1 FFI-link = 3. The link
  test takes function pointers to prove libwhisper symbols resolve
  without requiring a model file on disk.
- Wired into `workspace.members` and `workspace.dependencies` with
  `default-features = false, features = ["stub"]` so downstream crates
  (1bit-voice, 1bit-echo) pick up the stub by default and flip to
  `real-whisper` explicitly when they want the native path.
- Open follow-ups tracked by this doc (unchanged from the sections
  below): VAD-tail commit, per-stream `whisper_state` isolation,
  real `t0`/`t1` timestamps via `whisper_full_get_segment_t0/_t1`,
  1bit-voice `speak_from_stream` integration.

Companion notes:
- `project_halo_kokoro.md` — TTS side (kokoro, echo_mouth)
- `project_voice_latency_sharding.md` — current 1–3 s mouth-to-ear budget
- `docs/wiki/Why-Rust.md` — Rule A/B: C++ kernel, Rust everywhere else

## 1. Does whisper.cpp have streaming today?

**Partially — there is a `stream` example, not a streaming API.**

`whisper.cpp/examples/stream/stream.cpp` runs a tight loop that:

1. reads `step_ms` (default ~500 ms) of fresh PCM16 from SDL2,
2. prepends `keep_ms` of previous audio for word-boundary continuity,
3. calls `whisper_full(ctx, params, samples, n)` on the whole
   sliding window,
4. prints the result with `whisper_full_n_segments` +
   `whisper_full_get_segment_text`,
5. optionally carries last-segment tokens forward via
   `params.prompt_tokens` / `params.prompt_n_tokens` for context.

Flags exposed: `--step`, `--length`, `--keep`, and VAD mode via
`--step 0` + `-vth`. VAD path uses an internal `vad_simple()` energy
detector and only fires `whisper_full` once silence closes a phrase,
trading latency for one-shot accuracy.

So the "streaming" mode is **still batch under the hood** — it is just
many small batches with overlap. There is no token-by-token partial
emission in the model itself; every call re-decodes the whole window
from the encoder output. The partial-transcript behaviour users see
comes from the fact that each window is small (3–5 s).

Key public API functions this proves are stable in `whisper.h`:

| Function | Role in our streaming plan |
|---|---|
| `whisper_init_from_file_with_params` | one-shot ctx load |
| `whisper_init_state` / `whisper_free_state` | per-stream isolated state so one ctx can serve multiple concurrent `WhisperStream`s |
| `whisper_full_default_params` | build `whisper_full_params` with a sampling strategy |
| `whisper_full_with_state` | run inference on the current window using the per-stream state (what we actually call each tick) |
| `whisper_pcm_to_mel_with_state` | (optional) split mel compute from decode if we want to overlap IO and decode |
| `whisper_full_n_segments_with_state` | count segments in the last call |
| `whisper_full_get_segment_text_with_state` | pull UTF-8 text of segment `i` |
| `whisper_full_get_segment_t0 / _t1` | ms timestamps, for dedup on overlap |
| `whisper_full_n_tokens` / `whisper_full_get_token_id` | feed back into `params.prompt_tokens` as carry-over context |

The `_with_state` variants are the ones we want so that the same
model can drive many sessions; the no-state variants in the stream
example are a shortcut that only works for a single global session.

## 2. Trade-offs: fixed-window vs VAD chunking

### Fixed sliding window (default in stream example)

- **Latency**: bounded by `step_ms`. 500 ms is the typical floor; pushing
  below that starves the decoder because a single `whisper_full` call
  still costs tens of ms on a tiny/base model.
- **Cost per emission**: each tick re-encodes `length_ms` (commonly
  3000–5000) and re-decodes the whole window. That is **wasted work**
  — most of the window hasn't changed.
- **Hallucination risk**: very short chunks with no linguistic context
  make whisper hallucinate. `keep_ms` (typically 200 ms) and
  `params.prompt_tokens` from the last segment mitigate this but do
  not eliminate it.
- **Good for**: Echo's "start reacting while user is mid-sentence"
  use case — we want the partial even if it's noisy, 1bit-voice can
  gate on confidence.

### VAD-triggered (one phrase at a time)

- **Latency**: only emits after trailing silence ≥ threshold (e.g.
  250–500 ms). Adds a silence-hangover delay on top of model time.
- **Cost per emission**: single `whisper_full` per utterance, so much
  lower total CPU.
- **Accuracy**: highest — the model sees the whole phrase end-to-end.
- **Good for**: 1bit-mcp voice tool invocations, dictation mode.
- **Bad for**: barge-in / duplex conversation — we lose the pre-end
  partial we need to start warming the LLM.

### Re-hypothesis cost on overlap

Every sliding call redoes:

- mel spectrogram of `length_ms` (cheap, ~1–2% of the call),
- encoder pass (expensive — dominant cost, scales ~linearly with
  `length_ms`),
- beam/greedy decode (bounded by `max_tokens`).

The encoder does **not** have a way to "extend" mel; you must hand it
the full window again. So the re-hypothesis cost is real and
non-negotiable until upstream lands a streaming encoder. The Jun 2025
arXiv paper *"Adapting Whisper for Streaming Speech Recognition via
Two-Pass Decoding"* (2506.12154) is a research lead on a real
streaming head; not in upstream whisper.cpp as of 2026-04-20.

### Our pick

**Fixed window + VAD-on-tail hybrid.** Run the sliding window for
partials while speech is active, then on trailing silence do one
**committed** `whisper_full` over the whole utterance to replace the
streamed partials with the final hypothesis. 1bit-voice treats
partials as "draft" and the committed result as "final". This mirrors
what Google / Deepgram APIs expose (`is_final: bool`) and keeps
1bit-voice's state machine simple.

## 3. Rust-side FFI shape

`crates/1bit-whisper` (pending — **does not exist on disk today**,
verified 2026-04-20) will own the FFI boundary. `1bit-hip`
provides the pattern: thin `-sys` module + safe wrapper, no kernel
reimplementation. whisper.cpp is CPU-only in our use (the mel +
encoder + decoder are tiny relative to BitNet); no HIP port needed.

Proposed core trait:

```rust
pub struct Partial {
    pub text: String,      // UTF-8
    pub t0_ms: i64,        // segment start in stream time
    pub t1_ms: i64,        // segment end
    pub is_final: bool,    // true = committed after VAD tail
    pub seq: u64,          // monotonic per stream, for dedup
}

pub trait WhisperStream {
    /// Push PCM16 mono @ 16 kHz into the ring. Non-blocking. Never
    /// calls into whisper.cpp directly — scheduling decides when
    /// to fire `whisper_full_with_state`.
    fn feed_pcm16(&mut self, samples: &[i16]);

    /// Pull any Partials that have been emitted since the last call.
    /// Empty Vec is normal and cheap. Always returns quickly; the
    /// actual inference happens on a background worker so this trait
    /// is safe to poll at 60 Hz from the audio thread.
    fn drain_partials(&mut self) -> Vec<Partial>;
}
```

### Method → C function mapping

| Rust trait surface | whisper.cpp function(s) called |
|---|---|
| `WhisperStream::new(ctx, cfg)` | `whisper_init_state(ctx)` once per stream; `whisper_full_default_params(sampling)` cached |
| `feed_pcm16(&mut self, &[i16])` | **no FFI call**. Converts i16→f32 and pushes into a ring buffer owned by the Rust side. The scheduler on the worker thread decides when to fire a window. |
| worker: window tick (sliding) | `whisper_full_with_state(ctx, state, params, pcm_f32, n)` — `params.no_context=false`, `params.single_segment=true` (for low latency), `params.prompt_tokens` carries last segment's tail tokens |
| worker: harvest partials | `whisper_full_n_segments_with_state(state)` → loop `whisper_full_get_segment_text_with_state(state, i)` + `..._get_segment_t0/_t1(state, i)`. Build `Partial { is_final: false, ... }` |
| worker: VAD tail → commit | one more `whisper_full_with_state` over the whole utterance with `params.no_context=true, params.single_segment=false`; emit `Partial { is_final: true, ... }` |
| `drain_partials()` | pop from crossbeam channel fed by worker — no FFI |
| `Drop for WhisperStreamImpl` | `whisper_free_state(state)` (ctx stays alive, shared) |

Notes:

- VAD: whisper.cpp exposes its internal `vad_simple` only inside the
  stream example, not in `whisper.h`. We reimplement it in Rust
  (energy + zero-crossing over a 30 ms frame) — it is ~40 lines and
  avoids pulling in another dep. Silero-VAD is overkill here.
- Thread model: one **blocking** worker thread per stream holding
  the `whisper_state *`. The audio producer is wait-free via a
  SPSC ringbuffer (e.g. `ringbuf` crate). `drain_partials` is also
  wait-free via an MPSC channel. No `async` at the FFI boundary —
  whisper.cpp is synchronous and long-running.
- The global `whisper_context *` is `Arc`-shared and treated as
  read-only after init (mel filters, model weights). All mutable
  per-call state is in the `whisper_state *`. This is the
  upstream-blessed multi-session pattern.
- Memory: each `whisper_state` allocates KV + logits buffers for the
  decoder. Budget ~100 MB per concurrent stream on `base.en`. For
  Echo single-user this is fine; multi-user tenancy is a later
  problem.

### Out of scope for v1

- GPU backend for whisper encoder (Vulkan / Metal exist upstream;
  we stay CPU; CPU cost is dominated by BitNet anyway).
- Word-level timestamps (`whisper_full_get_token_t0`) — dictation-only
  feature; 1bit-voice doesn't need them.
- Language auto-detect streaming (`whisper_lang_auto_detect`) — we
  pin `en` at init. Multilingual is a later knob.

## 4. 1bit-voice integration

Today (`cpp/voice/src/pipeline.rs`): the pipeline is **output
only**. It takes a prompt string, fans out SSE from 1bit-server, runs
`SentenceSplitter`, and POSTs sentences at halo-kokoro. There is no
input side in this crate yet. `1bit-echo` is an empty `lib.rs`
scaffold.

Migration shape:

1. **New input type on `VoicePipeline`.** Add a `speak_from_stream`
   entrypoint that takes `impl Stream<Item = Partial>` instead of a
   `String` prompt. Keep `speak(prompt)` for CLI / batch callers.
2. **Partial → prompt policy.** Two options:
   - **Eager** (recommended v1): every time a new `Partial` arrives
     with `is_final=false` and changes the prefix, **cancel** the
     in-flight LLM SSE request and start a new one with the fresh
     partial. 1bit-server must learn `POST /v1/chat/completions` with
     `X-Halo-Cancel-Prev: <session>` or we accept wasted tokens. Ship
     the wasted-tokens version first; `1bit-server` cancel path is
     follow-up.
   - **Gated**: only fire the LLM once `is_final=true` arrives. This is
     just a 200–500 ms speedup over today (mic-stop → final partial)
     and misses the point of streaming STT. Rejected for v1.
3. **halo-kokoro side unchanged.** Sentence-splitter streaming TTS
   already works the moment the LLM starts emitting tokens.
4. **End of turn detection.** Barge-in: if a new `Partial` arrives
   while halo-kokoro is still speaking, 1bit-voice should **stop the
   current TTS playback** (caller-side; we already don't own
   playback) and emit a signal. Put a `VoiceEvent::BargeIn` variant
   on the output stream.
5. **Wire order.**
   - `1bit-whisper` crate + 3 tests (ctx load, one-shot tiny WAV,
     partial emission from synthetic PCM).
   - `1bit-voice::input` module with `MicSource` (cpal) →
     `WhisperStream` → `mpsc<Partial>`.
   - `1bit-voice::pipeline::speak_from_stream`.
   - 1bit-server cancel endpoint.
   - 1bit-echo WS frames for browser peer (Opus in → Partial out, via
     the same trait).

Estimated effort: ~3–4 sit-down days end-to-end, of which 1 is the
FFI crate, 1 is the cancel path on 1bit-server, and 1–2 is polishing
barge-in + dedup on overlap.

## 5. Rank vs MedusaBitNet

**MedusaBitNet ships first.** Reasoning:

- Medusa buys us tokens/s on the generation side. That compounds with
  every latency optimization we ever make — faster LLM = faster voice
  = faster mouth-to-ear, period.
- Streaming STT only helps the **input** side of Echo. Current mic-up
  → first-LLM-token latency budget (from
  `project_voice_latency_sharding.md`) is ~500–700 ms, roughly
  evenly split between whisper-batch and LLM TTFT. Streaming STT
  cuts a few hundred ms off the STT half; Medusa multiplies the LLM
  half by 1.5–2×.
- Medusa needs a BitNet-aware head, which is net-new kernel work on
  gfx1151 — it is also the kind of thing only we are positioned to
  do. Streaming whisper is a commoditized capability: the stream
  example already runs; we are just wrapping it.
- Streaming whisper can ship behind Medusa without blocking anyone —
  1bit-voice works today with batch STT. Medusa is a no-op until it
  lands.

Order: **Medusa → streaming-whisper → 1bit-echo WS → barge-in.**
Revisit if Medusa's timeline slips past one calendar month; at that
point streaming-whisper becomes a cheap morale-win ship.

## References

- Upstream stream example:
  https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/stream.cpp
- Stream README:
  https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/README.md
- Public API header:
  https://github.com/ggml-org/whisper.cpp/blob/master/include/whisper.h
- DeepWiki streaming overview:
  https://deepwiki.com/ggml-org/whisper.cpp/3.3-talk-llama
- Research lead, true streaming encoder:
  "Adapting Whisper for Streaming Speech Recognition via Two-Pass
  Decoding", arXiv:2506.12154 (Jun 2025)
