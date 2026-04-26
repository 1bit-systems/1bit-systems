# Architecture deep — the stack end to end

The stack behaves like a small station — discrete modules, one bus, hard seals at each airlock. This page walks every module, every data path, every external dependency, and every failure surface. Cross-links to [Why-Ternary](./Why-Ternary.md), [Why-Strix-Halo](./Why-Strix-Halo.md), [Why-This-Way-How](./Why-This-Way-How.md), [Installation](./Installation.md), [Clients](./Clients.md), [Troubleshooting](./Troubleshooting.md), [Observability](./Observability.md), and [Network-Topology](./Network-Topology.md) are inline where relevant. Numbers without a citation are the current production measurement on the strixhalo box at the date of writing (2026-04-21).

## Constellation

One host (strixhalo, gfx1151) serves every production HTTP surface. A small Headscale mesh binds one audio peer (sliger, Arc B580 Vulkan for STT/TTS), one kernel peer (ryzen, gfx1201 / RX 9070 XT), and one archive (pi). External callers speak OpenAI-compatible HTTP to a Caddy bearer-checked front door, which reverse-proxies `/v1/*` to the Rust `1bit-server` on `:8180`. The server dispatches through `lemond` to a HIP backend that crosses the FFI airlock into `rocm-cpp` for ternary GEMV, split-KV Flash-Decoding attention, RoPE, RMSNorm, SiLU, and the KV cache. Voice (halo-whisper, halo-kokoro — hosted on the sliger B580), image (sd.cpp, local), tooling (halo-mcp), and agents (1bit-agents) hang off the same control plane. Every long-lived process is a systemd unit. (Post-cutover state as of v0.1.0, 2026-04-24; the retired gen-1 C++ `bitnet_decode` on `:8080` is disabled.)

```
                              +-------------------+
                              |   HTTP clients    |
                              |   Open WebUI · MCP|
                              |   curl · Helm     |
                              +---------+---------+
                                        |
                              +---------v---------+
                              |   Caddy :443 TLS  |  bearer check
                              +---------+---------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
                   | /v1/*                                   | sidecars
                   v                                         v
         +------------------+                    +------------------+
         | 1bit-halo-server |                    | 1bit-lemonade    |
         | :8180 (Rust)     |                    | :8200 gateway    |
         +--------+---------+                    +------------------+
                  |
                  v
         +------------------+
         |   lemond    |
         |   backend = Hip  |
         +--------+---------+
                  | extern "C"
                  v
         +------------------------------------------+
         |            rocm-cpp (C++20 / HIP)        |
         |  ternary_gemv · attention_fd · rope      |
         |  rmsnorm · silu · kv_cache · h1b_loader  |
         +------------------+-----------------------+
                            | hipcc → gfx1151 ISA
                            v
                 +------------------+
                 | Radeon 8060S     |
                 | 40 CU · wave32   |
                 | WMMA · 256 GB/s  |
                 +------------------+

  Retired 2026-04-24 (v0.1.0 cutover): gen-1 bitnet_decode (C++) on :8080
    unit 1bit-halo-bitnet.service now disabled; :8080 does not bind.

  Sidecars (same host, same systemd):
    sd.cpp :8081            1bit-landing :8190     halo-mcp (stdio)
    1bit-agents (tokio bus) 1bit-watch-discord     1bit-watch-github
    halo-memory-sync

  Sliger audio node (separate host on the mesh, 100.64.0.2):
    halo-whisper :8190 (Vulkan, Arc B580)
    halo-kokoro  :8191 (Vulkan, Arc B580; CPU fallback)

  Mesh (Headscale 100.64.0.0/10):
    strixhalo 100.64.0.1 (this box, coordinator, LLM + image)
    sliger    100.64.0.2 (Arc B580 Vulkan, STT/TTS; 1080 Ti retired 2026-04-22)
    ryzen     100.64.0.3 (gfx1201, RX 9070 XT, second kernel target)
    pi        100.64.0.4 (ZFS 3.6 TB, canonical archive)
```

See the three-pillar table in [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) §"The three pillars" for the language / repo split. The short version: pillar 1 is HIP kernels (`rocm-cpp`), pillar 2 is the Rust caller (this workspace), pillar 3 is agents + Rust services on the same box. Apple-Silicon MLX is a feature-gated dev-only side path (not shipped); upstream Python Lemonade is a caller-side reference (not runtime). Our own `1bit-lemonade` crate (Rust) is a separate implementation, not a Lemonade port.

## Ports + surfaces

| port | binding | service | surface | notes |
|---:|---|---|---|---|
| 443 | public | caddy | TLS | bearer check on `/v1`, `/v2` |
| 8180 | 127.0.0.1 | 1bit-halo-server (Rust) | `/v1/*` | production LLM; feature `real-backend` |
| ~~8080~~ | — | ~~bitnet_decode (C++)~~ | ~~`/v1/*`~~ | retired at v0.1.0 cutover; unit disabled |
| 8190 | sliger | halo-whisper | STT | whisper.cpp Vulkan on Arc B580 (separate host) |
| 8191 | sliger | halo-kokoro | TTS | kokoro.cpp Vulkan on Arc B580 (separate host) |
| 8081 | 127.0.0.1 | sd.cpp | SDXL | image-gen sidecar |
| 8190 | 127.0.0.1 | 1bit-landing | HTML | `/studio/*` landing + wiki proxy |
| 8200 | 127.0.0.1 | 1bit-lemonade | `/v1/models` | OpenAI-compat model gateway |
| 8380 | 127.0.0.1 | headscale | coord | Caddy-fronted on `:443` for peers |
| 9090 | 127.0.0.1 | headscale | metrics | Prometheus scrape |
| stdio | — | halo-mcp | JSON-RPC | socket-activated, Claude Code / Claude Desktop |

Nothing but Caddy binds a public interface; everything else is loopback or the Headscale tailnet. See [Why-Caddy-Systemd](./Why-Caddy-Systemd.md).

## Request life-cycle

A chat-completion walked through every layer. All numbers are measured unless flagged `measured-TBD`.

### 1. Client → TLS

Client issues `POST https://halo.<host>/v2/chat/completions` with `Authorization: Bearer sk-halo-...`. TLS handshake terminates at Caddy. Bearer match is a constant-time string compare against the value in `/etc/caddy/Caddyfile` (root-only). Overhead: **<1 ms** for the bearer check, dominated by TCP+TLS setup on first request. See [Why-Caddy-Systemd](./Why-Caddy-Systemd.md).

### 2. Caddy → axum (:8180)

Caddy's `reverse_proxy localhost:8180` is a plain HTTP/1.1 hop over loopback. Caddy writes request headers and body unchanged; axum reads them with hyper. Per-hop overhead: `measured-TBD` (well under 1 ms loopback).

### 3. axum router + deserialize

`build_router_with_state(AppState { backend, metrics, sd_base_url, http_client })` dispatches on the path. The handler is `chat_completions`:

```rust
async fn chat_completions(
    State(s): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ServerError> { ... }
```

`Json<ChatCompletionRequest>` deserializes the body with `serde_json`. Shape errors return 400 before any inference work starts. The `metrics.histogram` timer starts after the 400 check — bad requests never skew latency histograms.

### 4. Router dispatch

`lemond`'s `Router::run(req: RouterRequest) -> RouterResponse`. `Backend::Hip` is the default and the only one compiled on production. `Backend::Cpu` scaffolds the AVX2 lane from `1bit-cpu` but returns `BackendError::CpuLaneStub` today. See [CPU-Lane-Plan](./CPU-Lane-Plan.md) for the wire-up path.

For long prompts, `prefill_routing_decision` consults `prefill_crossover_len` (default 33 — see `DEFAULT_PREFILL_CROSSOVER_L` in `crates/lemond/src/lib.rs`) to decide whether the prefill phase goes to CPU AVX2 or stays on the iGPU. Decode always stays on iGPU — bandwidth story wins, see [Why-Strix-Halo](./Why-Strix-Halo.md).

### 5. Session + KV-cache setup

The router locks a `tokio::sync::Mutex` around the shared KV cache. **`pos` is reset to 0 per request** — the 2026-04-19 SEGV was a `pos` accumulator bug that only appeared after ~200 sustained completions (see [Troubleshooting](./Troubleshooting.md)).

KV cache bytes for halo v2 (`hidden_size = 2048`, `num_kv_heads = 8`, `head_dim = 256`, FP16):

```
per-token KV = 2 × num_kv_heads × head_dim × sizeof(fp16)
             = 2 × 8 × 256 × 2 = 8192 bytes ≈ 8 KiB per layer
             × num_layers (30) = 240 KiB per token
```

At context length 4096: 4096 × 240 KiB ≈ **960 MiB of KV** — comfortably sub-GB, fits the 128 GB LPDDR5 budget thousands of times over.

### 6. Tokenize

`1bit-core::htok` decodes `<|begin_of_text|>`, `<|eot_id|>`, and the 128k Llama-3 BPE vocab. Special-token handling (Llama-3 EOT at ID 128009) is the fix that took burn-in parity from 18% to **96.67%** (see [Why-Shadow-Burnin](./Why-Shadow-Burnin.md)).

### 7. Prefill (FFI airlock)

For each prompt token, one forward pass. Rust calls into `1bit-hip`, which crosses to C via `extern "C"`:

```rust
pub fn ternary_gemv_halo_f16(
    packed_weights: DevicePtr<u8>,
    activations_i8: DevicePtr<i8>,
    activation_scale: f32,
    row_scales: DevicePtr<f32>,
    output_f16: DeviceMutPtr<u16>,
    m: i32,
    k: i32,
    stream: HipStream,
) -> RcppStatus
```

Per-call FFI overhead (`measured-TBD` — bindgen-less hand-rolled bindings, no Rust allocations per call, HIP stream enqueue is a few microseconds). `DevicePtr<T>` is a newtype over `*const T` / `*mut T` — no Rust references are ever constructed so aliasing rules don't apply at the boundary.

Each forward pass enqueues: `rmsnorm_fp16` → `ternary_gemv_halo_f16` for Q/K/V → `rope_fp16` → `kv_cache_attn_decode_fd` → `ternary_gemv_halo_f16` for O → `residual_add_fp16` → `rmsnorm_fp16` → `ternary_gemv_halo_f16` for gate/up → `silu_glu_fp16` → `ternary_gemv_halo_f16` for down → `residual_add_fp16`. Per layer. For 30 layers per token.

### 8. Decode loop

Sample argmax on host (`argmax_fp32`), append to `generated_ids`, write next K/V slot, check stop tokens, repeat until `<|eot_id|>` or `max_tokens`. Stop check runs on token IDs *before* detokenization so special tokens never leak into the output bytes.

Temperature > 0 engages the sampler from `1bit-core::sampler` (top-p, top-k, frequency/presence penalty). See `crates/1bit-core/src/sampler.rs::Sampler`.

### 9. Streaming return

If `req.stream` is true, the handler wraps the stream in an accounting iterator that tallies tokens per SSE frame and fires `metrics.record_request()` at end-of-stream. Frames are SSE (`data: {...}\n\n`), ending with `data: [DONE]\n\n`. If non-streaming, the full response is assembled server-side and returned as a single JSON body.

### 10. Close

Caddy forwards the response bytes unchanged. Bearer-authed connections reuse the TLS session on keep-alive. Per-request wall-clock on a 10-token prompt + 10-token reply: ~200 ms on the strixhalo box.

## Kernels

Source: `rocm-cpp/src/kernels/` + `rocm-cpp/include/rocm_cpp/`. Every kernel is HIP C++ targeting gfx1151 (wave32, WMMA). The C ABI (`extern "C"`) is the airlock — everything Rust sees is a thin wrapper in `crates/1bit-hip/src/lib.rs`.

### ternary_gemv_halo_f16 — the hot path

Signature in Rust: see `ternary_gemv_halo_f16` above. Under the hood:

- **Packed layout**: 2 bits per weight, `uint8[M, (K+3)/4]`. Four ternary weights per byte, high-to-low: `w0 w1 w2 w3`. Codes `00 → -1`, `01 → 0`, `10 → +1`, `11 → reserved`.
- **Per-row FP32 scale**: `row_scales[M]`. Output reconstruction is `y[m] = scale[m] · Σ_k sign(w[m,k]) · act[k] · act_scale`.
- **WMMA use**: the kernel uses `v_dot4_i32_i8` (4-wide int8 dot product) for the inner sum and reduces in wave32 lanes. WMMA accumulator is int32; we convert to FP32 once per tile and apply the row scale.
- **Why 92% of LPDDR5 peak**: measured on decode @ N=64. Weight bandwidth dominates 600:1 over activation bandwidth (see [activation sparsity memory](../../.claude/projects/-home-bcloud/memory/project_activation_sparsity_phase1.md)). Kernel is memory-bandwidth-bound. Compute utilization at the decode shape (M=2048, K=2048, N=1) is <10% of peak FLOPs.
- **What would get it higher**: (1) Sherry 1.25-bit packing — 5 bits per 4-weight group instead of 8, a ~1.6× bandwidth reduction, currently blocked on retrain quality (see [Sherry-Default-Decision](./Sherry-Default-Decision.md)); (2) TQ1 base-3 packing (20 ternaries in 32 bits = 1.6 bpw) via `ternary_gemv_tq1_halo_f16`; (3) K-outer tile retune to improve L1 residency (see the gerganov L1 tuning lead in memory).

### attention_fd — split-KV Flash-Decoding

Signature: `kv_cache_attn_decode_fd(q, k_cache, v_cache, out, pos, ...) -> RcppStatus`.

- **Topology**: per-head parallelism across thread-blocks. Each attention head splits its KV range (of length `pos + 1`) into `B` chunks (default B=4 at L ≤ 2048, scaled up at longer L). Each chunk reduces to a local `(m, l, o)` triple — max-logit, log-sum, partial output.
- **Log-sum-exp combine**: a second kernel pass merges the B triples per head into a final `(m*, l*, o*)` and writes one output vector per head. Numerically stable: subtract local max before exp.
- **Provenance**: replaced the single-block-per-head attention kernel 2026-04-19. **6.78× speedup at L=2048**, bit-exact against the reference path. Default in `bitnet_decode` and `1bit-halo-server`. See `project_attention_fd` in memory + `docs/wiki/Why-This-Way-How.md` §"end-to-end path".
- **Memory**: KV cache is FP16, laid out as `[num_layers][2 = K|V][num_kv_heads][max_seq_len][head_dim]`. Append-only ring with `pos` as the write cursor. Per-session allocation; never shared across sessions.

### rope_fp16 — HF split-half convention

Signature: `rope_fp16(q, k, pos, num_heads, head_dim, theta, stream) -> RcppStatus`.

The 2026-04-19 fix flipped the interleaved convention (`(q0, q1, q2, q3) → (q0·cos − q1·sin, q0·sin + q1·cos, ...)`) to the HF split-half convention (`(q[0..D/2], q[D/2..D]) → (q_low·cos − q_high·sin, q_low·sin + q_high·cos)`). Numbers from the memory entry:

| test | before | after |
|---|---:|---:|
| wikitext-103 PPL | 524 | ~12 |
| repeated-text PPL | 4.29 | 1.04 |

The interleaved path was wrong for every checkpoint in the wild — they all ship in HF split-half convention. See `project_rope_bug_fix` in memory.

### rmsnorm + silu — fused into FFN

- `rmsnorm_fp16(input_f16, weight_f16, eps, out_f16)` — variance across `hidden_size`, scale by `weight / sqrt(var + eps)`.
- `silu_glu_fp16(gate_f16, up_f16, out_f16)` — SwiGLU gate: `y = silu(gate) * up = (gate / (1 + exp(-gate))) * up`. Fused into the FFN path so we don't materialize a separate gate buffer between the gate-GEMV and the down-GEMV.
- `relu2_glu_fp16`, `relu2_glu_rmsnorm_fp16` — ReLU² variant for the activation-sparsity experiments. Phase 1 measured 79.91% activation sparsity vs Nurvitadhi's 50-60% prior (see `project_activation_sparsity_phase1` in memory). Bandwidth-dominated workload makes the upper-bound speedup 10-15%; deferred behind Sherry.

### kv_cache — append ring, pinned

- **Layout**: `[num_layers][2][num_kv_heads][max_seq_len][head_dim]` FP16. Contiguous per-layer.
- **Append semantics**: each decode step writes one K and one V slot per layer per head at `pos`, then increments. No eviction within a session.
- **Per session**: one KV-cache buffer per in-flight request. Today's production path serializes at the router mutex so there is exactly one live KV-cache at a time. Concurrency work is queued behind a KV-cache-compression experiment (rotorquant).
- **Alignment**: 128-byte aligned via `hipMalloc`. The strixhalo `amdgpu` driver emits the OPTC hang if the allocator is pressured concurrently — see the mitigation unit `halo-gpu-perf.service` + [Troubleshooting](./Troubleshooting.md).

## Memory model

Strix Halo's LPDDR5 is **unified** — the same 128 GB the CPU sees is the memory the iGPU reads. No PCIe copy, no discrete-GPU VRAM. This is the single biggest reason the stack picked this platform. See [Why-Strix-Halo](./Why-Strix-Halo.md).

Budget for halo v2 (2B, ternary, FP16 activations, 4096 context):

| region | size | lifetime | notes |
|---|---:|---|---|
| weights (`.h1b` mmap) | ~1.1 GiB | process | shared across sessions, page-cached |
| KV cache | up to ~960 MiB @ 4096 ctx | per session | pinned, FP16 |
| activations (scratch) | ~100 MiB | per forward | reused across layers |
| `.htok` tokenizer | ~3 MiB | process | mmap |
| HIP driver + ROCm | ~1 GiB | process | `/opt/rocm/lib` |
| OS + fish + ~ everything else | ~4 GiB | system | headroom |
| **subtotal** | **~7 GiB** | | out of **128 GiB available** |
| headroom | ~121 GiB | | available for sd.cpp, whisper, kokoro, agents, concurrent sessions |

Compared to a discrete GPU: weights would have to be DMA-copied into VRAM at startup and any CPU-side post-processing would pay PCIe round-trip latency. On unified memory the GEMV reads straight from the same DDR bank the CPU just wrote to; the FFI airlock passes a `DevicePtr<u8>` which on `hipMalloc`'d memory is a virtual address pointing into LPDDR5, not a copy target.

## Model formats

### `.h1b` — halo-1bit

Source: `crates/1bit-core/src/h1b.rs`. Version 2 layout:

```
offset  size  field
0x00    4     magic     = "H1B\0"
0x04    4     version   = 2 (int32 LE)
0x08    36    cfg       = 9 × int32
                hidden_size, intermediate_size, num_layers, num_heads,
                num_kv_heads, vocab_size, max_seq_len, tie_embeddings, reserved
0x2C    8     extras    = 2 × float32   (rope_theta, rms_norm_eps)  [v2+]
0x34    ....  weights   = per-layer packed ternary blobs + per-row f32 scales
```

The `reserved` int is a **flag word** — four bits are assigned today:

- `0x1` — `H1B_FLAG_HADAMARD_ROTATED` — weights pre-absorbed `W' = W @ H^T` for BitNet v2 (see [BitNet-v2-Hadamard-Plan](./BitNet-v2-Hadamard-Plan.md))
- `0x2` — `H1B_FLAG_SHERRY_FP16` — clean-room FP16 Sherry 1.25-bit dispatch
- `0x4` — `H1B_FLAG_BONSAI_Q1` — PrismML `Q1_0_g128` 1-bit format
- `0x8` — `H1B_FLAG_BONSAI_TQ2` — PrismML `TQ2_0_g128` ~1.585-bit format

Flags compose where mutually compatible; the Bonsai pair is mutually exclusive.

Per-layer tensor order (exactly what `rocm-cpp/src/h1b_loader.cpp` expects):

```
input_norm[hs], post_attn_norm[hs], attn_sub_norm[hs], (3 dup slots skipped),
(2 legacy FFN slots skipped), ffn_sub_norm[is],
Q, K, V, O, gate, up, down   — each is packed_bytes + [rows] f32 scales
```

See [Why-H1b-Format](./Why-H1b-Format.md) for the vs-GGUF rationale. Short version: GGUF carries per-tensor metadata we don't need, and the loader walked ~5× slower against a cold page cache; custom format is O(num_layers) offset math and effectively free.

### `.htok` — tokenizer

Source: `crates/1bit-core/src/htok.rs`. BPE with Llama-3's 128 256-entry vocab + special tokens. `HtokFile { vocab, merges, special, ... }`. All ints little-endian. Mmap'd, parsed once at startup.

### GGUF → `.h1b` conversion

Tool: `tools/gguf-to-h1b` (Rust, standalone binary). Reads a `microsoft/bitnet-b1.58-2B-4T` GGUF, re-packs ternary tensors into halo v2 layout, writes a `.h1b` + `.htok`. One-shot, never in a serving path (Rule A — see [Why-No-Python](./Why-No-Python.md)).

## Agents — 17 specialists

Source: `crates/1bit-agents/src/lib.rs`. The `Name` enum has 17 variants; the registry is the single source of truth for both the agents bus and the MCP tool list.

| specialist | role | input shape | output shape | dispatched by | returns to |
|---|---|---|---|---|---|
| Anvil | kernel rebuild + bench on `rocm-cpp` changes | `AnvilRequest { commit, run_bench }` | `AnvilResponse { built, bench_tok_s }` | anvil.timer | echo → `#changelog` |
| Carpenter | file scaffolding, new-crate boilerplate | `{ name, template }` | `{ paths_written }` | Planner | Planner |
| Cartograph | cross-repo changelog + topology snapshot | `{ repos[] }` | `{ summary, mapdot }` | Librarian | Librarian |
| EchoEar | STT ingress — halo-whisper transcript pipeline | `{ audio_path }` | `{ transcript, confidence }` | halo-voice | EchoMouth |
| EchoMouth | TTS egress — halo-kokoro streaming synth | `{ text, voice }` | `{ audio_path, samples }` | EchoEar / Herald | channel |
| Forge | PR drafting + commit-message composition | `{ diff, intent }` | `{ title, body, commits[] }` | Planner | Magistrate |
| Gateway | inbound classification + routing policy | `{ raw_msg, source }` | `{ route: Name, context }` | watchers | registry |
| Herald | comms / Q&A / chat reply | `{ query, ctx }` | `{ reply_md }` | Gateway | echo → channel |
| Librarian | CHANGELOG + wiki upkeep | `{ since_commit }` | `{ changelog_lines }` | librarian.timer | repo |
| Magistrate | PR review + Conventional-Commits lint + secret scan | `{ pr_url }` | `{ verdict, findings[] }` | gh-trio.timer | echo |
| Muse | long-form prose draft (README, posts) | `{ topic, tone }` | `{ draft_md }` | operator | operator |
| Planner | multi-step task decomposition | `{ objective }` | `{ steps[], owners[] }` | operator | operator |
| Quartermaster | issue triage — missing labels, stale PRs | `{ repos[] }` | `{ applied_labels[] }` | gh-trio.timer | repo |
| Scribe | doc edits, wiki page updates | `{ path, change }` | `{ diff }` | Librarian / ops | repo |
| Sentinel | incident watchdog — logs, metrics, alarms | `{ window }` | `{ findings[], severity }` | sentinel (continuous) | echo |
| Sommelier | model / backend recommendation | `{ task_shape }` | `{ recommended_backend }` | Planner | Planner |
| Warden | secret + credential drift detection | `{ paths[] }` | `{ leaks[] }` | ops | Magistrate |

Every specialist implements `Specialist` (dyn-safe, `async_trait`) or the typed `TypedSpecialist` adapter via `Typed<T>`:

```rust
pub trait Specialist: Send + Sync {
    fn name(&self) -> Name;
    fn description(&self) -> &'static str { "" }
    fn input_schema(&self) -> Value { json!({ "type": "object" }) }
    fn output_schema(&self) -> Value { json!({ "type": "object" }) }
    async fn handle(&self, req: Value) -> Result<Value>;
}
```

Registry dispatch:

```rust
pub struct Registry { ... }
impl Registry {
    pub fn register(&mut self, s: Arc<dyn Specialist>);
    pub async fn dispatch(&self, name: Name, req: Value) -> Result<Value>;
}
```

The registry is the boundary between the agents bus (tokio) and the MCP server (stdio JSON-RPC). See [Why-Halo-Agents](./Why-Halo-Agents.md) for the rationale.

## Discord pipeline

Binary: `1bit-watch-discord` (crates/1bit-agents/bin/1bit-watch-discord.rs). Identity: **halo** (listener). Gateway intents required:

- `GUILDS` — list channels
- `GUILD_MESSAGES` — receive messages
- `MESSAGE_CONTENT` (privileged) — read the body

Posts are made under a **separate** identity (**echo**) so the listener stays a lurker and the poster's avatar shows the answering specialist.

### Flow

1. halo receives a `MESSAGE_CREATE` event.
2. If `is_direct_mention(content, bot_id)` is false and channel is not in the allowlist, drop.
3. Strip the mention → `strip_mention(content, bot_id)`.
4. `classify(text) -> Classification` in `crates/1bit-agents/src/watch/discord.rs`:
   - Bug signals (`bug`, `error:`, `panic`, `stack trace`, `crash`, `segfault`, `regression`, `broken`, `doesn't work`, …) → `BugReport → Sentinel`
   - Feature signals (`feature`, `would be nice`, `can we add`) → `FeatureRequest → Magistrate`
   - Question (ends `?` or wh-leading) → `Question → Herald`
   - Else → `Chat → Herald`
5. `Registry::dispatch(name, req)` runs the specialist.
6. echo posts the response. On `BugReport` an auto-thread is created on the message so follow-ups don't spam the channel.

Config:

- `DISCORD_BOT_TOKEN` — root-protected drop-in at `~/.config/systemd/user/strix-watch-discord.service.d/token.conf`
- `HALO_DISCORD_CHANNELS` — comma-separated channel ID allowlist, same drop-in
- Unit is **not** enabled by default; operator runs `systemctl --user enable --now strix-watch-discord` after dropping the token

## GitHub pipeline

Binary: `1bit-watch-github`. Auth: fine-grained PAT, read-only scopes across the `DEFAULT_REPOS` list (`bong-water-water-bong/1bit-systems`, `bong-water-water-bong/bitnet-mlx.rs`, `strix-ai-rs/halo-workspace`).

Polling:

```rust
pub const DEFAULT_POLL_SECONDS: u64 = 300;
```

Lookback window is `poll_seconds × 2` (default 600 s) to absorb clock skew and transient failures without duplicating work. Dedupe state is a JSON file at `~/.local/state/1bit-watch-github/seen.json` (`measured-TBD` — confirm exact path against the running binary).

Classification in `crates/1bit-agents/src/watch/github.rs::classify`:

- Any PR → `Magistrate`
- Label `bug` or title containing `error` / `crash` / `fail` → `Sentinel`
- Label `enhancement` or `feature` → `Planner`
- Label `documentation` → `Scribe`
- Else → `Sentinel` (safe default — the watchdog is always on call)

Output is routed into the agents registry identically to Discord.

## MCP surface

Crate: `crates/1bit-mcp`. Wire: **stdio JSON-RPC**, one object per `\n`-delimited line (Claude Code convention, not LSP `Content-Length`). Protocol version constant:

```rust
pub const PROTOCOL_VERSION: &str = "2024-11-05";
pub const SERVER_NAME: &str = "1bit-mcp";
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
```

`StdioServer` exposes:

- `tools/list` — derived from `Name::ALL` + `description_for(name)` + the specialist's `input_schema()`
- `tools/call` — dispatches through `Arc<Registry>` via `Registry::dispatch(name, req)`
- `resources/list`, `resources/read` — skill store (`crates/1bit-mcp/src/skills.rs`)
- `memory/get`, `memory/put` — session memory (`crates/1bit-mcp/src/memory.rs`)

The server is built via `StdioServer::with_default_agents()` (prebuilt 17-stub registry) or `StdioServer::new(agents, skills, memory)` for custom wiring. Claude Desktop and Claude Code attach via stdio. 22 in-crate tests cover the registry, the wire framing, and the error codes (`INVALID_REQUEST = -32600`, `METHOD_NOT_FOUND = -32601`, `INVALID_PARAMS = -32602`, `INTERNAL = -32603`, `UNKNOWN_TOOL = -32001`).

See [Clients](./Clients.md) §"MCP clients" for the integration recipes.

## Training pipeline

Out-of-box path: no retraining required — consume the `microsoft/bitnet-b1.58-2B-4T` weights, repack to `.h1b`. When a new quantization regime needs data (Sherry, Sparse-BitNet, Bonsai) the retrain runs on a persistent RunPod H200 pod. See `project_sparse_bitnet_run` and `feedback_pod_lifecycle` in memory — the pod is **persistent**; never ask for API keys or to "spin it up" as if rebuilding.

### Stack

- TRL + HuggingFace `datasets` (streaming loader) for data.
- Model is a BitLinear-patched Llama-3 reference — not our runtime path. Training is Python; runtime is Rust.
- `torch.compile` for the training step; no compile at inference (Rule A).

### Step cadence

```
batch_size     = 16
seq_len        = 2048
grad_accum     = 32
log_every      = 10    steps
save_every     = 100   steps
verify_nm_mask_every = 500 steps
```

Per-step tokens: `16 × 2048 × 32 ≈ 1.05 M tokens`. 10 B-token budget ≈ 9 600 steps.

### Throughput

H200 measured **49.5k tok/s** on the Sparse-BitNet 3:4 config. 10 B tokens / 49.5k tok/s ≈ **56 h wall-clock** for a full run. Run 3 bailed at step 500 — see [Run-3-Autopsy](./Run-3-Autopsy.md) if present (memory: `project_sparse_bitnet_run`).

### Artifact flow

pod → rsync to pi 100.64.0.4 (archive, ZFS) → requantizer (local, Rust) → `.h1b` → scp to strixhalo → PPL + smoke → deploy. (Pre-v0.1.0 this flow included a shadow-burnin leg against gen-1; that leg closed at cutover.) See [Why-Shadow-Burnin](./Why-Shadow-Burnin.md).

## Shadow-burnin (historical — cutover complete at v0.1.0)

Continuous v1-vs-v2 argmax comparison ran between gen-1 C++ (`:8080`) and gen-2 Rust (`:8180`) through the 2026-04-21 → 04-23 window. Same prompt, same sampler, logits + argmax compared per position.

```
interval        : 30 s
state file      : ~/.local/share/1bit-halo/shadow-burnin.state
log (JSONL)     : ~/claude output/shadow-burnin.jsonl
cutover gate    : ≥ 96% bit-exact argmax across 72 h rolling
final           : 96.66% (v0.1.0, cutover landed)
```

The `~/claude output/` quoting matters — the space is intentional, see memory `feedback_benchmark_output_folder`. Harness lives at `benchmarks/shadow-burnin.sh`. Retained for future backend swaps (e.g. NPU-backed serving) where the same comparator is useful. See [Why-Parity-Gates](./Why-Parity-Gates.md).

Cutover gates that were satisfied pre-v0.1.0:

1. PPL parity on wikitext within **±0.05** of the gen-1 baseline (9.1607). Final gen-2: 9.1805 → delta +0.02 → PASS.
2. Shadow-burnin **≥96%** byte-exact across 1500+ rounds → PASS.

As of v0.1.0 (2026-04-24) gen-2 Rust `1bit-server` owns `/v1/*` on `:8180`. The gen-1 `bitnet_decode` unit is disabled.

## Mesh

Four-node Headscale private tailnet. Coordinator lives on strixhalo.

```
  strixhalo  100.64.0.1  gfx1151 iGPU   primary inference, Caddy, Headscale
  sliger     100.64.0.2  NVIDIA 1080 Ti failover candidate, control-plane mirror
  ryzen      100.64.0.3  RX 9070 XT     second kernel target (gfx1201)
  pi         100.64.0.4  ARM + ZFS 3.6T canonical archive, nightly rsync
```

Full table in [Network-Topology](./Network-Topology.md). IPv6 companions under `fd7a:115c:a1e0::/64`.

### Who talks to what

- **strixhalo → pi (100.64.0.4)**: nightly rsync of `/home/bcloud/claude output/` + model tombstones + `.h1b` experiments. pi is the canonical archive; local originals are pruned after 7 days (memory: `feedback_archive_retention`).
- **strixhalo → sliger (100.64.0.2)**: ad-hoc dev; control-plane failover candidate.
- **strixhalo → ryzen (100.64.0.3)**: kernel work for the second target (gfx1201). `rocm-cpp` builds cross-compile in theory but realistically SSH + native compile. 2.5 Gb LAN fronts the tailnet.
- **pi → strixhalo**: halo-brain recall reads from the archive; the pi never initiates control-plane traffic, only serves reads.

### Why CGNAT space is fine

The `100.64.0.0/10` range is CGNAT-reserved in RFC 6598; it is specifically allocated for carrier-grade NAT and will never collide with a publicly routed address the host might see. Tailscale/Headscale picked this range for exactly that reason. NAT traversal happens via Headscale's STUN on `:3478`; every pair in the current topology shows `in 0s–1ms` on `tailscale ping` — direct LAN, no DERP. See [Network-Topology](./Network-Topology.md).

## Deployment

Systemd unit graph (user scope, installed from `strixhalo/systemd/`):

```
  strix-server.service        — 1bit-halo-server :8180   (production LLM, Rust)
  strix-lemonade.service      — 1bit-lemonade :8200
  strix-landing.service       — 1bit-landing :8190
  strix-echo.service          — echo poster (Discord)
  strix-watch-discord.service — halo listener
  strix-watch-github.timer    — 300 s poll
  1bit-halo-sd.service        — sd.cpp :8081
  1bit-halo-anvil.timer       — kernel rebuild on rocm-cpp commits
  1bit-halo-memory-sync.timer — GH push of ~/.claude/.../memory every 15m
  strix-cloudflared.service   — optional public ingress

  Retired (disabled, kept for archive):
  1bit-halo-bitnet.service    — gen-1 bitnet_decode :8080 (cutover at v0.1.0)
  strix-burnin.service        — shadow-burnin harness (kept for future backend swaps)

  On sliger (separate host, 100.64.0.2):
  1bit-halo-whisper.service   — STT :8190 (Arc B580 Vulkan)
  1bit-halo-kokoro.service    — TTS :8191 (Arc B580 Vulkan)
```

Caddy fronts TLS at `:443` (system-level service, not user). Headscale fronts at `:8380` (loopback) fronted by Caddy as `headscale.<host>` on `:443`. Full layout: [Why-Caddy-Systemd](./Why-Caddy-Systemd.md).

### Install flow

`1bit install <component>` reads `packages.toml` at the repo root, which lists every unit + binary + source path:

```toml
[components.1bit-server]
unit   = "strix-server.service"
binary = "1bit-server-real"
source = "crates/1bit-server"
deps   = ["rocm-cpp"]
```

`1bit install` stops the unit → `cargo install --path <source>` → copies to `~/.local/bin/` → `systemctl --user start`. Idempotent; re-runs are safe. See [Installation](./Installation.md) for the operator-facing walkthrough.

## Supply chain

Dependencies we trust, with rationale:

| dep | where | why |
|---|---|---|
| TheRock (ROCm dist) | system `/opt/rocm/` | AMD-official; hipcc + runtime |
| composable_kernel | reference | HIP kernel patterns (reference-only, not linked) |
| serenity-rs | `1bit-watch-discord` | Discord gateway client, pure Rust, maintained |
| octocrab | `1bit-watch-github` | GitHub API, typed, pure Rust |
| axum + tower | `1bit-halo-server` | HTTP framework, tokio-native |
| hyper + reqwest | server + client | HTTP/1.1 + HTTP/2 |
| serde + serde_json | everywhere | typed JSON at the deserializer boundary |
| async-openai | client examples only | not linked into the server |
| nlohmann/json | `rocm-cpp` tools | C++ JSON for model metadata |
| cpp-httplib | `rocm-cpp` tools | C++ HTTP for standalone utilities |

### Dependencies we deliberately don't use

- **hipBLAS (runtime)** — banned in the runtime path (Rule C). Heuristic collapses on the skinny ternary GEMV shape; native kernels win outright.
- **torch / any Python-serving-path lib** — Rule A. Python is fine for one-shot scripts (requantizer, H200 training) and forbidden in anything that serves HTTP.
- **Python Open WebUI in-proc** — allowed as a caller-side `/v2/*` client only (`feedback_openwebui_exemption`), sunsets on Helm v0.3.
- **wgpu / gfx ecosystem** — candidate for the cross-platform renderer in Helm, not touched in the serving path.

## Failure surface + recovery

Per-service failure modes + the `Restart=` policy + rollback pattern. See [Troubleshooting](./Troubleshooting.md) for the operator-facing diagnostics.

| service | likely failure | `Restart=` | blast radius | recovery |
|---|---|---|---|---|
| strix-server | OOM, FFI panic, ROCm driver stuck | `on-failure`, 5 s | `/v2/*` returns 502 via Caddy | restart unit; if ROCm stuck, `hip-dev-reset.sh` or reboot |
| 1bit-halo-bitnet | same | `on-failure`, 5 s | `/v1/*` | same |
| 1bit-halo-whisper | STT model load fail | `on-failure`, 10 s | voice loop goes silent | re-download model, restart |
| 1bit-halo-kokoro | onnxruntime init fail | `on-failure`, 10 s | voice loop silent | check ORT dylib path |
| 1bit-halo-sd | VAE OOM on large image | `on-failure`, 10 s | image-gen 5xx | cap resolution; restart |
| strix-watch-discord | token missing | `on-failure` (clean exit ignored) | no listener | drop in token override, enable |
| strix-watch-github | PAT expired | `on-failure`, 10 s | no GH triage | rotate PAT, restart |
| 1bit-halo-memory-sync | GH creds expired | timer unit | memory not pushed | rotate PAT (memory: `project_halo_memory_sync_broken`) |
| headscale | coordinator crash | `on-failure` | mesh loses new-node admit; existing tunnels persist | restart |
| caddy | config reload fail | `always` | TLS down | `caddy validate` then reload |

### amdgpu OPTC CRTC hang

The signature `REG_WAIT timeout 1us * 100000 tries - optc35_disable_crtc` is a gfx1151 kernel bug that freezes Wayland and requires a power-cycle (memory: `project_amdgpu_optc_hang`). Mitigations:

- `halo-gpu-perf.service` pins SCLK high via sysfs (memory: `project_optc_mitigation`)
- Tier-2 tried GFXOFF-off + runpm=0 (memory: `project_session_crash_20260421`) — crashes 5 & 6 still happened
- 2026-04-22 kernel rollback to 6.18.22-lts: sustained freeze gone, early-boot OPTC signature still emits (non-fatal); NPU unavailable on LTS (memory: `project_kernel_rollback_20260422`)

### Rollback via snapper

Btrfs + snapper snapshots. Snapshot #6 "7.00 with claude" is the designated pre-rollback baseline (memory: `fallback_snapshot`). Kernel-level issues that persist across reboots: roll back to #6, reboot into the limine entry, lose ~hours of state but keep the repo untouched (repo is independently git-tracked).

### Memory-sync recovery

`halo-memory-sync.timer` fires every 15 minutes, pushes `~/.claude/projects/-home-bcloud/memory/` to the private GH repo. 2026-04-21 failure: GH PAT expired; timer logged "push failed (credentials?)". Post-reboot works; if it resumes failing, rotate the PAT (memory: `project_halo_memory_sync_broken`).

## Cross-links

- [Installation](./Installation.md)
- [Troubleshooting](./Troubleshooting.md)
- [Clients](./Clients.md)
- [Observability](./Observability.md)
- [Why-Ternary](./Why-Ternary.md)
- [Why-Strix-Halo](./Why-Strix-Halo.md)
- [Network-Topology](./Network-Topology.md)
- [Why-Shadow-Burnin](./Why-Shadow-Burnin.md)
- [Why-H1b-Format](./Why-H1b-Format.md)
- [Why-Caddy-Systemd](./Why-Caddy-Systemd.md)
- [Why-No-Python](./Why-No-Python.md)
- [Why-Rust](./Why-Rust.md)
- [Why-Parity-Gates](./Why-Parity-Gates.md)
- [Why-No-NPU-Yet](./Why-No-NPU-Yet.md)
- [Why-This-Way-How](./Why-This-Way-How.md)
- [ARCHITECTURE.md](../../ARCHITECTURE.md)
- [CUTOVER.md](../../CUTOVER.md)
- [DEMO.md](../../DEMO.md)

## Appendix A — byte accounting for one 32-token reply

Breakdown for a 10-token prompt producing a 32-token reply on halo v2 (2B, 30 layers, hidden 2048, head_dim 256, 8 KV heads, FP16). Weight bytes dominate — this is why bandwidth, not compute, is the ceiling.

### Per forward pass (one token)

```
Q/K/V GEMV reads (packed ternary):
  hidden_size × hidden_size × 2 bits / 8    = 2048 × 2048 × 0.25 B = 1.0 MiB × 3 ≈ 3.0 MiB per layer
O GEMV reads:                                                      ≈ 1.0 MiB per layer
gate + up GEMVs:
  hidden × intermediate × 2 bits / 8        = 2048 × 5504 × 0.25 B ≈ 2.7 MiB × 2 ≈ 5.4 MiB per layer
down GEMV:                                                         ≈ 2.7 MiB per layer
row-scale reads (f32):                                             ≈ 32 KiB per layer
per-layer subtotal                                                 ≈ 12.1 MiB
× 30 layers                                                        ≈ 363 MiB per token
```

### Per-reply total

```
prefill: 10 prompt tokens × 363 MiB      ≈ 3.5 GiB of weight reads
decode : 32 reply tokens × 363 MiB       ≈ 11.3 GiB of weight reads
total                                    ≈ 14.8 GiB
```

At 240 GB/s observed bandwidth (92% of the 256 GB/s LPDDR5 peak), theoretical floor = `14.8 GiB / 240 GB/s ≈ 63 ms`. Observed wall-clock on decode-only at 83 tok/s = `32 / 83 ≈ 385 ms`. Gap is attention + FFN fused-activation + sampler + host roundtrip. This is the work Sherry (1.25-bit packing) and MedusaBitNet (speculative decode) attack.

### KV-cache growth

```
per-token per-layer KV = 2 × 8 heads × 256 dim × 2 B = 8 KiB
× 30 layers            = 240 KiB per token
10 prompt + 32 reply   = 42 tokens × 240 KiB ≈ 10 MiB
```

KV is a rounding error at short contexts. At 4096 tokens: 960 MiB. At 32768: 7.7 GiB — still comfortable in the 128 GiB budget.

## Appendix B — FFI boundary cheat sheet

The airlock between Rust and `rocm-cpp`. Every function is `extern "C"`, no C++ name mangling, no Rust panic safety crossed. Pointers are newtyped (`DevicePtr<T>`, `DeviceMutPtr<T>`) to prevent accidental CPU-side dereference in Rust.

Representative signatures from `crates/1bit-hip/src/lib.rs`:

```rust
pub fn ternary_gemv_halo_f16(
    packed_weights: DevicePtr<u8>,
    activations_i8: DevicePtr<i8>,
    activation_scale: f32,
    row_scales: DevicePtr<f32>,
    output_f16: DeviceMutPtr<u16>,
    m: i32, k: i32,
    stream: HipStream,
) -> RcppStatus;

pub fn kv_cache_attn_decode_fd(
    q: DevicePtr<u16>,
    k_cache: DevicePtr<u16>,
    v_cache: DevicePtr<u16>,
    output: DeviceMutPtr<u16>,
    pos: i32, num_heads: i32, num_kv_heads: i32,
    head_dim: i32, max_seq_len: i32,
    stream: HipStream,
) -> RcppStatus;

pub fn rope_fp16(
    q: DeviceMutPtr<u16>, k: DeviceMutPtr<u16>,
    pos: i32, num_heads: i32, num_kv_heads: i32,
    head_dim: i32, theta: f32,
    stream: HipStream,
) -> RcppStatus;

pub fn rmsnorm_fp16(
    input: DevicePtr<u16>, weight: DevicePtr<u16>,
    output: DeviceMutPtr<u16>,
    rows: i32, cols: i32, eps: f32,
    stream: HipStream,
) -> RcppStatus;

pub fn silu_glu_fp16(
    gate: DevicePtr<u16>, up: DevicePtr<u16>,
    output: DeviceMutPtr<u16>,
    n: i32, stream: HipStream,
) -> RcppStatus;
```

Status codes:

```rust
pub enum RcppStatus {
    Ok = 0,
    InvalidArg = 1,
    DeviceError = 2,
    OutOfMemory = 3,
    Unimplemented = 4,
    Internal = 5,
}
```

`RcppStatus::from_raw` converts the C int. Anything non-zero bubbles up through `1bit-hip` as `RcppError`, into `lemond` as `BackendError::Hip`, and into the HTTP layer as `ServerError::Backend` (HTTP 500). No panic crosses the FFI airlock — C++ exceptions are caught at the extern boundary in `rocm-cpp`.

## Appendix C — who-calls-what (concrete call graph)

Top-down, loopback hop callers only. External callers (Open WebUI, curl, Helm) arrive at Caddy.

```
HTTP client
  → caddy.service (/v1/* or /v2/*)
    → bitnet_decode :8080  (C++, gen-1)       — /v1/*
    → 1bit-halo-server :8180  (Rust, gen-2)   — /v2/*
        → AppState { backend: SharedBackend, metrics, sd_base_url, http_client }
        → build_router_with_state(state)
          → route("/v1/chat/completions")  → chat_completions handler
            → Backend::Hip via lemond::Router::run
              → HipBackend::forward
                → 1bit-hip::ternary_gemv_halo_f16 (Q, K, V, O, gate, up, down)
                → 1bit-hip::rmsnorm_fp16
                → 1bit-hip::rope_fp16
                → 1bit-hip::kv_cache_attn_decode_fd
                → 1bit-hip::silu_glu_fp16
                → 1bit-hip::residual_add_fp16
                → 1bit-hip::argmax_fp32 (or sampler path)
          → route("/v1/models")            → list_models
          → route("/ppl")                  → ppl (wikitext-103 harness)
          → route("/v2/images/generations") → images_generations
            → reqwest → sd.cpp :8081
          → route("/v1/npu/status")        → npu::npu_status
        → metrics.record_request()

1bit-watch-discord (listener)
  → serenity gateway
  → classify() → Name::{Sentinel, Magistrate, Herald}
  → Registry::dispatch(name, req)
  → echo (poster identity) → channel or thread

1bit-watch-github (poller, 300 s)
  → octocrab
  → classify() → Name::{Magistrate, Sentinel, Planner, Scribe}
  → Registry::dispatch(name, req)
  → echo or repo action

halo-mcp (stdio)
  → JSON-RPC 2.0 over stdin/stdout
  → StdioServer::handle_request
  → Registry::dispatch(name, req)
```

## Appendix D — configuration surface

Environment variables read at startup (not exhaustive):

| var | consumer | default | notes |
|---|---|---|---|
| `HALO_BACKEND` | lemond | `hip` | `hip` \| `cpu` |
| `HALO_SAMPLER` | lemond | `inline` | `inline` \| `cpu` |
| `HALO_PREFILL_CROSSOVER_L` | lemond | 33 | prompt length at which prefill moves to CPU AVX2 lane |
| `HALO_SD_URL` | 1bit-halo-server | `http://127.0.0.1:8081` | image-gen sidecar |
| `HALO_GH_REPOS` | 1bit-watch-github | see `DEFAULT_REPOS` | comma-separated owner/repo |
| `HALO_GH_POLL_SECONDS` | 1bit-watch-github | 300 | poll interval |
| `DISCORD_BOT_TOKEN` | 1bit-watch-discord | (required) | drop-in at systemd override |
| `HALO_DISCORD_CHANNELS` | 1bit-watch-discord | (none) | comma-separated channel allowlist |
| `RUST_LOG` | every Rust binary | `info` | tracing subscriber filter |
| `LD_LIBRARY_PATH` | strix-server.service | `<home>/repos/rocm-cpp/build:/opt/rocm/lib` | finds `librocm_cpp.so` |

On-disk state:

| path | consumer | contents |
|---|---|---|
| `/etc/caddy/Caddyfile` | caddy | TLS + bearer + path split |
| `~/.local/bin/1bit-server-real` | strix-server | built binary |
| `~/1bit systems/models/halo-1bit-2b.h1b` | 1bit-halo-server | model weights (space intentional) |
| `~/.local/share/1bit-halo/shadow-burnin.state` | burnin harness | last-tested prompt cursor |
| `~/claude output/shadow-burnin.jsonl` | burnin harness | one line per compared completion |
| `~/claude output/*.json` | benchmark scripts | bench outputs (memory: `feedback_benchmark_output_folder`) |
| `~/.config/systemd/user/strix-watch-discord.service.d/token.conf` | discord watcher | bot token override |
| `packages.toml` | `1bit install` | component manifest |

## Appendix E — metrics exposed

`GET /metrics` on :8180 returns a `MetricsSnapshot` (JSON, not Prometheus — the Prometheus export lives behind `feat/prom-exporter` still). Shape (see `crates/1bit-server/src/metrics.rs`):

```json
{
  "requests_total": 0,
  "prompt_tokens_total": 0,
  "completion_tokens_total": 0,
  "latency_ms_p50": 0,
  "latency_ms_p99": 0,
  "backend": "hip",
  "uptime_s": 0
}
```

Observability path: `journalctl --user -u strix-server -f` for logs; `halo logs <unit>` wraps that. See [Observability](./Observability.md) for the rocprof bandwidth check and the PPL harness.

## Open numbers (measurement-TBD)

- Per-call FFI overhead from Rust into `rocm-cpp` (`ternary_gemv_halo_f16`, `kv_cache_attn_decode_fd`, `rmsnorm_fp16`). Hand-rolled `extern "C"`, no bindgen; target measurement is median + p99 per call.
- Caddy → axum loopback HTTP overhead on a warm pool.
- `~/.local/state/1bit-watch-github/seen.json` exact path (confirm against running binary).
- p99 decode latency at L=2048 under concurrent 2-session load (KV-cache mutex contention).
- Bytes-per-step math cross-check on H200 (`16 × 2048 × 32 ≈ 1.05 M`) against the live TRL logger.
