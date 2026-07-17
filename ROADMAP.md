# Roadmap: Zaya Co-Host — Your AI Voice, Rented Out

**The Zaya inference stack becomes the backbone of a voice cloning + real-time AI co-host platform.** Clone your voice once, deploy it as a low-latency streaming API, and rent it out for $19.95/month. Your voice becomes a co-host for podcasts, streams, tutorials, customer service, interactive fiction, or anything else.

```
🎤 Mic → [STT] → [Zaya LLM] → [TTS (cloned voice)] → 🔊 Speaker
         NPU       NPU/GPU       CPU/NPU               Any device
```

---

## Phase 1: Voice Cloning Pipeline (Now — 2 weeks)

### 1.1 Voice Capture & Training
| Task | Status | Tech |
|------|--------|------|
| Record ~30min high-quality voice samples | 🔲 Prompt set ready | 24kHz mono, arecord |
| Train audio codec on voice samples | 🔲 | Pure PyTorch RVQ-VAE (5.87M params) |
| Train ZAYA voice adapter (QLoRA: text→codec tokens) | 🔲 | QLoRA on AMD ROCm, 4-bit NF4 |
| Export to ONNX for agnostic inference | 🔲 | Pure PyTorch → ONNX, runs anywhere |
| Validate clone quality (MOS score >4.0) | 🔲 | Blind A/B test vs original |

**Deliverable:** A single voice pack file (`.voice`) containing:
- Codec decoder weights (5.87M params — pure PyTorch)
- Speaker embedding (~512 bytes)
- Voice metadata (name, sample rate, pitch range)
- **Total: ~25 MB** (vs 50-100 MB for XTTS-v2)

### Why our codec instead of XTTS-v2

| Factor | XTTS-v2 | Our RVQ-VAE Codec |
|--------|---------|-------------------|
| **Params** | ~1.2B | **5.87M** |
| **Voice pack size** | 50-100 MB | **~25 MB** |
| **Vendor lock-in** | Coqui-AI specific | **Agnostic** (pure PyTorch) |
| **Hardware** | Needs GPU for training | **AMD ROCm, CUDA, CPU** |
| **LLM integration** | Separate model | **Tight — tokens flow through ZAYA** |
| **Streaming** | Chunked utterance | **Frame-level (13ms frames)** |
| **License** | CC-BY-NC 4.0 | **MIT** |

### 1.2 Agnostic TTS Engine (replaces Piper)
| Task | Status | Notes |
|------|--------|-------|
| Python codec decoder → ONNX export | ✅ Built | `zaya_audio/codec.py` — 5.87M params |
| ONNX decoder for GPU/NPU/CPU inference | 🔲 | ONNX export path |
| C++ codec decoder for embedded | 🔲 | llama.cpp-style GGUF conversion |
| Streaming audio output (chunked HTTP/WS) | 🔲 | SSE / WebSocket |
| Voice quality / latency benchmarks | 🔲 | Target: <100ms to first audio |

### 1.3 Jarvis Audio Server Updates
- [ ] Add `/v1/voice/clone` — upload samples, train, download `.voice` pack
- [ ] Add `/v1/audio/speech` — accept `voice` param (cloned or stock)
- [ ] Add `/v1/audio/stream` — WebSocket streaming endpoint
- [ ] Hot-reload voice packs without restart

---

## Phase 2: Zaya Co-Host Runtime (2-4 weeks)

### 2.1 Real-Time Voice Pipeline

```
[User Mic] → Whisper STT → Zaya LLM → Voice TTS → [Streaming Audio]
                NPU           GPU/NPU     CPU/NPU      HTTP/WS
               ~1.5s        69+ tok/s   <100ms       Chunked
```

| Component | Target | Current | Gap |
|-----------|--------|---------|-----|
| STT latency (first word) | <500ms | ~1.5s | Optimize Whisper or switch to faster model |
| LLM time-to-first-token | <50ms | ~9ms/tok | ✅ Already fast enough |
| TTS time-to-first-audio | <100ms | ~50ms (Piper) | Need cloned voice engine |
| End-to-end latency | <1.5s | ~2s | Optimize STT + parallelize pipeline |
| Streaming support | Chunked audio | Not yet | Add WebSocket + SSE |

### 2.2 Co-Host Intelligence
- [ ] **Persona system** — Define voice personality, speaking style, catchphrases, knowledge domains
- [ ] **Context memory** — Per-session conversation history with summarization
- [ ] **Knowledge base RAG** — Existing Jarvis KB system (docs, facts, scripts)
- [ ] **Interruption handling** — Detect when user interrupts, stop generation gracefully
- [ ] **Turn-taking** — Natural conversation flow with VAD (Voice Activity Detection)

### 2.3 Commercial API (SaaS Layer)
| Endpoint | Purpose | Pricing |
|----------|---------|---------|
| `POST /v1/chat/completions` | Text chat (existing) | Included |
| `POST /v1/audio/chat` | Voice in → voice out | Per-minute billing |
| `WS /v1/audio/stream` | Streaming voice conversation | Per-minute billing |
| `POST /v1/voice/clone` | Create voice clone | One-time fee |
| `GET /v1/voice/packs` | List available voice packs | Included |

**Pricing:** $19.95/month for:
- 1 cloned voice
- 10 hours of voice conversation / month
- Web dashboard + usage analytics
- API key access
- Custom persona configuration

---

## Phase 3: Platform (4-8 weeks)

### 3.1 Billing & Accounts
- [ ] Stripe integration (already have Stripe account)
- [ ] API key management with rate limiting
- [ ] Usage tracking (minutes of voice, tokens processed)
- [ ] Monthly billing with overage metering
- [ ] Free tier: 30 min demo / month

### 3.2 Web Dashboard
- [ ] Voice pack management (upload, test, deploy)
- [ ] Persona editor (tone, style, knowledge sources)
- [ ] Usage analytics (graphs, logs, quality metrics)
- [ ] API key management
- [ ] Billing portal (Stripe customer portal)

### 3.3 Voice Marketplace
- [ ] Voice pack template library (stock voices)
- [ ] Community voice packs (user-created, royalty-managed)
- [ ] Voice pack versioning and rollback
- [ ] Quality leaderboard (MOS scores, user ratings)

---

## Phase 4: Scale & Monetize (8-16 weeks)

### 4.1 Platform Features
| Feature | Priority | Notes |
|---------|----------|-------|
| Multi-voice conversations | High | Podcast with multiple AI co-hosts |
| Live streaming integration | High | OBS plugin, Twitch/YouTube integration |
| Phone call integration | Medium | Twilio SIP → voice pipeline |
| Multi-language voice cloning | Medium | Clone voice in any language |
| Emotional range | Low | Happy/sad/excited variants |

### 4.2 Enterprise Tier
- **$99/mo** — 5 voices, 100 hours, team accounts
- **$499/mo** — Unlimited voices, white-label API, SLA
- **Custom** — On-premise deployment (for studios, broadcasters)

### 4.3 Distribution Channels
- **OBS Plugin** — Real-time AI co-host for streamers
- **Discord Bot** — Voice co-host for communities
- **Twitch Extension** — Interactive AI co-host for live chat
- **Web Embed** — `<script>` tag for any website
- **Mobile SDK** — iOS/Android voice SDK

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Client Layer                        │
│  Web App │ OBS Plugin │ Discord Bot │ Mobile SDK │ API  │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP/WS
┌────────────────────────▼────────────────────────────────┐
│                    API Gateway                           │
│  Auth │ Rate Limiting │ Billing │ Usage Tracking         │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Zaya Co-Host Server (Agnostic)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ Whisper  │  │ Any LLM  │  │ Voice Codec (5.87M)  │  │
│  │ STT (NPU)│  │(agnostic)│  │ tokens ↔ audio       │  │
│  └──────────┘  └──────────┘  └──────────────────────┘  │
│  ┌──────────────────────────────────────────┐           │
│  │  Persona Engine │ Context Memory │ RAG    │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 Voice Pipeline Flow                      │
│                                                         │
│  ┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────┐  │
│  │ Mic  │───→│  Codec   │───→│   Any    │───→│ Codec│──→│
│  │      │    │ Encoder  │    │   LLM    │    │Decoder│  │
│  │      │    │ audio→   │    │ tokens→  │    │ tokens│  │
│  │      │    │ tokens   │    │ tokens   │    │→audio │  │
│  └──────┘    └──────────┘    └──────────┘    └──────┘  │
│               ▲                             ▲           │
│         ┌─────┴──────────┐          ┌──────┴────────┐  │
│         │  Voice Pack     │          │  Voice Pack   │  │
│         │  (speaker       │          │  (decoder     │  │
│         │   embedding)    │          │   weights)    │  │
│         └────────────────┘          └───────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 Storage Layer                            │
│  Voice Packs │ User Data │ KB Docs │ Usage Logs          │
└─────────────────────────────────────────────────────────┘
```

## Hardware Requirements

| Component | Requirement | Current Hardware |
|-----------|-------------|-----------------|
| LLM Inference | NPU or GPU | ✅ Strix Halo (69 tok/s NPU, 426 tok/s GPU) |
| STT (Whisper) | NPU recommended | ⚠️ Runs on CPU (faster-whisper) |
| TTS (Voice Clone) | CPU, GPU, or NPU | 🟡 Codec decoder (5.87M params) — pure PyTorch, ONNX pending |
| Streaming | Network | ✅ Local server running |

## Success Metrics

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|--------|---------|---------|---------|---------|
| Voice clone quality (MOS) | >4.0 | >4.2 | >4.5 | >4.5 |
| STT latency (first word) | <1.5s | <500ms | <300ms | <200ms |
| TTS latency | <100ms | <80ms | <50ms | <30ms |
| End-to-end voice latency | <2s | <1.5s | <1s | <500ms |
| Paying customers | — | 10 | 100 | 1,000+ |
| Monthly revenue | — | $200 | $2,000 | $20,000+ |
| Uptime SLA | — | 99% | 99.5% | 99.9% |

## Business Model

**Price:** $19.85/mo per user.
**Hardware:** Strix Halo (owned) → MI300X (scale).
**Inference cost:** $0 on owned hardware.
**Strix Halo capacity:** 28 voice users @ 426 tok/s.
**MI300X break-even:** 37 users ($720/mo rent).

See [docs/business-plan.md](docs/business-plan.md) for full economics.

## Immediate Next Steps (This Week)

1. **Record voice samples** — 30 min clean audio, 24kHz mono, use `jarvis/voice/record.py`
2. **Train audio codec** — Fit our 5.87M RVQ-VAE on voice data (AMD ROCm)
3. **Train ZAYA voice adapter** — QLoRA fine-tune (text→codec tokens) on Radeon 8060S
4. **Export codec to ONNX** — Agnostic inference path (CPU, GPU, NPU)
5. **Integrate voice pipeline into Jarvis** — Replace Piper with codec-based voice engine
6. **Test end-to-end** — 🎤 → Codec Encode → ZAYA → Codec Decode → 🔊
7. **Set up Stripe billing** — Already have Stripe account (acct_1TUvAmJnXt3bWED8)

*Agnostic by design — voice packs work with any LLM, any hardware, any platform.*

---

*Built on 1bit.systems · AMD Strix Halo · NPU + GPU + CPU · Zaya inference engine · Jarvis audio server · MI300X scale*
