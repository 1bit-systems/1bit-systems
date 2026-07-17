# Zaya Co-Host — Business Plan

**Product:** Agnostic voice AI platform. Clone your voice once, deploy anywhere.
**Price:** $19.85/month per user.
**Hardware:** AMD Strix Halo → MI300X (scale).
**Backend:** ZAYA-8B + agnostic codec decoder (5.87M params).
**Edge:** Zero inference cost on owned hardware. Full voice pipeline on AMD.

---

## 1. Product Overview

### What it is

A complete voice AI agent platform:
- **Voice cloning** — Record 30min of audio → `.voice` pack (~25MB)
- **Real-time TTS** — Any LLM → codec tokens → cloned voice → waveform
- **Agnostic** — Works with ZAYA, GPT, Claude, Llama, any model
- **Embeddable** — OBS plugin, Discord bot, Web SDK, API

### What it is not

- Not a SaaS with per-token fees
- Not locked to NVIDIA
- Not dependent on cloud inference

### The moat

Your hardware is the moat. Strix Halo does 426 tok/s on ZAYA-8B and costs $0/mo to run inference. Competitors pay $1-3/hr per GPU. You pay power.

---

## 2. Economics

### 2.1 Unit Economics

| Item | Cost |
|------|------|
| Strix Halo inference | **$0.00/mo** (already owned) |
| Power (120W × 24/7 × $0.12/kWh) | $10.37/mo |
| Bandwidth (10 users, 1TB) | ~$10/mo |
| **Cost to serve 10 users** | **~$20/mo** |
| **Revenue per user** | **$19.85/mo** |
| **Gross margin** | **~90%** |

### 2.2 Capacity

| Hardware | Memory | Max voice users | Max chat users |
|----------|--------|-----------------|----------------|
| **Strix Halo** (owned) | 128.8 GB | **28** (compute) / **183** (memory) | 14 |
| **MI300X** (rented, $1/hr) | 192 GB | **16** (compute) / **292** (memory) | 8 |
| **2× MI300X** | 384 GB | **32** / **584** | 16 |

*Bottleneck is compute (426 tok/s GPU ternary), not memory (115GB available for KV cache).*

### 2.3 Cost Structure

| Item | Monthly |
|------|---------|
| Strix Halo (included) | ~~$1,000+~~ **$0** |
| MI300X rent (@ $1/hr 24/7) | $720 |
| Power (Strix + MI300X) | ~$20 |
| Domain + DNS | ~$10 |
| Stripe fees (2.9% + $0.30) | ~$0.88 per transaction |
| CDN / object storage | ~$10 |
| **Fixed cost** | **~$40/mo** (no cloud GPU) |
| **With 1 MI300X** | **~$760/mo** |

### 2.4 Revenue Scenarios

#### Phase 1: Strix Halo only (month 1-3)

| Users | Revenue | Cost | **Net** | Margin |
|-------|---------|------|---------|--------|
| 1 | $20 | $10 | **+$10** | 50% |
| 5 | $99 | $10 | **+$89** | 90% |
| 10 | $198 | $10 | **+$188** | 95% |
| 20 | $397 | $10 | **+$387** | 97% |
| **28** (max) | **$556** | **$10** | **+$546** | **98%** |

#### Phase 2: Strix + 1× MI300X (month 3-6)

| Users | Revenue | Cost | **Net** | Note |
|-------|---------|------|---------|------|
| 30 | $595 | $760 | **-$165** | Underwater |
| **37** | **$734** | **$760** | **+$14** | **Break-even** |
| 50 | $993 | $760 | **+$233** | Profitable |
| 75 | $1,489 | $760 | **+$729** | Healthy |
| **100** | **$1,985** | **$760** | **+$1,225** | **Solo founder salary** |
| 150 | $2,978 | $1,480 | **+$1,498** | 2 cards |
| 200 | $3,970 | $1,480 | **+$2,490** | 2 cards |

#### Phase 3: Scale (month 6+)

| Users | MI300X | Rent | Revenue | **Net** |
|-------|--------|------|---------|---------|
| 100 | 1 | $720 | $1,985 | **+$1,265** |
| 200 | 2 | $1,440 | $3,970 | **+$2,530** |
| 500 | 5 | $3,600 | $9,925 | **+$6,325** |
| 1,000 | 10 | $7,200 | $19,850 | **+$12,650** |
| 5,000 | 50 | $36,000 | $99,250 | **+$63,250** |
| 10,000 | 100 | $72,000 | $198,500 | **+$126,500** |

### 2.5 Break-even Summary

| Scenario | Break-even | Take rate |
|----------|------------|-----------|
| Strix Halo only | **1 user** (power + bandwidth) | Instant |
| Strix + 1× MI300X | **37 users** ($720 rent) | ~3 months |
| Strix + 2× MI300X | **74 users** ($1,440 rent) | ~4 months |
| Cloud-only (no Strix) | **37 users** ($720 rent) | Same |

---

## 3. Enterprise Tier

| Tier | Price | What you get |
|------|-------|-------------|
| **Individual** | $19.85/mo | 1 voice, 10 hrs voice/mo, web dashboard |
| **Creator** | $49/mo | 3 voices, 50 hrs, OBS plugin, priority |
| **Team** | $99/mo | 5 voices, 100 hrs, team accounts, API keys |
| **Enterprise** | $499/mo | Unlimited voices, white-label, on-prem, SLA |

### Enterprise infrastructure

| Component | Enterprise option |
|-----------|------------------|
| **Hardware** | Customer provides or rent from us |
| **Deployment** | Docker → bare metal → their cloud |
| **SLA** | 99.5% uptime, <500ms latency |
| **Support** | Slack channel, 4hr response |
| **Custom** | Custom voices, custom LLM fine-tuning |

---

## 4. Distribution Channels

| Channel | Reach | Revenue model | Build time |
|---------|-------|---------------|------------|
| **Direct API** | Developers, SaaS | $19.85/mo per seat | 2 weeks |
| **OBS Plugin** | Streamers, podcasters | $49/mo Creator | 4 weeks |
| **Discord Bot** | Communities, gamers | $19.85/mo per server | 3 weeks |
| **Twitch Extension** | Live streamers | $49/mo + per-sub share | 6 weeks |
| **Web Embed** | Any website | $19.85/mo per domain | 3 weeks |
| **Mobile SDK** | iOS/Android apps | Enterprise only | 8 weeks |

---

## 5. Roadmap to Revenue

### Month 1: MVP

- [ ] **Record voice** — 30min audio, build first `.voice` pack
- [ ] **Train codec** — Fit 5.87M RVQ-VAE on voice data
- [ ] **Train ZAYA adapter** — QLoRA (text→codec tokens)
- [ ] **Ship API** — `/v1/audio/speech` with cloned voice
- [ ] **First 10 users** — Free trial → $19.85/mo

**Goal: $198/mo revenue. Cost: $10/mo.**

### Month 2: Product

- [ ] **Stripe billing** — Subscriptions, proration, dunning
- [ ] **Web dashboard** — Usage analytics, voice management
- [ ] **OBS plugin** — Real-time AI co-host for streamers
- [ ] **Discord bot** — Voice co-host for communities

**Goal: $556/mo (Strix Halo capacity). Cost: $10/mo.**

### Month 3-4: Scale

- [ ] **Rent MI300X** at 28+ users
- [ ] **Harden serving** — vLLM, batching, concurrent users
- [ ] **Usage monitoring** — Per-user metering, rate limits
- [ ] **Hit 50+ users**

**Goal: $993/mo. Cost: $760/mo. Net: $233/mo.**

### Month 5-6: Growth

- [ ] **Enterprise tier** — White-label, on-prem, SLA
- [ ] **Voice marketplace** — Community voice packs
- [ ] **Multi-language** — Clone voice in any language
- [ ] **Hit 100+ users**

**Goal: $1,985/mo. Cost: $760/mo. Net: $1,225/mo.**

### Month 7-12: Platform

- [ ] **2+ MI300X** — Scale to 200+ users
- [ ] **Mobile SDK** — iOS/Android voice SDK
- [ ] **Phone integration** — Twilio SIP → AI co-host
- [ ] **Hit 500+ users**

**Goal: $9,925/mo. Cost: $3,600/mo. Net: $6,325/mo.**

---

## 6. Competitive Analysis

| Factor | Us | ElevenLabs | Play.ht | Coqui-AI | OpenAI TTS |
|--------|----|------------|---------|----------|------------|
| **Price** | **$19.85/mo** | $99/mo | $39/mo | Free (self-host) | $20/mo |
| **Voice clone** | ✅ Included | ✅ | ✅ | ✅ | ❌ |
| **Own hardware** | ✅ AMD | ❌ | ❌ | ✅ Any | ❌ |
| **Zero inference cost** | ✅ | ❌ | ❌ | ❌ (GPU cost) | ❌ |
| **Real-time streaming** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Open source codec** | ✅ MIT | ❌ | ❌ | ❌ (NC license) | ❌ |
| **LLM agnostic** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **OBS / Discord** | ✅ | ❌ | ❌ | ❌ | ❌ |

### Our advantage

Everyone else charges per-character or per-minute because they pay for cloud GPU. We don't. Our hardware is already paid for. That means:

- **Unlimited voice generation** — no metering, no surprises
- **$19.85 flat** — vs ElevenLabs $99 for 100 min
- **Enterprise can self-host** — no data leaves their network
- **AMD native** — no NVIDIA tax

---

## 7. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Voice quality not good enough | High | Iterate codec training; fallback to Piper during dev |
| Strix Halo can't handle 28 concurrent users | Medium | Lower target; optimize batching; add MI300X earlier |
| MI300X rent goes up | Low | Lock 6-month term on Vast; negotiate bulk |
| Open source competitors | Low | Moat is hardware + tight ZAYA integration |
| Users won't pay $19.85 | Medium | A/B test $9.99, $14.99, $19.85; add free tier |
| Regulation on voice cloning | Medium | Watermark all generated audio; opt-in consent flow |

---

## 8. The Asymmetric Bet

```
You already own the hardware that makes this viable.
Your marginal cost to serve 28 users is $10/mo.
At $19.85/user, that's $546/mo pure margin.
The MI300X only enters the picture at 37+ users.

No fundraising needed. No cloud credits. No NVIDIA.
Just AMD silicon already sitting on your desk.
```

---

*Built on 1bit.systems · AMD Strix Halo · Radeon 8060S · ZAYA-8B · Agnostic codec decoder*
