# 1bit.systems — Complete Website Spec (v2, 2026-04-21)

**Target:** WordPress install (Gutenberg blocks), calm-technical voice, local-first product presentation with a built-in wiki.

**Voice:** calm technical. No memes. No 80s refs. No exclamation marks. (Per `project_brand_voice_split.md`; the meme voice stays on the GH README.)

**Purpose of this doc:** hand to a WP editor. Every section is fillable without asking the engineer follow-up questions. Where a value depends on a future decision, it is marked `[DECIDE]`; where a value is pulled from canonical project memory, it is marked `[CANON]` and is ready to paste.

---

## 0. Meta / ops

| Field | Value |
|---|---|
| Domain | `1bit.systems` |
| WWW canonical | `https://1bit.systems/` (301 `www.` → apex) |
| CMS | WordPress 6.x, Gutenberg block editor |
| Theme | minimal block theme (no page-builder plugins) `[DECIDE: build vs adapt]` |
| Cookie banner | **none** — we ship no first-party cookies, no analytics JS, no ad scripts |
| Analytics | server-side log counters only, none user-visible |
| Comments | disabled site-wide except on blog posts (moderated, no guest) |
| Search | native WP search + `Relevanssi` for wiki body search `[DECIDE]` |
| Editor roles | `author` (blog), `editor` (docs/wiki), `admin` (structure) |
| Deploy | WP on `[DECIDE: provider]`; static mirror by `wp2static` nightly to CF Pages `[DECIDE]` |
| Backups | nightly DB dump + `wp-content` rsync to Pi (`100.64.0.4`, ZFS) |
| SSL | Let's Encrypt via provider |

**Guard rails the editor must not violate:**

- No third-party script tags. No Google Fonts (self-host). No YouTube embed iframes (link out instead).
- No TypeScript anywhere. No Node build step. No React. Vanilla PHP templating only.
- No tracking pixels, no Facebook / Twitter / LinkedIn scripts, no Intercom, no Hotjar.
- No outbound telemetry of any kind from the site itself.

---

## 1. Information architecture

### 1.1 Primary menu (top nav)

1. Product
2. Models
3. Docs
4. Wiki
5. Bench
6. Roadmap
7. Blog

### 1.2 Footer menu (three columns)

- **Product:** Product, Install, Hardware, Download, Changelog, Compare
- **Learn:** Docs, Wiki, FAQ, Glossary, Tutorials, Blog
- **Project:** About, Contribute, License, Privacy, Brand, Contact

### 1.3 URL structure

| Pattern | Example |
|---|---|
| `/` | home |
| `/product/halo/` | product page (one per surface: halo, echo, helm) |
| `/install/` | install hub |
| `/install/cachyos/` | per-OS install |
| `/hardware/` | hardware targets |
| `/models/` | model zoo index |
| `/models/halo-1bit-2b/` | per-model |
| `/docs/` | docs hub |
| `/docs/<slug>/` | doc pages |
| `/wiki/` | wiki hub |
| `/wiki/<slug>/` | wiki pages |
| `/bench/` | bench index |
| `/bench/<run-id>/` | per-run |
| `/roadmap/` | public roadmap |
| `/changelog/` | release list |
| `/changelog/<version>/` | per-release notes |
| `/blog/` | blog index |
| `/blog/<yyyy>/<mm>/<slug>/` | post (date-slug hybrid) |
| `/glossary/` | glossary index |
| `/glossary/<term>/` | term page |
| `/faq/` | FAQ hub |
| `/compare/<other>/` | comparison pages (e.g. `/compare/petals/`) |
| `/privacy/` | privacy statement |
| `/license/` | license summary |
| `/about/` | about / team |

Permalink base: `/%postname%/`. CPT slugs follow the table above.

---

## 2. Custom Post Types (CPTs)

Each CPT listed with fields the editor fills in Gutenberg sidebar (ACF or native custom fields).

### 2.1 `Model`

Canonical record of a model we ship, stage, or reference.

| Field | Type | Example [CANON] |
|---|---|---|
| `display_name` | text | `Halo-1bit-2B` |
| `model_id` | text | `halo-1bit-2b` |
| `family` | taxonomy `model_family` | `BitNet-1.58` |
| `parameter_count` | number | `2_000_000_000` |
| `bits_per_weight` | decimal | `1.58` |
| `file_format` | select | `h1b` | `onnx` | `gguf` | `safetensors` |
| `file_size_bytes` | number | `1_100_000_000` |
| `context_length` | number | `2048` |
| `vocab_size` | number | `128256` |
| `architecture` | text | `LLaMA` / `BitNet` / `custom` |
| `hardware_target` | taxonomy `hardware_target` | `gfx1151` |
| `backend` | select | `hip` | `ort-cpu` | `ort-vitisai` | `cpu` |
| `license` | text | `MIT` |
| `upstream_url` | URL | HF repo link |
| `status` | select | `live` | `staged` | `blocked` | `planned` | `archived` |
| `baseline_tok_s_64` | decimal | `83` |
| `baseline_tok_s_1024` | decimal | `33` |
| `ppl_wikitext103` | decimal | `1.04` |
| `notes` | rich text | freeform |
| `download_url` | URL | direct or HF |
| `system_prompt_default` | text | optional |

### 2.2 `Service` (halo-* daemon)

One record per systemd unit the user installs.

| Field | Example [CANON] |
|---|---|
| `name` | `halo-bitnet` |
| `binary` | `bitnet_decode` |
| `port` | `8080` |
| `api` | `OpenAI-compatible` |
| `role` | `primary decode` |
| `language` | `C++20 / HIP` |
| `deps` | `librocm_cpp, hipBLAS-banned` |
| `systemd_unit_path` | `/etc/systemd/system/halo-bitnet.service` |
| `health_endpoint` | `GET /health` |
| `docs_link` | `/docs/services/halo-bitnet/` |

Pre-populated list:

```
halo-bitnet          :8080  bitnet_decode            primary decode (ternary)
halo-server          :8180  halo-server-real         Rust gateway
strix-server         :8180  1bit-server              gen-2 gateway (sibling)
strix-lemonade       :8200  1bit-lemonade            model gateway
halo-sd              :8081  sd-server                SDXL native HIP
halo-whisper         :8082  whisper-server           STT
halo-kokoro          [DECIDE] kokoro-server         TTS
halo-agent           [DECIDE] onebit-agents         agent runtime
strix-landing        :8190  1bit-landing             status page
```

### 2.3 `Benchmark`

One record per bench run (pulls JSON from `/home/bcloud/claude output/` on strixhalo).

| Field | Type |
|---|---|
| `run_id` | slug |
| `model` | relation → Model |
| `hardware` | relation → Hardware |
| `kernel_sha` | text |
| `metric_tok_s_64` | decimal |
| `metric_tok_s_1024` | decimal |
| `metric_ppl` | decimal |
| `bandwidth_util_pct` | decimal |
| `ran_on` | date |
| `raw_json` | URL or attachment |
| `notes` | rich text |

### 2.4 `Release`

Semantic release. One per tag in stampby/bong-water-water-bong repos.

| Field | Example |
|---|---|
| `version` | `0.8.2` |
| `released_on` | date |
| `headline` | text |
| `highlights` | rich text |
| `fixes` | list |
| `known_issues` | list |
| `upgrade_notes` | rich text |
| `signed_artifacts` | URL list |

### 2.5 `Doc`

Long-form documentation. Tiered:

| Field | Example |
|---|---|
| `tier` | `overview` | `howto` | `reference` | `explanation` (Diátaxis) |
| `audience` | `operator` | `integrator` | `contributor` |
| `prerequisites` | relation → Doc |
| `last_reviewed` | date |
| `status` | `draft` | `published` | `stale` |

### 2.6 `Glossary` term

| `term`, `aliases`, `short_def`, `long_def`, `see_also` (relation → Glossary), `canonical_source_url` |

### 2.7 `FAQ`

| `question`, `short_answer`, `long_answer`, `category` (tax), `related` (→ Doc / → Model) |

### 2.8 `Tutorial`

| `title`, `est_time_min`, `skill_level` (`beginner` / `intermediate` / `advanced`), `prereqs`, `body_steps` (ordered), `expected_output`, `troubleshooting` |

### 2.9 `Post` (blog)

Standard WP post. Author bio required. Author = `The Architect` by default (per `project_four_surface_strategy.md`).

---

## 3. Taxonomies

| Taxonomy | Terms (starter set) |
|---|---|
| `model_family` | BitNet-1.58, Sherry-1.25, TriLM-ternary, LLaMA-FP16, Wan-video, SDXL, Whisper, Kokoro, Medusa-heads |
| `hardware_target` | gfx1151 (Strix Halo iGPU), XDNA2 (Strix Halo NPU), Zen5 CPU, Apple MLX (reference), Battlemage (distill lane) |
| `language_tier` | C++20 runtime, Rust runtime, Python caller, Shell |
| `category` (blog) | announcements, engineering, research, ops, brand |
| `tag` (blog) | bitnet, sherry, xdna2, rocm, vitisai, onnx, sdxl, whisper, kokoro, helm, cachyos, packaging, perplexity, bench |
| `doc_tier` | overview, howto, reference, explanation |
| `faq_category` | install, models, performance, hardware, licensing, privacy, troubleshooting, voice, video, images |

---

## 4. Page specs (singular pages)

Each page below: purpose, audience, headings, per-section word count, CTAs, internal links, schema type.

### 4.1 Home `/`

- Purpose: 10-second pitch + entry point
- Audience: first-time visitor
- Schema: `WebSite` + `Organization`
- Blocks (top-to-bottom):
  1. **Hero** — 1 heading + 1 tagline + 1 CTA
     - H1: `Local AI, one box.`
     - Tagline: `A 1-bit inference stack that runs on AMD Ryzen AI Max+ 395. No cloud. No telemetry. Your closet.`
     - CTA primary: `Install` → `/install/`
     - CTA secondary: `Bench numbers` → `/bench/`
  2. **Live status strip** (server-side probe of `strixhalo/100.64.0.1:8080/health`)
     - If reachable: `83 tok/s, live` (green dot)
     - If not: `offline` (grey dot)
  3. **Three-up feature grid** — Chat, Voice, Images
  4. **Proof bar** — three facts with units, no marketing:
     - `1.58 bits/weight`
     - `92% of LPDDR5 bandwidth ceiling`
     - `0 bytes sent off-box`
  5. **Latest blog** — three cards
  6. **Footer**

Word count target: 180 words of prose excluding nav/footer.

### 4.2 Product hub `/product/`

Index card for three sub-products.

- `/product/halo/` — inference engine
- `/product/echo/` — voice loop (STT + TTS around halo)
- `/product/helm/` — desktop shell (voice-first, plugin API via MCP)

Each product page template:

| Section | Contents |
|---|---|
| Hero | H1 name + one-line definition |
| What it is | 2 paragraphs, no jargon |
| How it works | block diagram (SVG) + 3-step caption |
| What's inside | list of components (pulled from `Service` CPT) |
| Numbers | 3-4 hard metrics, linked to `Benchmark` records |
| Requirements | hardware + OS + memory |
| Status | one of: `shipping`, `beta`, `staged behind ship gate` |
| Related | links to Models, Install, Docs, FAQ |

### 4.3 Install `/install/`

Lists install paths. First visible: CachyOS (`project_fixes_20260421.md`, `install-strixhalo.sh` canonical).

Subpages:

- `/install/cachyos/` — bare-metal CachyOS (primary) `[CANON]`
- `/install/arch/` — upstream Arch variant
- `/install/fedora/` `[DECIDE: not yet]`
- `/install/debian/` `[DECIDE: not yet]`
- `/install/oob-iso/` — one-shot install ISO (gated, per `project_helm_scope_v2.md`)
- `/install/uninstall/` — revert

Per-OS page body structure:

1. Prerequisites (hardware, kernel, ROCm version, fish/bash, sudo posture)
2. One-shot command block (syntax-highlighted bash, copy button)
3. What that command does (bulleted list)
4. Verify (health checks, `halo status`, `halo doctor`)
5. First run (chat / voice / image)
6. Uninstall (link)
7. Troubleshoot (link to FAQ)

### 4.4 Hardware `/hardware/`

| Section | Content |
|---|---|
| Primary SKU | AMD Ryzen AI Max+ 395 (Strix Halo) [CANON] |
| iGPU | Radeon 8060S, gfx1151, wave32 WMMA [CANON] |
| NPU | XDNA2, gated until Linux STX-H EP ships [CANON] |
| Memory | 128 GB LPDDR5 unified iGPU/CPU [CANON] |
| Chassis | CAD $3000 mini-PC, silent-closet target [CANON] |
| ROCm version | 7.x native, hipBLAS runtime banned, Tensile only [CANON] |
| OS baseline | CachyOS 7.0.0-1, Btrfs + snapper + limine, fish shell [CANON] |
| Kernel mitigations | `ppfeaturemask=0xffffbfff`, `runpm=0`, SCLK pinned (per `project_optc_mitigation.md`) [CANON] |
| Tested configurations | table (populate as community reports) |
| Not supported | NVIDIA, Apple Silicon (MLX is reference only, not runtime) [CANON] |

### 4.5 Models `/models/`

Index of all `Model` CPT entries, filterable by `family`, `status`, `hardware_target`.

Per-model page template:

- Title (display name)
- One-line summary
- Numbers block (pulled from Benchmark CPT)
- Architecture diagram
- Quant recipe (how it was quantized, links to tools)
- File layout
- Example API call (code block)
- System prompt default (code block)
- License + upstream link
- Changelog of this model's versions

Seed entries to pre-create:

1. `halo-1bit-2b` — BitNet-1.58 2B, h1b, hip, status=live, 83/33 tok/s, PPL 1.04
2. `trilm-3.9b-int4-n4` — LLaMA-arch, onnx, ort-cpu, status=staged (see `project_trilm_export_20260421.md`)
3. `sherry-1.25` — retrain lane, status=blocked (see `project_sherry_root_cause.md`)
4. `sdxl-base-1.0` — native HIP, status=live
5. `whisper-base.en` — STT, status=live
6. `kokoro-onnxruntime` — TTS, status=staged
7. `wan-2.2-ti2v-5b` — video, status=planned (see `project_video_gen_pick.md`)

### 4.6 Bench `/bench/`

Index of all `Benchmark` records. Sortable table. CSV export link.

Per-run template:

- Metadata block (model, hardware, kernel sha, date)
- Metrics block (tok/s @ 64, @ 1024, PPL, bandwidth util)
- Command used (exact shell)
- Raw JSON download link
- Context (one paragraph, what this run was testing)

### 4.7 Roadmap `/roadmap/`

**Public face of the ship gate.** Per `project_ship_gate_npu.md`: no launch claims until XDNA2 unlocks; roadmap can describe intent without committing dates.

Sections:

- Now (shipping)
- Next (in progress)
- Later (researched, not built)
- Watching (upstream dependencies)
- Deferred (explicitly punted, with reason)

Populate from memory, but soften: no calendar dates on unshipped items. Use status words (`in progress`, `gated on AMD Linux STX-H EP`, `researched`, `deferred behind X`).

### 4.8 Compare `/compare/<other>/`

One page per named alternative. Starter set:

| Slug | Other | Position [CANON] |
|---|---|---|
| `petals` | Petals (BigScience) | P2P layer-sharded pipeline, likely dormant since Sept 2023; we're single-box. |
| `prismml` | PrismML fork | We borrowed Bonsai concept only, not their kernels (70× slower). |
| `llama-cpp` | llama.cpp | General-purpose, GGUF-centric, no BitNet-1.58 kernel on gfx1151. We are 1-bit-specialized. |
| `ollama` | Ollama | Convenient installer over llama.cpp; not a kernel project. |
| `papeg-ai` | Papeg.ai | Browser WebGPU, user hardware ranges widely. We own one SKU and ship for it. |
| `amd-ryzenai-sw` | AMD Ryzen AI 1.7 | Windows-first, UINT4 AWQ only, no BitNet. Linux STX-H path pending. |
| `lemonade-sdk` | Lemonade SDK | Upstream integration; we run a native Rust gateway on :8200 (`1bit-lemonade`). |
| `microsoft-bitnet` | Microsoft bitnet.cpp | Reference code + 2B-4T weights. We extend to gfx1151 + ship Sherry + NPU path. |

Each compare page: neutral tone, factual table, one-paragraph summary, link to the other project. No trash talk.

### 4.9 Privacy `/privacy/`

The strongest page on the site (lead with facts).

- No telemetry, zero outbound HTTP from runtime units.
- Only outbound: model downloads (user-initiated), archive rsync to user's own Pi.
- systemd `RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX`.
- No cookies. No analytics. No ad network. No third-party JS.
- What the site itself logs (server access log only, IP + UA + path + status; rotated 14 days).
- How to audit: `journalctl -u halo-*` on your own box.
- Data retention: zero, we don't store user data.
- Contact for privacy questions: `[DECIDE: contact mailbox]`.

### 4.10 Download `/download/`

| Artifact | Link | Sig |
|---|---|---|
| Install script (CachyOS) | `install-strixhalo.sh` | SHA256 + GPG `[DECIDE]` |
| `halo-1bit-2b.h1b` weights | HF repo `[DECIDE: HF account]` | — |
| `trilm-3.9b-int4-n4` ONNX bundle | HF repo `[DECIDE]` | — |
| Source tree (stampby mirror) | GH link | — |
| Source tree (bong-water-water-bong canonical) | GH link | — |

Checksums table rendered from manifest file `[DECIDE: manifest path]`.

### 4.11 Changelog `/changelog/`

Reverse-chronological list of Release CPT entries. Each entry:

- Version + date
- Headline
- Highlights (3-5 bullets)
- Fixes
- Known issues
- Upgrade notes
- Link to full release notes on GH

### 4.12 About `/about/`

- Who / what / why (3 paragraphs, neutral)
- Origin story (from `project_halo_vision.md`): efficiency play for CAD $3000 Strix Halo mini-PC, silent-closet BYOA inference, shared machine-specific kernels. We're the only public gfx1151 ternary stack.
- Principles: Rule A (no Python at runtime), Rule B (C++20 default), Rule C (no hipBLAS), Rule D (Rust 1.86), Rule E (ORT C++ + VitisAI for NPU). Each as its own paragraph with one-sentence justification.
- Contact: `[DECIDE: mailbox + matrix/discord optional]`
- Maintainers (by handle, not name): `bong-water-water-bong` (canonical commits). `stampby` archived.

### 4.13 Contact `/contact/`

Short form. No third-party form provider.

- GitHub issues (primary)
- Email `[DECIDE]`
- Security: `[DECIDE: security@ or /privacy/#contact]`
- Press: `[DECIDE: press@ or n/a]`

### 4.14 License `/license/`

Matrix of per-repo licenses:

| Repo | License | Notes |
|---|---|---|
| halo-ai-core | MIT `[VERIFY]` | |
| strix-ai-rs / halo-workspace | MIT | per `crates/*/Cargo.toml` |
| halo-kokoro | MIT | upstream fork |
| halo-mcp | `[DECIDE]` | |
| TriLM-3.9B-Unpacked weights | Apache-2.0 | SpectraSuite upstream |
| Sherry retrain weights | `[DECIDE]` | |
| sd.cpp-derived | MIT | |

### 4.15 Brand `/brand/`

- Logo (SVG, light + dark)
- Wordmark
- Primary palette (hex)
- Typography (system stack, no Google Fonts)
- Voice guide (link to `project_brand_voice_split.md` summary):
  - 1bit.systems: calm technical, no memes
  - GH README: 80s + Matrix refs permitted
  - WP blog: the-architect, personal
  - All surfaces: no TypeScript, no emojis in body prose (ok in UI microcopy)
- Spelling conventions: `1-bit` (hyphen) in prose; `1bit` (no hyphen) in product names
- Product name cases: `Halo`, `Echo`, `Helm`, `halo-*` for services

### 4.16 Contribute `/contribute/`

- How to file a bug (link to GH issue template)
- How to submit a PR (conventional commits; `bong` remote; `why` line required)
- Test gate (`cargo test --workspace --release`; ~90 tests baseline)
- Code of conduct `[DECIDE]`
- Roadmap input (where in repo to propose)

### 4.17 Community `/community/`

- GH discussions link
- `[DECIDE: Matrix / Discord]` — memory notes: halo-mcp has Discord integration; also `project_helm_scope_v2.md` mentions Discord mode. Decide one public channel, pin it here.
- Office hours `[DECIDE]`
- Reddit: **omit until NPU ship gate** (per `project_ship_gate_npu.md`); re-enable Reddit section after unlock

### 4.18 Docs hub `/docs/`

Index of `Doc` CPT entries, grouped by tier (Diátaxis):

- **Overview** — what is halo, what is echo, what is helm
- **How-to** — install, run, configure, integrate
- **Reference** — API (`/v1/*`, `/v2/*`), config files, CLI flags
- **Explanation** — why BitNet-1.58, why gfx1151, why Rust, why no Python runtime

### 4.19 FAQ hub `/faq/`

See §8 for full 30-question seed bank.

### 4.20 Wiki hub `/wiki/`

See §7 for full wiki structure.

---

## 5. Per-page content templates

### 5.1 Doc page body

```
# {title}

> Audience: {audience}. Prerequisites: {prereqs}. Time: {est_time}.

## What you'll learn
- bullet 1
- bullet 2
- bullet 3

## Background
(2-3 paragraphs)

## Steps
1. ...
2. ...
3. ...

## Verify
```code block```

## Troubleshoot
- problem → fix
- problem → fix

## See also
- link
- link
```

### 5.2 Tutorial page body

```
# {title}

| Skill level | {beginner|intermediate|advanced} |
| Estimated time | {n} min |
| What you'll build | one sentence |

## Before you start
- requirement
- requirement

## Step 1 — {action}
(prose + command block + expected output)

## Step 2 — ...

## Check your work
- expected final state

## Next
- link to next tutorial / doc
```

### 5.3 Model page body

```
# {display_name}

One-line summary.

## Numbers
| Metric | Value |
|---|---|
| Parameters | {n} |
| Bits/weight | {n} |
| tok/s @ 64 ctx | {n} |
| tok/s @ 1024 ctx | {n} |
| PPL (wikitext-103) | {n} |

## Architecture
(diagram + paragraph)

## Quantization recipe
(code block showing how it was produced)

## Files
(table of artifact files + sizes + checksums)

## Example
```api call```

## License
{license} — {upstream_url}

## History
(changelog per version)
```

### 5.4 Compare page body

```
# {halo} vs {other}

Short positioning paragraph (neutral).

## Feature matrix
| | halo | {other} |
|---|---|---|
| ... | ... | ... |

## Where {other} is better
- bullet
- bullet

## Where halo is better
- bullet
- bullet

## Interop
(how they can coexist, if at all)
```

---

## 6. Blog structure

- Category-ordered feed at `/blog/`
- Sidebar: recent posts, category cloud, tag cloud, RSS
- Per-post:
  - Featured image (16:9, JPEG, 1600 × 900, <150 KB)
  - Author bio block (The Architect default)
  - Est. read time (auto)
  - `updated_on` visible below `published_on` when they differ
  - Share buttons: none (no third-party scripts)
  - Comments: off by default; opt-in per post
- RSS: `/feed/`, full body (not excerpt)
- Starter post seeds (8):
  1. Why BitNet-1.58 on Strix Halo (tech / explainer)
  2. RoPE convention bug: from PPL 524 to 1.04 (engineering)
  3. Split-KV Flash-Decoding: 6.78× at L=2048 (engineering)
  4. Sherry 1.25-bit: what worked, what didn't (research)
  5. Why no Python at runtime (principles)
  6. The halo-pkg story (ops)
  7. Building for one box (brand, the-architect voice)
  8. What the NPU ship gate buys us (roadmap)

---

## 7. Wiki structure

Editorial, deep-dive reference. Longer than docs, less structured than reference. Written by The Architect.

### Categories

- Architecture
- Kernels (HIP)
- Kernels (AIE / NPU future)
- Runtime (Rust)
- Models
- Quantization
- Tokenization
- Sampling
- Serving (HTTP + gateway)
- Voice (STT + TTS + streaming)
- Images (SDXL)
- Video (Wan 2.2)
- Packaging (halo-pkg, CachyOS ISO)
- Operations (systemd, monitoring, journalctl)
- Networking (Headscale, mesh)
- Archive (Pi, rsync, KCS)
- Research notes
- Glossary (overflow from `Glossary` CPT)

### Seed wiki pages (60; fill over time)

**Architecture**

1. `Halo stack overview` — one-page diagram of the whole
2. `Layering: kernels → router → server → gateway`
3. `Rule A through Rule E explained`
4. `Why Rust above C++20`
5. `Two-tree story: halo-ai-core (gen-1) vs strix-ai-rs (gen-2)`

**Kernels (HIP)**

6. `Ternary GEMV on gfx1151` (92% bandwidth ceiling)
7. `Wave32 WMMA for wmma tiles`
8. `RMSNorm + SiLU + RoPE fusion`
9. `Flash-Decoding with split-KV`
10. `KV cache layout (h1b)`
11. `RoPE convention: interleaved vs split-half` (the bug that fixed PPL)
12. `sd.cpp native-HIP port` (SDXL path)

**Kernels (AIE / NPU future)**

13. `VitisAI EP 101`
14. `MatMulNBits N=4 vs N=2` (why N=2 is blocked)
15. `AIE kernel authoring via Peano` (reference)
16. `IRON as compile-time tool, not runtime` (clarification)

**Runtime (Rust)**

17. `Workspace layout (20 crates)`
18. `lemond and the backend dispatcher`
19. `1bit-server HTTP surface (v1 + v2 + /ppl + /metrics)`
20. `1bit-onnx: loading OGA Model Builder output` (new, this session)
21. `Sampler modes and CPU pipelining`
22. `Tokenizer (.htok) vs HF tokenizer.json`
23. `Systemd units for halo-* services`

**Models**

24. `halo-1bit-2b under the hood`
25. `TriLM 3.9B and the ternary-to-int4 path`
26. `Sherry 1.25-bit: plan, retrain lane, validator`
27. `BitNet v2 / Hadamard outlook`
28. `MedusaBitNet: speculative heads (deferred)`
29. `SDXL on gfx1151` (kernel inventory)
30. `Wan 2.2 TI2V-5B port plan`

**Quantization**

31. `BitNet training vs post-training quant`
32. `RTN vs k-quant in MatMulNBits`
33. `Activation sparsity: what we measured` (79.91%)
34. `AWQ via AMD Quark (caller-side only)`
35. `GGUF: why we don't ship it`

**Tokenization**

36. `.htok format`
37. `HF tokenizers interop in `1bit-onnx`
38. `Special tokens and EOS in BitNet vs TriLM`

**Sampling**

39. `Argmax / temperature / top-k / top-p`
40. `Pipelined CPU sampler` (CpuLane)

**Serving**

41. `/v1/chat/completions vs /v2/*`
42. `Gateway dispatch (1bit-lemonade, lemond)`
43. `Strix-landing status page on :8190`

**Voice**

44. `Echo loop: whisper → halo-server → kokoro`
45. `Whisper streaming (sliding-window)`
46. `Kokoro TTS integration`
47. `Mouth-to-ear latency budget (1-3 s today)`

**Images + video**

48. `sd.cpp fork policy`
49. `SDXL prompt → pixel pipeline`
50. `Wan 2.2 memory budget`

**Packaging + operations**

51. `halo-pkg manifest and install flow`
52. `CachyOS ISO build plan`
53. `journalctl recipes for halo-*`
54. `Monitoring: /metrics and status page`
55. `OPTC CRTC hang mitigation (ppfeaturemask + runpm)`

**Networking**

56. `Headscale tailnet layout (strixhalo 100.64.0.1, sliger 100.64.0.2, pi 100.64.0.4)`

**Archive**

57. `Pi as canonical archive (ZFS + rsync + KCS)`
58. `Archive retention policy` (delete local >7 days)

**Research notes**

59. `Upstream watch list`
60. `Quarterly ecosystem scan template`

Each wiki page: 800-2500 words, liberal use of diagrams, crosslinks.

---

## 8. FAQ bank (30 seed questions)

Grouped by category. Editor fills short + long answer.

**Install / setup**

1. What hardware do I need?
2. Does it work on Intel / Nvidia?
3. Does it work on Apple Silicon?
4. Which Linux distro should I use?
5. Can I run without the NPU?
6. How much disk do I need?

**Models**

7. What models do you ship?
8. Can I use my own GGUF model?
9. Can I quantize my own model?
10. Why 1.58 bits?
11. What about llama.cpp?

**Performance**

12. How fast is it?
13. Why is decode slower at long context?
14. What bandwidth ceiling do you hit?
15. Can I overclock?

**Voice / images / video**

16. How good is the voice latency?
17. Which STT model ships?
18. Which TTS model ships?
19. Does image generation work?
20. When does video arrive?

**Privacy / security**

21. Does the box phone home?
22. Can I run fully offline?
23. How do I audit outbound traffic?

**Licensing / sources**

24. What license is the code?
25. What licenses are the weights?
26. Can I ship this commercially?

**Troubleshooting**

27. The screen freezes — what happened?
28. Why did my system reboot?
29. Why is TriLM on CPU and not NPU?
30. How do I revert a kernel?

Populate short answers from memory; long answers expand into Docs/Wiki as needed.

---

## 9. Glossary (50+ terms, seed)

Format per term: `term`, `aliases`, `short_def`, `long_def`, `see_also`.

**Seed list:**

| Term | Short def |
|---|---|
| 1-bit | Shorthand for BitNet-1.58 (ternary) and Sherry-1.25 lineage |
| BitNet-1.58 | Microsoft ternary transformer; weights ∈ {−1, 0, +1} |
| Sherry-1.25 | 1.25 bits/weight via 3:4 sparsity; 2 bits zero-pos + 3 bits signs |
| TriLM | SpectraSuite ternary-trained LLaMA (3.9B FP16 unpacked reference) |
| gfx1151 | AMD Radeon 8060S iGPU architecture in Strix Halo |
| XDNA 2 | AMD NPU in Strix / Strix Halo |
| WMMA | Wave Matrix Multiply-Accumulate intrinsic |
| ROCm | AMD GPU compute stack |
| hipBLAS | AMD BLAS library — banned in our runtime |
| Tensile | ROCm native matmul kernel library (what we use) |
| MatMulNBits | ONNX custom op for n-bit quantized matmul (`com.microsoft`) |
| VitisAI EP | ONNX Runtime Execution Provider for AMD AIE (XDNA 2) |
| Peano | Xilinx LLVM-AIE toolchain for AIE kernel authoring |
| OGA | ONNX Runtime GenAI Model Builder |
| OGA int4 | OGA's MatMulNBits N=4 export target |
| Ternary GEMV | matrix-vector multiply with ternary weights |
| PPL | Perplexity (lower is better) |
| RoPE | Rotary Position Embedding |
| Flash-Decoding | Split-KV attention kernel for long-context decode |
| Split-KV | Parallelization scheme across the KV dimension |
| KV cache | Per-layer stored keys/values across decode steps |
| h1b | halo-1bit weights file format |
| htok | halo tokenizer file format |
| OptC CRTC | AMD display pipeline controller; source of 2026-04-21 hangs |
| Rule A | No Python at runtime |
| Rule B | C++20 by default for new components |
| Rule C | hipBLAS runtime-banned |
| Rule D | Rust 1.86, edition 2024 |
| Rule E | ORT C++ + VitisAI EP for NPU |
| halo-pkg | 1bit-systems package manager |
| Helm | 1bit-systems desktop shell (voice-first) |
| Echo | 1bit-systems voice loop (STT + TTS + orchestration) |
| Halo | 1bit-systems inference engine |
| Strix Halo | AMD Ryzen AI Max+ 395 SKU family |
| LPDDR5 | unified memory type on Strix Halo |
| Headscale | self-hosted Tailscale control plane; our mesh |
| Pi archive | Raspberry Pi at 100.64.0.4 with 3.6 TB ZFS |
| BitNet a4.8 | activation-4/weight-8 staged recipe (superseded by BitNet v2) |
| BitNet v2 | W1.58A4 native via Hadamard transform |
| Activation sparsity | fraction of near-zero activations skipped in compute |
| KVTQ | KV cache ternary quantization |
| Bonsai model | PrismML architecture; we borrow the concept only |
| MCP | Model Context Protocol (halo-mcp Rust server) |
| MCP-out | outgoing MCP client in onebit-agents |
| 1bit-server | canonical gen-2 server binary |
| halo-server-real | gen-1 server binary |
| lemonade | OpenAI-compat model gateway (ours: 1bit-lemonade) |
| Open WebUI | external Python client, allowed as /v2 caller, sunsets at Helm v0.3 |
| CachyOS | our baseline distro |
| BYOA | Bring Your Own AI (project framing) |

Add per-term long_def + canonical_source_url on write.

---

## 10. SEO + schema

- Robots: `index, follow` site-wide; `noindex` for `/wiki/*?draft=true`.
- Sitemap: `/sitemap.xml`, include all CPT archives.
- OpenGraph + Twitter card meta on every public page.
- JSON-LD:
  - `WebSite` with `SearchAction` pointing at `/?s={query}`
  - `Organization` on `/about/`
  - `SoftwareApplication` on `/product/halo/` with fields `operatingSystem: "Linux"`, `applicationCategory: "DeveloperApplication"`
  - `FAQPage` on `/faq/*` pages (question + answer pairs)
  - `TechArticle` on `/wiki/*` and `/docs/*`
  - `BlogPosting` on `/blog/*`
  - `Review`-free (no marketing reviews)
- Page metadata fields per post/page: meta description (≤160 chars), OG image (1200 × 630, JPEG, <200 KB), canonical URL, hreflang `en` only initially.
- Internal linking: every page links to at least 2 siblings; orphan pages go to the backlog.

---

## 11. Media / assets

- Logo: SVG, light + dark. Filename: `1bit-logo-light.svg`, `1bit-logo-dark.svg`.
- Favicons: 16, 32, 180 (Apple touch), 512 (maskable). Generated from logo.
- Diagrams: SVG only, editable source in `/wp-content/uploads/diagrams/src/*.drawio` or equivalent.
- Product screenshots: 1600 × 900 JPEG, <200 KB, accessibility text required.
- Videos: self-hosted if <10 MB, otherwise link out.
- Code blocks: `prism.js` or built-in Gutenberg Code block. Languages to register: `bash`, `rust`, `cpp`, `hip`, `python`, `json`, `toml`, `onnx` (fall back to `text`).
- No carousels, no autoplay, no video backgrounds.

---

## 12. Accessibility

- WCAG 2.2 AA target.
- Color contrast ≥ 4.5:1 for body text.
- All interactive elements keyboard-navigable.
- No color-only information (status dots pair with text).
- Alt text mandatory on Featured Image and inline images.
- No motion on scroll.
- Font size 16 px min body; line-height 1.6.

---

## 13. Admin setup (for WP editor hand-off)

### Plugins to install

| Plugin | Purpose | Reason |
|---|---|---|
| `advanced-custom-fields` (free) | CPT field editor | simpler than block attributes for editors |
| `relevanssi` | Search | full-body wiki search |
| `wp-sitemap-page` (or native) | Sitemap | SEO |
| `disable-comments` | Comments | off-by-default except blog |
| `health-check` | Diagnostics | admin only |
| `wp-2fa` | 2FA on admin | `[DECIDE]` |

### Plugins to refuse

- Any analytics plugin (Jetpack, MonsterInsights, GA for WordPress)
- Any ad plugin
- Any "AI SEO" plugin that sends post content to third-party APIs
- Any cache plugin that injects `<script>` from a CDN

### Writing workflow

- Drafts in `draft` status, editor review required before `publish`
- Scheduled publish OK
- `last_reviewed` date on all Docs; nightly cron flags docs older than 90 days

### Hand-off checklist

- [ ] All CPTs registered
- [ ] All taxonomies registered
- [ ] Primary + footer menus wired
- [ ] Home page built with live-status block
- [ ] 4.3 – 4.17 pages created as stubs
- [ ] 6 blog posts seeded
- [ ] 60 wiki pages seeded (titles only, body later)
- [ ] 30 FAQ entries seeded
- [ ] 50 glossary terms seeded with short_def
- [ ] Seed `Model` records (7)
- [ ] Seed `Service` records (9)
- [ ] Seed `Benchmark` records (pull from `/home/bcloud/claude output/*.json`)
- [ ] `/privacy/` page live and linkable
- [ ] Robots.txt and sitemap.xml live
- [ ] 404 page styled
- [ ] Search page styled

---

## 14. What goes on the site only AFTER the NPU ship gate lifts

Per `project_ship_gate_npu.md`: no public launch claims until XDNA 2 unlocks on Strix Halo Linux. Until then, keep the site in soft-launch mode:

- Roadmap page live (status words, no dates)
- FAQ question about NPU answered honestly ("staged; gated on AMD Linux STX-H EP")
- No press push, no Reddit, no HN
- No "order now" or "install now" CTAs on hero; use "read the bench" instead
- Model page for TriLM int4 can publish with status `staged` + the artifact path (CPU inference works, NPU does not)

After unlock:

- Hero CTA flips to `Install`
- Roadmap `Now` and `Next` columns shift
- Blog post: `NPU unlock day`
- Reddit window re-opens under `bong-water-water-bong`

---

## 15. Outstanding decisions (inventory)

Pull these from `[DECIDE]` tags above for a single meeting:

- WP hosting provider
- Theme: build vs adapt
- Static mirror on / off
- Contact mailbox address
- Security contact
- Press contact
- Matrix vs Discord for community
- Office hours cadence
- HF account for weight hosting
- Code signing / GPG key
- License of halo-mcp repo
- 2FA plugin choice on WP
- Whether to translate (hreflang) any content past English
