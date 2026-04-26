# Why 1bit-agents?

**One-line answer**: 17 typed specialists in one Rust file. Not a framework, not a DAG engine, not a "multi-agent orchestrator" — just a registry of concrete ops the box needs to do, with typed inputs and typed outputs. Each one is tiny; the boundary between them is a `Name` enum.

## One file, one source of truth

Everything lives in [`crates/1bit-agents/src/lib.rs`](../../crates/1bit-agents/src/lib.rs). The enum and its `ALL` constant are the registry:

```rust
pub enum Name {
    Anvil, Carpenter, Cartograph, EchoEar, EchoMouth, Forge, Gateway, Herald,
    Librarian, Magistrate, Muse, Planner, Quartermaster, Scribe, Sentinel,
    Sommelier, Warden,
}

impl Name {
    pub const ALL: &'static [Name] = &[ /* 17 entries */ ];
}
```

One test asserts `Name::ALL.len() == 17`. Add a specialist, the test breaks until you list it. No runtime registration, no plugin loader, no YAML.

## The 17 specialists

Each name maps to one concrete job on strixhalo — not a "role" or a "persona":

- **anvil** — kernel builder (hipcc + rocprof wrappers)
- **carpenter** — scaffolds new Rust crates
- **cartograph** — repo-graph + dep analyzer
- **echo_ear** — whisper.cpp STT driver
- **echo_mouth** — kokoro TTS driver
- **forge** — requantizer frontend (safetensors → `.h1b`)
- **gateway** — OpenAI-compat router smoke tests
- **herald** — announcement + changelog drafter
- **librarian** — MEMORY.md / USER.md curator
- **magistrate** — PPL + parity gate enforcer
- **muse** — prompt refiner + sampler sweep
- **planner** — task decomposition for Claude / Hermes
- **quartermaster** — `halo install` packager
- **scribe** — session transcript writer
- **sentinel** — shadow-burnin watcher + alerter
- **sommelier** — model picker (size vs quality trade)
- **warden** — `halo doctor` health checks

Not 17 as a target — 17 as today's count of recurring ops tasks. Growth is fine; churn breaks the test, which is the point.

## TypedSpecialist: DSPy-style, serde-typed

Every specialist declares `Input` + `Output` structs with `#[derive(Serialize, Deserialize, JsonSchema)]`:

```rust
impl TypedSpecialist for AnvilSpecialist {
    type Input = AnvilInput;
    type Output = AnvilOutput;
    const NAME: Name = Name::Anvil;
    const DESCRIPTION: &'static str = "Build + profile a HIP kernel";
    async fn handle_typed(&self, req: AnvilInput) -> Result<AnvilOutput> { ... }
}
```

The `Typed<T>` adapter wraps any `TypedSpecialist` and implements the dyn-safe `Specialist` trait. One blanket impl drives serde + `schema_for!`. MCP `tools/list` falls out for free — Claude Code and DSPy render typed call UIs without round-tripping through the specialist first.

## Why not LangChain / CrewAI / AutoGen

- **Python in the runtime.** Rule A. Not negotiable.
- **`**kwargs` contracts.** No typed input/output. A bad field fails six levels deep in a stack trace the user sees as a 500.
- **Runtime agent graphs.** We need 17 one-shot tools, not a graph engine.

## Why not one god-class

Hermes' `AIAgent` is ~10,700 LOC in one file. That shape means:

- Every tool share globals; blast radius on a bug is the whole agent.
- Test fixtures need the full instantiation of the god-object.
- Inputs are untyped `Value` — schema lives in the prompt.

Our 17 specialists are each 100-400 LOC. Tests are per-specialist. Inputs are typed. Blast radius = one crate.

## How Claude / Hermes see them

Via [`1bit-mcp`](../../cpp/mcp/) — a tokio stdio JSON-RPC bridge. All 17 specialists surface as MCP tools, plus `skill_manage` and `memory_manage` (file-backed, see [Hermes Integration](./Hermes-Integration.md)). That's **19 MCP tools total**. Any MCP-speaking client drives the box.

## Pointers

- Registry: [`crates/1bit-agents/src/lib.rs`](../../crates/1bit-agents/src/lib.rs)
- Bridge: [`cpp/mcp/`](../../cpp/mcp/)
- External client setup: [Hermes Integration](./Hermes-Integration.md)
