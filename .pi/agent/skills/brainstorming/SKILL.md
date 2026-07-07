---
name: brainstorming
description: Structured ideation, strategic thinking, and creative problem-solving techniques. Use for open-ended problems, architecture decisions, feature planning, debugging strategies, and any situation needing creative options before committing to a path.
---

# Brainstorming Superpowers

## When to Use This Skill

Activate when the user asks any variation of:
- "How should I...?" / "What's the best way to...?"
- "What are my options for...?"
- "Help me think about..."
- "What could be causing X?"
- "Should I do A or B?"
- "Plan this feature / project / refactor"

## The 5-Phase Brainstorming Framework

### Phase 1: Problem Framing (1 min)
Before generating ideas, frame the problem:
- **Goal**: What does success look like?
- **Constraints**: What can't we change? (budget, time, technology, skills)
- **Stakeholders**: Who cares about the outcome?
- **Scope**: What's in/out of bounds?

### Phase 2: Divergent Thinking (generate)
Generate ideas across **three horizons**:

| Horizon | Description | Example |
|---------|-------------|---------|
| **Conventional** | Safe, well-known approaches | Use existing library X |
| **Creative** | Cross-domain or hybrid ideas | Combine approach A and B |
| **Blue Sky** | No constraints, ideal world | Build from scratch with new architecture |

**Techniques to use:**
- **SCAMPER**: Substitute, Combine, Adapt, Modify, Put to other use, Eliminate, Reverse
- **Inversion**: What would we do if we wanted to FAIL?
- **Analogy**: How would another domain solve this? (games, biology, finance, etc.)
- **First Principles**: Strip to fundamentals, rebuild from there
- **Random Stimulus**: Pick a random concept and force a connection

### Phase 3: Convergent Evaluation (narrow)
Evaluate each idea systematically:

| Criterion | Scale | Question |
|-----------|-------|----------|
| Feasibility | 1-5 | Can we reasonably do this? |
| Impact | 1-5 | How valuable is the outcome? |
| Effort | 1-5 | How much work? (1 = trivial, 5 = months) |
| Risk | Low/Med/High | What could go terribly wrong? |
| Learning | 1-5 | How much do we learn even if it fails? |

Eliminate clearly infeasible ideas. Keep everything else for synthesis.

### Phase 4: Synthesis (combine)
Mix and match ideas:
- **Hybrid**: Take parts of multiple ideas
- **Phased**: Do A first, then B
- **Contingency**: Plan A with Plan B as fallback
- **Research-first**: Validate the riskiest assumption first

### Phase 5: Action Plan (commit)
Output a concrete plan:

```
## Recommended Approach
[Clear description]

## Why
[Rationale based on evaluation]

## Implementation Sketch
[3-5 bullet steps]

## Key Risks & Mitigations
[What could go wrong + how to handle it]

## Next Actions
1. [Immediate concrete step]
2. [Next step]
3. [Step after that]
```

## Specialized Brainstorming Modes

### Debugging / Root Cause Analysis
```
Symptom → Hypotheses (5+ possible causes) → 
  Evidence gathering (for each) → 
  Likeliest cause → Fix → Verify
```

### Architecture Decision
```
Requirements → Options (3-5 architectures) → 
  Trade-off matrix → Recommendation → 
  Migration path (if changing from existing)
```

### Feature Planning
```
User need → Solution options → 
  Effort/impact ranking → MVP scope → 
  Roadmap (now/next/later)
```

### Risk Assessment
```
What could go wrong → Likelihood × Impact → 
  Top 5 risks → Mitigations → Monitoring
```

## Deeper Exploration

For complex brainstorming, delegate to the `brainstormer` sub-agent.
See the **subagent-deployment** skill for how to deploy sub-agents
(tool mechanics, agent selection) and **supervisor-orchestration**
for orchestration patterns (fan-out, pipeline, gated review).

## Quick Reference: Prompt Templates

```
"Let's brainstorm [topic]. Consider conventional, creative, and blue-sky approaches."

"What are 3-5 different approaches to [problem], with pros/cons for each?"

"Help me decide between [option A] and [option B]. Build a comparison matrix."

"What could be causing [bug symptom]? Generate hypotheses and how to test each."

"Design [feature/architecture]. Generate options, evaluate, and recommend."
```
