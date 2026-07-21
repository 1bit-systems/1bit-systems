"""Multi-step task planning across the local model roster.

A small, fast model (qwen3:0.6b) decomposes the request into an ordered list
of subtasks. Each subtask is then routed to whichever local model in
routing.MODEL_ROUTING actually fits it — reasoning subtasks to the larger
qwen3.6:35b, vision subtasks to qwen3vl, everything else to the fast default —
and, since this same engine can hold multiple of those models resident at
once, steps run without paying a cold-load penalty between them. Tool calls
inside a subtask go through jarvis.tools with the same permission gate as
single-turn chat. The final step synthesizes all subtask outputs into one
answer.
"""

import json
import re

from jarvis.routing import flm_chat, ollama_chat, resolve_model
from jarvis.tools import SYSTEM_PROMPT_TOOLS, format_tool_followup, parse_tool_call, run_tool

PLANNER_MODEL = "qwen3:0.6b"
SYNTH_MODEL = "qwen3.6:35b"

PLAN_PROMPT = """Break the following request into 2-5 concrete subtasks, ordered by
execution. Reply with ONLY a JSON list of strings, e.g.
["look up X", "compute Y from the result", "summarize for the user"].
If the request is a single simple step, reply with a one-item list.

Request: {request}"""

# subtask -> model_id, keyed by a cheap keyword sniff. Falls through to
# PLANNER_MODEL (fast default) if nothing matches.
SUBTASK_MODEL_HINTS = [
    (r"\b(image|photo|picture|diagram|screenshot)\b", "qwen3vl:4b"),
    (r"\b(reason|analy[sz]e|compare|plan|strategy|deep|complex)\b", SYNTH_MODEL),
]


def _chat_one(model_id, messages, max_tokens=256):
    route = resolve_model(model_id)
    if route.get("backend") == "gpu":
        r = ollama_chat(route["ollama_model"], messages, max_tokens=max_tokens)
        return r.get("response", "") if "error" not in r else f"[error: {r['error']}]"
    r = flm_chat(route.get("flm_model", model_id), messages, max_tokens=max_tokens)
    if "error" in r:
        return f"[error: {r['error']}]"
    return r.get("choices", [{}])[0].get("message", {}).get("content", "")


def _pick_model(subtask_text):
    for pattern, model_id in SUBTASK_MODEL_HINTS:
        if re.search(pattern, subtask_text, re.IGNORECASE):
            return model_id
    return PLANNER_MODEL


def make_plan(request):
    msgs = [{"role": "user", "content": PLAN_PROMPT.format(request=request)}]
    raw = _chat_one(PLANNER_MODEL, msgs, max_tokens=200)
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return [request]
    try:
        steps = json.loads(m.group(0))
        steps = [s for s in steps if isinstance(s, str) and s.strip()]
        return steps[:5] or [request]
    except json.JSONDecodeError:
        return [request]


def run_step(step, allow_write=False):
    model_id = _pick_model(step)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_TOOLS},
        {"role": "user", "content": step},
    ]
    reply = _chat_one(model_id, msgs, max_tokens=300)

    call = parse_tool_call(reply)
    if call:
        result, allowed = run_tool(call["name"], call.get("arguments", {}), allow_write=allow_write)
        msgs.append({"role": "assistant", "content": reply})
        msgs.append({"role": "user", "content": format_tool_followup(result, allowed)})
        reply = _chat_one(model_id, msgs, max_tokens=300)
        return {"step": step, "model": model_id, "tool_call": call, "tool_allowed": allowed, "output": reply}

    return {"step": step, "model": model_id, "tool_call": None, "output": reply}


def run_plan(request, allow_write=False):
    plan = make_plan(request)
    step_results = [run_step(s, allow_write=allow_write) for s in plan]

    # Always synthesize, even for a single-step plan: a subtask model's raw
    # reply can ignore or misreport a tool result it just received (small/fast
    # models do this), so skipping straight to step_results[0]["output"]
    # would surface that broken reply instead of grounding on the real tool
    # output the same way the multi-step path does.
    def _fmt(r):
        line = f"- {r['step']}: {r['output']}"
        tc = r.get("tool_call")
        if tc and tc.get("allowed"):
            # Ground synthesis in the actual tool output, not just the
            # subtask model's own paraphrase of it (which, for small/fast
            # models, sometimes ignores or misreports what the tool returned).
            line += f"\n  (tool {tc['name']} actually returned: {json.dumps(tc['result'])})"
        return line

    synthesis_input = "\n".join(_fmt(r) for r in step_results)
    synth_msgs = [{
        "role": "user",
        "content": f"Original request: {request}\n\nSubtask results:\n{synthesis_input}\n\n"
                   f"Give one concise final answer to the original request.",
    }]
    final = _chat_one(SYNTH_MODEL, synth_msgs, max_tokens=400)
    return {"plan": plan, "steps": step_results, "answer": final}
