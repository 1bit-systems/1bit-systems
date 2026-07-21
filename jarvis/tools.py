"""Tool invocation with explicit permission gating.

Every tool declares a class: "safe" (read-only, always runs) or "sensitive"
(mutates local state — writes to the knowledge base, etc). Sensitive tools
only run if the caller's request explicitly opted in (`allow_write: true`).
Every call is appended to a local-only audit log — nothing here is ever
transmitted off-box, which is the privacy-protection half of this mechanism.
"""

import json
import re
import time

from jarvis.rag import _kb

AUDIT_LOG = _kb.root / "tools" / "audit.log"

TOOL_CALL_RE = re.compile(r"TOOL_CALL:\s*(\{)")

SYSTEM_PROMPT_TOOLS = """You have access to tools, but only these four —
never invent a tool name that isn't listed here:
- search_knowledge(query: str) — search the local knowledge base
- get_time() — current local date/time
- list_models() — list available local models
- add_note(title: str, content: str) — save a fact/note to the local knowledge base (requires write permission)

Only call a tool when the task genuinely needs one of these four things
(looking something up, saving a note, checking the time/model list). For
anything else — including math, reasoning, writing, or general knowledge —
answer directly yourself; do not call a tool.

To use one of the four tools, respond with ONLY a single line:
TOOL_CALL: {"name": "<tool_name>", "arguments": {...}}"""


def _audit(name, arguments, allowed, result_summary):
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "tool": name,
        "arguments": arguments,
        "allowed": allowed,
        "result_summary": result_summary,
    }
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _t_search_knowledge(args):
    q = args.get("query", "")
    results = _kb.search(q, max_results=5)
    return {"results": results}


def _t_get_time(_args):
    return {"time": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}


def _t_list_models(_args):
    from jarvis.routing import MODEL_ROUTING
    return {"models": list(MODEL_ROUTING.keys())}


def _t_add_note(args):
    title = args.get("title", "note")
    content = args.get("content", "")
    path = _kb.add_document(title, content)
    return {"saved": path}


TOOLS = {
    "search_knowledge": {"fn": _t_search_knowledge, "class": "safe"},
    "get_time": {"fn": _t_get_time, "class": "safe"},
    "list_models": {"fn": _t_list_models, "class": "safe"},
    "add_note": {"fn": _t_add_note, "class": "sensitive"},
}


def parse_tool_call(text):
    """Extract the first TOOL_CALL: {...} directive from model output, if
    present. Brace-matched rather than regex-captured to the closing brace —
    a model that emits multiple TOOL_CALL lines (small models do) would
    otherwise have its first call's JSON swallow everything up to the last
    line's closing brace, producing invalid JSON that silently fails to parse."""
    m = TOOL_CALL_RE.search(text or "")
    if not m:
        return None
    start = m.start(1)
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    call = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
                if "name" not in call:
                    return None
                call.setdefault("arguments", {})
                return call
    return None


def format_tool_followup(result, allowed):
    if not allowed:
        return (f"Tool call was denied ({result.get('error', 'not permitted')}). "
                f"Do not attempt another tool call — answer the original question "
                f"directly using your own knowledge instead.")
    return f"Tool result: {json.dumps(result)}\n\nNow answer the original question using this result."


def run_tool(name, arguments, allow_write=False):
    """Execute a tool under the permission gate. Always audited."""
    spec = TOOLS.get(name)
    if spec is None:
        _audit(name, arguments, False, "unknown tool")
        return {"error": f"unknown tool: {name}"}, False

    if spec["class"] == "sensitive" and not allow_write:
        _audit(name, arguments, False, "blocked: write permission not granted")
        return (
            {"error": f"tool '{name}' requires write permission (pass allow_write: true)"},
            False,
        )

    try:
        result = spec["fn"](arguments)
        _audit(name, arguments, True, str(result)[:200])
        return result, True
    except Exception as e:
        _audit(name, arguments, True, f"error: {e}")
        return {"error": str(e)}, True
