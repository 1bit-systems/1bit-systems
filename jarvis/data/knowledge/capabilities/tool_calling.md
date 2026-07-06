---
type: Capability
title: Tool Calling
description: LLM tool execution — calculator, Python code execution, file operations, and web search. All local.
tags: [tools, execution, python, calculator, files]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

JARVIS can execute tools on demand when the LLM generates tool calls. Tools are executed locally via Python.

## Available Tools

| Tool | Function | Description |
|------|----------|-------------|
| `calculator` | `_tool_calculator()` | Safe math expression evaluation |
| `python_exec` | `_tool_python_exec()` | Execute arbitrary Python code |
| `read_file` | `_tool_read_file()` | Read file contents (up to 4KB) |
| `list_dir` | `_tool_list_dir()` | List directory contents |
| `web_search` | `_tool_web_search()` | Web search query |

Enabled via `config.tools_enabled` list.

## Tool Result Flow

1. LLM generates a tool call with name + arguments JSON
2. `agent._execute_tool()` routes to the appropriate handler
3. Result (or error) returned as `ToolResult`
4. Injected back into LLM context for follow-up

## Security

- **Calculator**: Restricted to math operations only — no `__builtins__`
- **Python exec**: Full `builtins` available — caution advised
- **Read file**: Capped at 4000 chars
- **List dir**: Capped at 50 entries

## Citations

[1] [Tool Reference](/references/tool_reference.md)
[2] [Server Stack](/architecture/server_stack.md)
