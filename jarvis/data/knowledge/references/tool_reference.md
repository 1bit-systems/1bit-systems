---
type: Reference
title: Tool Reference
description: Complete reference for all JARVIS tool calls — parameters, security, examples.
tags: [reference, tools, api, execution]
timestamp: 2026-07-06T00:00:00Z
---

# Tool Reference

## calculator

Safe math expression evaluator.

```json
{
  "name": "calculator",
  "arguments": "{\"expression\": \"2 + 2 * 3\"}"
}
```

**Security**: Restricted to `math` module functions plus `abs`, `pow`, `round`, `min`, `max`, `sum`, `len`. No `__builtins__`.

## python_exec

Execute arbitrary Python code.

```json
{
  "name": "python_exec",
  "arguments": "{\"code\": \"print('hello world')\"}"
}
```

**Security**: Full `builtins` available. Output captured from stdout.

## read_file

Read file contents.

```json
{
  "name": "read_file",
  "arguments": "{\"path\": \"/home/bcloud/jarvis/server/server.py\"}"
}
```

**Limits**: 4000 character cap on output.

## list_dir

List directory contents.

```json
{
  "name": "list_dir",
  "arguments": "{\"path\": \"/home/bcloud/jarvis\"}"
}
```

**Limits**: 50 entry cap on output.

## web_search

Search the web.

```json
{
  "name": "web_search",
  "arguments": "{\"query\": \"NPU benchmarks 2026\"}"
}
```

**Note**: Currently returns a stub message. Full integration with browser tool pending.

## Citations

[1] [Tool Calling](/capabilities/tool_calling.md)
[2] [API Reference](/references/api_reference.md)
