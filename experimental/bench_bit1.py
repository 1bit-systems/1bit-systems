#!/usr/bin/env python3
"""Comprehensive benchmarks for the bit1-mlx package."""

import time
import tracemalloc
import sys
import gc
import json

BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"

def header(s):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {s}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def bench(name, iterations=1000):
    """Decorator-like helper for benchmarking."""
    def decorator(fn):
        # Warmup
        for _ in range(10):
            fn()
        gc.collect()

        # Timing
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed
        us_per_op = (elapsed / iterations) * 1_000_000

        print(f"  {YELLOW}{name:<45}{RESET} {ops_per_sec:>12.1f} ops/s  ({us_per_op:>6.1f} µs/op)")
        return fn
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# 1. IMPORT TIME & MEMORY
# ═══════════════════════════════════════════════════════════════════════════════
header("1. Import Time & Memory Footprint")

# Measure cold import time
tracemalloc.start()
start = time.perf_counter()

import bit1_mlx.model_profile
import bit1_mlx.model_aliases
import bit1_mlx.api.models
import bit1_mlx.api.anthropic_adapter
import bit1_mlx.api.anthropic_models
import bit1_mlx.api.tool_calling
import bit1_mlx.api.utils
import bit1_mlx.api.guided
import bit1_mlx.api.responses_adapter
import bit1_mlx.api.responses_models
import bit1_mlx.tool_parsers
import bit1_mlx.middleware.auth
import bit1_mlx.middleware.body_size
import bit1_mlx.middleware.body_depth
import bit1_mlx.middleware.exception_handlers
import bit1_mlx.engine.base
import bit1_mlx.cache.protocol
import bit1_mlx.config
import bit1_mlx.output_router
import bit1_mlx.output_router_harmony
import bit1_mlx.output_collector

elapsed = time.perf_counter() - start
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"  Cold import time:        {elapsed*1000:.1f} ms")
print(f"  Peak memory (traced):    {peak / 1024 / 1024:.1f} MB")
print(f"  Current memory (traced): {current / 1024 / 1024:.1f} MB")

# Measure process RSS after import
import os
if hasattr(os, "getpid"):
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        print(f"  Process RSS after import: {rss / 1024 / 1024:.1f} MB")
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL ALIASES
# ═══════════════════════════════════════════════════════════════════════════════
header("2. Model Alias Resolution")

from bit1_mlx.model_aliases import resolve_model, list_aliases, resolve_profile, suggest_similar

aliases = list_aliases()
alias_names = list(aliases.keys())
print(f"  Total aliases loaded: {len(aliases)}")

# Batch resolve — how fast can we resolve all 163 aliases
t0 = time.perf_counter()
for name in alias_names:
    resolve_model(name)
t1 = time.perf_counter()
print(f"  Batch resolve ({len(alias_names)} aliases): {(t1-t0)*1000:.1f} ms  ({(t1-t0)/len(alias_names)*1e6:.1f} µs/alias)")

# Fuzzy suggestion
t0 = time.perf_counter()
for name in alias_names[:50]:
    suggest_similar(name, n=3)
t1 = time.perf_counter()
print(f"  Fuzzy suggestion (50 calls): {(t1-t0)*1000:.1f} ms  ({(t1-t0)/50*1e6:.1f} µs/call)")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PYTHON OBJECT CREATION
# ═══════════════════════════════════════════════════════════════════════════════
header("3. Object Construction Throughput")

from bit1_mlx.model_profile import ModelProfile
from bit1_mlx.api.models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk,
    ChatCompletionChunkChoice, ChatCompletionChunkDelta,
    CompletionRequest, AssistantMessage, Message, ToolCall, FunctionCall, Usage
)
from bit1_mlx.api.anthropic_models import AnthropicRequest
from bit1_mlx.engine.base import GenerationOutput
from bit1_mlx.cache.protocol import SaveOutcome, LoadResult

# ModelProfile
@bench("ModelProfile()", iterations=50000)
def b():
    ModelProfile(hf_path="test/model", tool_call_parser="hermes")

# ChatCompletionRequest — simple
@bench("ChatCompletionRequest (1 msg)", iterations=20000)
def b():
    ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello, how are you?"}]
    )

# ChatCompletionRequest — with tools
@bench("ChatCompletionRequest (1 msg + 3 tools)", iterations=10000)
def b():
    ChatCompletionRequest(
        model="test-model",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What's the weather in Paris?"}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send an email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"}
                        },
                        "required": ["to", "subject", "body"]
                    }
                }
            }
        ]
    )

# ChatCompletionRequest — long context (100 messages)
messages_100 = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Message number {i}"} for i in range(100)]

@bench("ChatCompletionRequest (100 msgs)", iterations=5000)
def b():
    ChatCompletionRequest(model="test-model", messages=messages_100)

# GenerationOutput
@bench("GenerationOutput()", iterations=50000)
def b():
    GenerationOutput(text="Hello world", prompt_tokens=10, completion_tokens=3)

# Usage
@bench("Usage()", iterations=50000)
def b():
    Usage(prompt_tokens=50, completion_tokens=150, total_tokens=200)

# AnthropicRequest
@bench("AnthropicRequest()", iterations=20000)
def b():
    AnthropicRequest(
        model="test-model",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}]
    )

# SaveOutcome
@bench("SaveOutcome()", iterations=100000)
def b():
    SaveOutcome(outcome="committed")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANTHROPIC ↔ OPENAI CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════
header("4. Anthropic ←→ OpenAI Conversion")

from bit1_mlx.api.anthropic_adapter import anthropic_to_openai, openai_to_anthropic

anthropic_req = AnthropicRequest(
    model="claude-3-opus",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    system="You are a helpful physics tutor."
)

from bit1_mlx.api.models import ChatCompletionResponse, ChatCompletionChoice, AssistantMessage

openai_resp = ChatCompletionResponse(
    id="chatcmpl-123",
    object="chat.completion",
    created=1234567890,
    model="gpt-4",
    choices=[
        ChatCompletionChoice(
            index=0,
            message=AssistantMessage(role="assistant", content="Quantum computing uses qubits..."),
            finish_reason="stop"
        )
    ],
    usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
)

@bench("Anthropic → OpenAI", iterations=20000)
def b():
    anthropic_to_openai(anthropic_req)

@bench("OpenAI → Anthropic", iterations=20000)
def b():
    openai_to_anthropic(openai_resp, model="claude-3-opus")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TOOL CALLING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
header("5. Tool Calling Helpers")

from bit1_mlx.api.tool_calling import (
    build_json_system_prompt, convert_tools_for_template,
    extract_json_schema_for_guided, parse_tool_calls
)

sample_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

sample_rf = {"type": "json_object"}

@bench("build_json_system_prompt", iterations=20000)
def b():
    build_json_system_prompt(sample_rf)

@bench("convert_tools_for_template", iterations=20000)
def b():
    convert_tools_for_template(sample_tools)

sample_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "get_weather",
        "schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}

@bench("extract_json_schema_for_guided", iterations=20000)
def b():
    extract_json_schema_for_guided(sample_schema)

parse_output = '<tool_call><invoke name="get_weather"><parameter name="location">Paris</parameter></invoke></tool_call>'

@bench("parse_tool_calls (auto-detect)", iterations=20000)
def b():
    parse_tool_calls(parse_output)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TOOL PARSER THROUGHPUT (Model-Native Parsing)
# ═══════════════════════════════════════════════════════════════════════════════
header("6. Tool Parser Throughput (Model-Native)")

from bit1_mlx.tool_parsers import ToolParserManager, ExtractedToolCallInformation

tpm = ToolParserManager()
tool_parsers = tpm.list_registered()

# We need a tokenizer for real parsers — use a simple mock
class MockTokenizer:
    def decode(self, tokens):
        if isinstance(tokens, list):
            return " ".join([str(t) for t in tokens])
        return str(tokens)
    def get_vocab(self):
        return {}
    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, list):
            return [str(i) for i in ids]
        return str(ids)
    @property
    def chat_template(self):
        return None
    @property
    def all_special_tokens(self):
        return []
    @property
    def all_special_ids(self):
        return []

mock_tokenizer = MockTokenizer()

# Parse throughput for key parsers
sample_outputs = {
    "hermes": '<tool_call><invoke name="get_weather"><parameter name="location">Paris</parameter><parameter name="units">celsius</parameter></invoke></tool_call>',
    "mistral": "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]",
    "llama": '<function=get_weather>{"location": "Paris", "units": "celsius"}</function>',
    "qwen": '<tool_calls>\n<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>\n</tool_calls>',
    "deepseek": '�get_weather\n{"location": "Paris", "units": "celsius"}',
    "gemma4": '<function_call>\n<function_name>get_weather</function_name>\n<parameters>{"location": "Paris"}</parameters>\n</function_call>',
    "harmony": '{"type": "function", "name": "get_weather", "arguments": {"location": "Paris"}}',
}

for parser_name, sample_output in sample_outputs.items():
    try:
        parser_cls = tpm.get_tool_parser(parser_name)
        parser = parser_cls(mock_tokenizer)

        # Warmup
        for _ in range(10):
            parser.extract_tool_calls(sample_output)

        # Benchmark
        t0 = time.perf_counter()
        n = 10000
        for _ in range(n):
            parser.extract_tool_calls(sample_output)
        elapsed = time.perf_counter() - t0
        ops = n / elapsed
        us = (elapsed / n) * 1_000_000
        print(f"  {YELLOW}{parser_name:<20}{RESET} {ops:>12.1f} ops/s  ({us:>6.1f} µs/parse)")
    except Exception as e:
        print(f"  {YELLOW}{parser_name:<20}{RESET} ERROR: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RESPONSE SERIALIZATION (JSON output)
# ═══════════════════════════════════════════════════════════════════════════════
header("7. Response Serialization")

from bit1_mlx.api.models import ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChunkDelta

# Build a streaming chunk
chunk = ChatCompletionChunk(
    id="chatcmpl-123",
    object="chat.completion.chunk",
    created=1234567890,
    model="test-model",
    choices=[
        ChatCompletionChunkChoice(
            index=0,
            delta=ChatCompletionChunkDelta(
                content="Hello",
                role="assistant"
            ),
            finish_reason=None
        )
    ]
)

@bench("ChatCompletionChunk → JSON", iterations=50000)
def b():
    chunk.model_dump_json()

# Build a full response
response = ChatCompletionResponse(
    id="chatcmpl-123",
    object="chat.completion",
    created=1234567890,
    model="test-model",
    choices=[
        ChatCompletionChoice(
            index=0,
            message=AssistantMessage(
                role="assistant",
                content="Hello! How can I help you today?"
            ),
            finish_reason="stop"
        )
    ],
    usage=Usage(prompt_tokens=10, completion_tokens=7, total_tokens=17)
)

@bench("ChatCompletionResponse → JSON", iterations=50000)
def b():
    response.model_dump_json()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. JSON DEPTH VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
header("8. JSON Depth Validation (DoS Defense)")

from bit1_mlx.utils.json_depth import json_nesting_depth_exceeds, resolve_max_body_depth

# Flat structure
flat = {"a": 1, "b": 2, "c": 3}
@bench("json_depth flat (depth=2)", iterations=100000)
def b():
    json_nesting_depth_exceeds(flat, 64)

# Deeply nested (depth 50)
deep = {}
cur = deep
for i in range(50):
    cur["a"] = {}
    cur = cur["a"]

@bench("json_depth deep (depth=50)", iterations=50000)
def b():
    json_nesting_depth_exceeds(deep, 64)

# Tool schema depth
tool_schema = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "properties": {
                            "deep": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
}

@bench("json_depth tool schema", iterations=50000)
def b():
    json_nesting_depth_exceeds(tool_schema, 64)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MIDDLEWARE OVERHEAD
# ═══════════════════════════════════════════════════════════════════════════════
header("9. Middleware Overhead (Auth, Body Size)")

from bit1_mlx.middleware.auth import RateLimiter, configure_rate_limiter, _verify_api_key_values

# API key verification
@bench("_verify_api_key_values (no auth)")
def b():
    _verify_api_key_values(None, None)

@bench("_verify_api_key_values (matching)")
def b():
    _verify_api_key_values("test-key", "test-key")

# Rate limiter
limiter = RateLimiter(requests_per_minute=10000, enabled=True)
@bench("RateLimiter.is_allowed() (not limited)")
def b():
    limiter.is_allowed("127.0.0.1")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. STREAMING ROUTERS
# ═══════════════════════════════════════════════════════════════════════════════
header("10. Streaming Output Routers")

from bit1_mlx.api.utils import StreamingThinkRouter, StreamingToolCallFilter

# StreamingThinkRouter — simulate a reasoning model stream
think_tokens = []
for i in range(100):
    if i < 15:
        think_tokens.append(f"<think>thinking step {i}")
    elif i < 30:
        think_tokens.append(f"more thinking {i}")
    elif i < 32:
        think_tokens.append("</think>")
    else:
        think_tokens.append(f"content token {i}")

@bench("StreamingThinkRouter (thinking→content)", iterations=20000)
def b():
    router = StreamingThinkRouter()
    for t in think_tokens:
        router.process(t)

# StreamingToolCallFilter
tool_filter = StreamingToolCallFilter()
tool_stream = []
for i in range(100):
    if i < 50:
        tool_stream.append(f"regular text {i}")
    else:
        tool_stream.append(f"<tool_call>tool call {i}</tool_call>")

@bench("StreamingToolCallFilter", iterations=20000)
def b():
    for t in tool_stream:
        tool_filter.process(t)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
header("SUMMARY")
print(f"  Total aliases in registry:    {len(aliases)}")
print(f"  Total tool parsers:           {len(tool_parsers)}")
print(f"  Python files:                 53")
print(f"  Package size:                 ~26 KB source")
