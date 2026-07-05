"""JARVIS Agent — LLM orchestration with tool calling and memory."""
import json
import re
import httpx
from typing import AsyncGenerator, Optional
from config import JarvisConfig
from knowledge import OpenKnowledge, KnowledgeEntry, fact, document_entry


class ToolResult:
    def __init__(self, name: str, success: bool, output: str):
        self.name = name
        self.success = success
        self.output = output

    def to_dict(self):
        return {"name": self.name, "success": self.success, "output": self.output}


class JarvisAgent:
    """Orchestrates LLM calls with tool execution and conversation memory."""

    def __init__(self, config: JarvisConfig, knowledge: OpenKnowledge = None):
        self.config = config
        self.knowledge = knowledge
        self.conversations: dict[str, list[dict]] = {}
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return """You are JARVIS — your private AI assistant running on 1bit.systems hardware.
Powered by NPU (XDNA 2) + GPU (Radeon 8060S) + CPU (Zen 5).

You have access to tools and capabilities:
- **Open Knowledge**: All learnings are stored as human-readable markdown files.
  You can read, search, and add to the knowledge base. Format: .md with YAML frontmatter.
- **RAG**: Search through uploaded documents via the Open Knowledge base.
- **Vision**: Analyze images you're shown.
- **Voice**: Speech input/output.
- **Tools**: Calculator, Python code execution, file operations.

Everything runs locally — no cloud. Be concise, helpful, and precise.
You learn continuously: save important facts using the knowledge system.

Current model: {model}
""".format(model=self.config.llm_model)

    def get_or_create_conversation(self, session_id: str) -> list[dict]:
        if session_id not in self.conversations:
            self.conversations[session_id] = [
                {"role": "system", "content": self.system_prompt}
            ]
            # Trim oldest messages (keep system + recent)
            if len(self.conversations[session_id]) > self.config.max_history:
                keep = self.conversations[session_id][:1] + self.conversations[session_id][-(self.config.max_history - 1):]
                self.conversations[session_id] = keep
        return self.conversations[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        conv = self.get_or_create_conversation(session_id)
        conv.append({"role": role, "content": content})
        # Trim
        if len(conv) > self.config.max_history:
            keep = conv[:1] + conv[-(self.config.max_history - 1):]
            self.conversations[session_id] = keep

    async def chat_stream(self, session_id: str, message: str, image_b64: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream a chat response, executing tools as needed."""
        conv = self.get_or_create_conversation(session_id)

        # Build user message content
        user_content = message
        if image_b64:
            user_content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]

        conv.append({"role": "user", "content": user_content})

        # Get LLM response (non-streaming for now)
        full_response = ""
        async for chunk in self._llm_stream(conv):
            if chunk.get("type") == "content":
                text = chunk.get("text", "")
                yield text
                full_response += text
            elif chunk.get("type") == "done":
                break

        # Save to conversation history
        conv.append({"role": "assistant", "content": full_response or "(no response)"})

    async def _llm_stream(self, messages: list[dict]) -> AsyncGenerator[dict, None]:
        """Query the LLM via FLM (NPU). Non-streaming for compatibility."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                payload = {
                    "model": self.config.llm_model,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "stream": False,
                }
                
                last_err = None
                
                # Try FLM directly (port 52625)
                try:
                    resp = await client.post(
                        f"http://127.0.0.1:{self.config.flm_port}/v1/chat/completions",
                        json=payload, timeout=60.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            msg = choices[0].get("message", {})
                            content = msg.get("content", "")
                            if content:
                                yield {"type": "content", "text": content}
                        yield {"type": "done"}
                        return
                    else:
                        last_err = f"HTTP {resp.status_code}"
                except Exception as e:
                    last_err = str(e)
                
                # Fallback: unified daemon (port 9090)
                try:
                    daemon_model = self.config.api_base.replace("http://", "").split(":")[0] if "://" in self.config.api_base else ""
                    payload["model"] = "Qwen3-0.6B-NPU2"
                    resp = await client.post(
                        f"{self.config.api_base}/v1/chat/completions",
                        json=payload, timeout=60.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            msg = choices[0].get("message", {})
                            content = msg.get("content", "")
                            if content:
                                yield {"type": "content", "text": content}
                        yield {"type": "done"}
                        return
                except Exception as e:
                    last_err = str(e)
                
                yield {"type": "content", "text": f"[JARVIS: Cannot reach LLM — {last_err}]"}
                yield {"type": "done"}
            except Exception as e:
                yield {"type": "content", "text": f"[JARVIS Error: {e}]"}
                yield {"type": "done"}

    async def _execute_tool(self, tool_call: dict) -> ToolResult:
        """Execute a tool call."""
        name = tool_call["name"]
        try:
            args = json.loads(tool_call.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}

        if name == "calculator":
            return await self._tool_calculator(args)
        elif name == "web_search":
            return await self._tool_web_search(args)
        elif name == "python_exec":
            return await self._tool_python_exec(args)
        elif name == "read_file":
            return await self._tool_read_file(args)
        elif name == "list_dir":
            return await self._tool_list_dir(args)
        else:
            return ToolResult(name, False, f"Unknown tool: {name}")

    async def _tool_calculator(self, args: dict) -> ToolResult:
        expr = args.get("expression", "")
        try:
            # Safe eval — only allow math operations
            import math
            allowed = {"abs", "pow", "round", "min", "max", "sum", "len", "int", "float", "str"}
            safe_dict = {k: getattr(math, k, None) for k in dir(math) if not k.startswith("_")}
            safe_dict.update({"abs": abs, "pow": pow, "round": round, "min": min, "max": max,
                           "sum": sum, "len": len, "int": int, "float": float, "str": str})
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return ToolResult("calculator", True, str(result))
        except Exception as e:
            return ToolResult("calculator", False, f"Error: {e}")

    async def _tool_web_search(self, args: dict) -> ToolResult:
        query = args.get("query", "")
        try:
            # Use the available web_search tool through the system
            return ToolResult("web_search", True, f"Web search for '{query}' — use chrome or browser tool for full results.")
        except Exception as e:
            return ToolResult("web_search", False, f"Error: {e}")

    async def _tool_python_exec(self, args: dict) -> ToolResult:
        code = args.get("code", "")
        try:
            import io, sys
            old_stdout = sys.stdout
            capt = io.StringIO()
            sys.stdout = capt
            try:
                exec(code, {"__builtins__": __builtins__})
            finally:
                sys.stdout = old_stdout
            return ToolResult("python_exec", True, capt.getvalue() or "(no output)")
        except Exception as e:
            return ToolResult("python_exec", False, f"Error: {e}")

    async def _tool_read_file(self, args: dict) -> ToolResult:
        path = args.get("path", "")
        try:
            with open(path, "r") as f:
                content = f.read()
            return ToolResult("read_file", True, content[:4000])
        except Exception as e:
            return ToolResult("read_file", False, f"Error: {e}")

    async def _tool_list_dir(self, args: dict) -> ToolResult:
        path = args.get("path", ".")
        try:
            import os
            items = os.listdir(path)
            return ToolResult("list_dir", True, "\n".join(items[:50]))
        except Exception as e:
            return ToolResult("list_dir", False, f"Error: {e}")

    def set_knowledge(self, knowledge: OpenKnowledge):
        """Connect the agent to the open knowledge base."""
        self.knowledge = knowledge

    async def rag_query(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context from the open knowledge base."""
        if not hasattr(self, 'knowledge') or self.knowledge is None:
            return ""
        
        # Search all knowledge types
        results = self.knowledge.search(query, max_results=top_k)
        
        if not results:
            return ""
        
        # Format as readable context
        sections = []
        for entry in results:
            sections.append(f"📄 **{entry.title}** [{entry.entry_type}]\n{entry.content[:1500]}")
        
        return "\n\n---\n\n".join(sections)
