import os
import re
import time
from pathlib import Path

KNOWLEDGE_DIR = os.environ.get("JARVIS_KNOWLEDGE_DIR", os.path.expanduser("~/jarvis/data/knowledge"))


class KnowledgeBase:
    """RAG: full-text keyword search over markdown files."""

    def __init__(self, root=KNOWLEDGE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ["facts", "documents", "conversations", "tools"]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def all_files(self):
        return sorted(self.root.rglob("*.md"))

    def search(self, query, max_results=5):
        query_lower = query.lower()
        terms = query_lower.split()
        results = []
        for fpath in self.all_files():
            try:
                text = fpath.read_text()
            except:
                continue
            text_lower = text.lower()
            score = sum(text_lower.count(t) for t in terms)
            if score == 0:
                continue
            title = ""
            for line in text.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            snippet = ""
            for term in terms:
                idx = text_lower.find(term)
                if idx >= 0:
                    start = max(0, idx - 60)
                    end = min(len(text), idx + 120)
                    snippet = text[start:end].replace("\n", " ").strip()
                    break
            rel_path = str(fpath.relative_to(self.root))
            results.append({"path": rel_path, "title": title, "score": score, "snippet": snippet[:200]})
        results.sort(key=lambda x: -x["score"])
        return results[:max_results]

    def add_document(self, filename, content):
        doc_dir = self._resolve_safe("documents")
        doc_dir.mkdir(exist_ok=True)
        safe_name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', filename)
        fpath = self._resolve_safe("documents", safe_name)
        if not safe_name.endswith(".md"):
            fpath = doc_dir / (safe_name + ".md")
            content = f"---\ntype: document\ncreated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\nsource: upload\n---\n\n# {filename}\n\n" + content
        fpath.write_text(content)
        return str(fpath.relative_to(self.root))

    def get_knowledge_context(self, query, max_results=3):
        results = self.search(query, max_results)
        if not results:
            return ""
        parts = ["Here is relevant information from the knowledge base:"]
        for r in results:
            parts.append(f"\n--- {r['title']} ---")
            parts.append(r['snippet'])
        return "\n".join(parts)

    # ── Local multi-turn memory ──────────────────────────────────────
    # Session transcripts live under conversations/<session_id>.md, append-only.
    # This is server-side memory: it persists across devices/clients hitting the
    # same session_id (e.g. phone app + desktop), independent of whatever
    # truncated history a given client happens to send in `messages`.

    def _resolve_safe(self, *components):
        """Resolve a path under self.root with path-traversal protection."""
        candidate = self.root.joinpath(*components).resolve()
        # Ensure the resolved path stays within self.root (defense-in-depth)
        if not str(candidate).startswith(str(self.root.resolve())):
            raise ValueError(f"Path traversal blocked: {candidate}")
        return candidate

    def _session_path(self, session_id):
        safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', session_id)[:128] or "default"
        return self._resolve_safe("conversations", f"{safe}.md")

    def save_turn(self, session_id, role, content):
        fpath = self._session_path(session_id)
        ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        if not fpath.exists():
            fpath.write_text(f"---\ntype: conversation\nsession: {session_id}\ncreated: {ts}\n---\n\n# Session {session_id}\n\n")
        with fpath.open("a") as f:
            f.write(f"\n## {ts} — {role}\n\n{content}\n")

    def get_recent_conversation(self, session_id, max_turns=10):
        fpath = self._session_path(session_id)
        if not fpath.exists():
            return []
        text = fpath.read_text()
        turns = []
        for block in text.split("\n## ")[1:]:
            header, _, body = block.partition("\n")
            m = re.match(r".*? — (\w+)", header)
            if not m:
                continue
            turns.append({"role": m.group(1), "content": body.strip()})
        return turns[-max_turns:]

    def list_sessions(self):
        conv_dir = self.root / "conversations"
        return sorted(p.stem for p in conv_dir.glob("*.md"))


_kb = KnowledgeBase()