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
        doc_dir = self.root / "documents"
        doc_dir.mkdir(exist_ok=True)
        safe_name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', filename)
        fpath = doc_dir / safe_name
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


_kb = KnowledgeBase()