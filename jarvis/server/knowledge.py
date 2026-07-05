"""JARVIS Open Knowledge Format — transparent, file-based knowledge system.

Every piece of knowledge JARVIS learns is stored as a human-readable markdown
file with YAML frontmatter. You can read, edit, add, or delete knowledge with
any text editor. No locked databases. No proprietary formats.

Format:
  /home/bcloud/jarvis/data/knowledge/
    ├── index.json              # Full-text search index (auto-generated)
    ├── conversations/          # Chat histories (markdown logs)
    ├── documents/              # Uploaded RAG documents
    ├── facts/                  # Structured facts JARVIS learned
    └── tools/                  # Tool output snapshots

File format (.md):
  ---
  type: fact | document | conversation | tool_output
  created: 2026-07-05T18:30:00Z
  tags: [npu, performance, benchmark]
  source: user | web | tool | inference
  confidence: 0.0-1.0
  ---
  # Title
  Content in markdown...
"""
import os
import json
import glob
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class KnowledgeEntry:
    """A single knowledge entry in open format."""

    def __init__(self, path: str = None, entry_type: str = "fact",
                 title: str = "", content: str = "", tags: list = None,
                 source: str = "user", confidence: float = 1.0):
        self.path = path
        self.entry_type = entry_type  # fact, document, conversation, tool_output
        self.title = title
        self.content = content
        self.tags = tags or []
        self.source = source
        self.confidence = min(max(confidence, 0.0), 1.0)
        self.created = datetime.now(timezone.utc)
        self.updated = self.created

    @property
    def filename(self) -> str:
        safe = re.sub(r'[^a-zA-Z0-9_\-\s]', '', self.title).strip().replace(' ', '_')
        return f"{safe}_{self.created.strftime('%Y%m%d_%H%M%S')}.md"

    @property
    def yaml_frontmatter(self) -> str:
        return f"""---
type: {self.entry_type}
created: {self.created.isoformat()}
updated: {self.updated.isoformat()}
tags: [{', '.join(self.tags)}]
source: {self.source}
confidence: {self.confidence}
---"""

    def to_markdown(self) -> str:
        return f"{self.yaml_frontmatter}\n\n# {self.title}\n\n{self.content}\n"

    @staticmethod
    def from_markdown(path: str) -> 'KnowledgeEntry':
        with open(path) as f:
            text = f.read()

        entry = KnowledgeEntry(path=path)

        # Parse YAML frontmatter
        if text.startswith('---'):
            parts = text.split('---', 2)
            if len(parts) >= 3:
                fm = parts[1]
                entry.content = parts[2].strip()
                for line in fm.strip().split('\n'):
                    if ':' in line:
                        key, _, val = line.partition(':')
                        key = key.strip()
                        val = val.strip()
                        if key == 'type':
                            entry.entry_type = val
                        elif key == 'tags':
                            entry.tags = [t.strip() for t in val.strip('[]').split(',') if t.strip()]
                        elif key == 'source':
                            entry.source = val
                        elif key == 'confidence':
                            try:
                                entry.confidence = float(val)
                            except ValueError:
                                pass
                        elif key == 'created':
                            try:
                                entry.created = datetime.fromisoformat(val)
                            except ValueError:
                                pass

        # Extract title from first heading
        if not entry.title:
            for line in entry.content.split('\n'):
                if line.startswith('# '):
                    entry.title = line[2:].strip()
                    break

        if not entry.title:
            entry.title = Path(path).stem.replace('_', ' ').title()

        return entry


class OpenKnowledge:
    """Open knowledge base — all learnings in human-readable markdown files."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.knowledge_dir = self.data_dir / "knowledge"
        self.index_path = self.knowledge_dir / "index.json"

        # Create directory structure
        for subdir in ["conversations", "documents", "facts", "tools"]:
            (self.knowledge_dir / subdir).mkdir(parents=True, exist_ok=True)

        self._entries: dict[str, KnowledgeEntry] = {}
        self._dirty = False
        self._load_index()

    # ─── Index Management ────────────────────────────────────────

    def _load_index(self):
        """Load the knowledge index from disk."""
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text())
                for path, meta in data.get("entries", {}).items():
                    if os.path.exists(path):
                        self._entries[path] = KnowledgeEntry.from_markdown(path)
                return
            except (json.JSONDecodeError, KeyError):
                pass
        # Fallback: scan directories
        self._scan_all()

    def _save_index(self):
        """Save the knowledge index."""
        index = {
            "version": 2,
            "format": "open-knowledge-markdown",
            "description": "JARVIS Open Knowledge — all files are human-readable markdown",
            "updated": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(self._entries),
            "entries": {
                path: {
                    "type": e.entry_type,
                    "title": e.title,
                    "tags": e.tags,
                    "source": e.source,
                    "confidence": e.confidence,
                    "created": e.created.isoformat(),
                    "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
                }
                for path, e in self._entries.items()
            },
        }
        self.index_path.write_text(json.dumps(index, indent=2))
        self._dirty = False

    def _scan_all(self):
        """Scan knowledge directories for all .md files."""
        self._entries = {}
        for subdir in ["conversations", "documents", "facts", "tools"]:
            dir_path = self.knowledge_dir / subdir
            if dir_path.exists():
                for fpath in dir_path.glob("*.md"):
                    try:
                        entry = KnowledgeEntry.from_markdown(str(fpath))
                        self._entries[str(fpath)] = entry
                    except Exception:
                        pass
        self._save_index()

    def save_on_exit(self):
        """Call on graceful shutdown."""
        if self._dirty:
            self._save_index()

    # ─── CRUD Operations ──────────────────────────────────────────

    def add(self, entry: KnowledgeEntry) -> str:
        """Add a knowledge entry. Returns file path."""
        if entry.entry_type == "fact":
            subdir = "facts"
        elif entry.entry_type == "document":
            subdir = "documents"
        elif entry.entry_type == "conversation":
            subdir = "conversations"
        elif entry.entry_type == "tool_output":
            subdir = "tools"
        else:
            subdir = "facts"

        path = str(self.knowledge_dir / subdir / entry.filename)
        entry.path = path

        with open(path, 'w') as f:
            f.write(entry.to_markdown())

        self._entries[path] = entry
        self._dirty = True
        return path

    def get(self, path: str) -> Optional[KnowledgeEntry]:
        return self._entries.get(path)

    def delete(self, path: str) -> bool:
        if path in self._entries:
            try:
                os.remove(path)
                del self._entries[path]
                self._dirty = True
                return True
            except OSError:
                return False
        return False

    def update(self, path: str, content: str = None, tags: list = None,
               confidence: float = None) -> bool:
        entry = self._entries.get(path)
        if not entry:
            return False
        if content is not None:
            entry.content = content
        if tags is not None:
            entry.tags = tags
        if confidence is not None:
            entry.confidence = min(max(confidence, 0.0), 1.0)
        entry.updated = datetime.now(timezone.utc)

        with open(path, 'w') as f:
            f.write(entry.to_markdown())

        self._dirty = True
        return True

    # ─── Search ──────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 10,
               entry_types: list[str] = None) -> list[KnowledgeEntry]:
        """Search knowledge by keyword (full-text on content + title + tags)."""
        query_terms = set(query.lower().split())
        scored = []

        for entry in self._entries.values():
            if entry_types and entry.entry_type not in entry_types:
                continue

            text = (entry.title + " " + entry.content).lower()
            tags_text = " ".join(entry.tags).lower()

            # Score: matches in title > tags > content
            score = 0
            for term in query_terms:
                if term in entry.title.lower():
                    score += 3
                if term in tags_text:
                    score += 2
                score += entry.content.lower().count(term) * 0.5

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:max_results]]

    def list_by_type(self, entry_type: str) -> list[KnowledgeEntry]:
        return [e for e in self._entries.values() if e.entry_type == entry_type]

    def all(self) -> list[KnowledgeEntry]:
        return list(self._entries.values())

    def stats(self) -> dict:
        counts = {}
        for e in self._entries.values():
            counts[e.entry_type] = counts.get(e.entry_type, 0) + 1
        return {
            "total": len(self._entries),
            "by_type": counts,
            "directory": str(self.knowledge_dir),
            "format": "Open Knowledge Markdown (.md with YAML frontmatter)",
        }


# ─── Convenience ───────────────────────────────────────────────

def fact(title: str, content: str, tags: list = None,
         source: str = "inference", confidence: float = 0.9) -> KnowledgeEntry:
    return KnowledgeEntry(entry_type="fact", title=title,
                          content=content, tags=tags,
                          source=source, confidence=confidence)


def document_entry(title: str, content: str, tags: list = None) -> KnowledgeEntry:
    return KnowledgeEntry(entry_type="document", title=title,
                          content=content, tags=tags, source="user")


def conversation_entry(title: str, content: str) -> KnowledgeEntry:
    return KnowledgeEntry(entry_type="conversation", title=title,
                          content=content, tags=["conversation"],
                          source="user", confidence=1.0)
