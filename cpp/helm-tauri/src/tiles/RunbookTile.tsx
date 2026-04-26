import { useEffect, useMemo, useState, type JSX } from "react";
import MarkdownIt from "markdown-it";
import { listRunbooks, readRunbook } from "../api/tauri";

const md = new MarkdownIt({ html: false, linkify: true, breaks: false });

interface TocEntry {
  readonly level: number;
  readonly text: string;
  readonly anchor: string;
}

function buildToc(source: string): TocEntry[] {
  const out: TocEntry[] = [];
  const lines = source.split("\n");
  for (const line of lines) {
    const m = /^(#{1,6})\s+(.+)$/.exec(line);
    if (!m) continue;
    const level = m[1]!.length;
    const text = m[2]!.trim();
    const anchor = text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
    out.push({ level, text, anchor });
  }
  return out;
}

export function RunbookTile(): JSX.Element {
  const [files, setFiles] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [source, setSource] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listRunbooks()
      .then((paths) => {
        setFiles(paths);
        if (paths.length > 0 && paths[0]) setSelected(paths[0]);
      })
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : String(e)),
      );
  }, []);

  useEffect(() => {
    if (!selected) return;
    readRunbook(selected)
      .then(setSource)
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : String(e)),
      );
  }, [selected]);

  const toc = useMemo(() => buildToc(source), [source]);
  const html = useMemo(() => md.render(source), [source]);

  const jump = (anchor: string): void => {
    const el = document.getElementById(`rb-${anchor}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    el.classList.add("rb-flash");
    window.setTimeout(() => el.classList.remove("rb-flash"), 800);
  };

  return (
    <div className="tile runbook-tile">
      <aside className="runbook-toc">
        <ul className="runbook-files">
          {files.map((f) => (
            <li
              key={f}
              className={f === selected ? "active" : ""}
              onClick={() => setSelected(f)}
            >
              {f}
            </li>
          ))}
        </ul>
        <hr />
        <ul className="runbook-headings">
          {toc.map((t, i) => (
            <li
              key={i}
              style={{ paddingLeft: `${(t.level - 1) * 8}px` }}
              onClick={() => jump(t.anchor)}
            >
              {t.text}
            </li>
          ))}
        </ul>
      </aside>
      <article className="runbook-body">
        {error ? <p className="error">{error}</p> : null}
        {/* markdown-it sanitizes html=false and the host whitelists the
            runbooks dir, so injecting the rendered html is safe here. */}
        <div
          className="runbook-md"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </article>
    </div>
  );
}
