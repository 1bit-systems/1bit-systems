import { useEffect, useState, type JSX } from "react";
import { listTools, invokeTool, type MCPTool } from "../api/mcp";

type ArgValue = string | number | boolean;

export function MCPTile(): JSX.Element {
  const [tools, setTools] = useState<MCPTool[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<MCPTool | null>(null);
  const [args, setArgs] = useState<Record<string, ArgValue>>({});
  const [output, setOutput] = useState<string>("");

  useEffect(() => {
    const ac = new AbortController();
    listTools(ac.signal)
      .then(setTools)
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => ac.abort();
  }, []);

  const invoke = async (): Promise<void> => {
    if (!selected) return;
    setOutput("...");
    try {
      const res = await invokeTool(selected.id, args);
      setOutput(res.ok ? res.output : (res.error ?? "error"));
    } catch (e) {
      setOutput(e instanceof Error ? e.message : String(e));
    }
  };

  if (error) {
    return (
      <div className="tile mcp-tile">
        <p className="error">halo-agent unreachable: {error}</p>
      </div>
    );
  }

  if (selected) {
    return (
      <div className="tile mcp-tile mcp-detail">
        <div className="mcp-header">
          <button type="button" onClick={() => setSelected(null)}>
            back
          </button>
          <h3>{selected.title}</h3>
        </div>
        <p className="muted">{selected.description}</p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void invoke();
          }}
        >
          {selected.args.map((a) => (
            <label key={a.name} className="mcp-arg">
              <span>
                {a.name}
                {a.required ? " *" : ""}
              </span>
              <input
                type={a.type === "number" ? "number" : "text"}
                onChange={(e) =>
                  setArgs((prev) => ({
                    ...prev,
                    [a.name]:
                      a.type === "number"
                        ? Number(e.target.value)
                        : e.target.value,
                  }))
                }
              />
            </label>
          ))}
          <button type="submit">invoke</button>
        </form>
        <pre className="mcp-output">{output}</pre>
      </div>
    );
  }

  return (
    <div className="tile mcp-tile">
      <ul className="mcp-grid">
        {tools.map((t) => (
          <li key={t.id} className="mcp-card" onClick={() => setSelected(t)}>
            <div className="mcp-card-title">{t.title}</div>
            <div className="mcp-card-source">{t.source}</div>
            <div className="mcp-card-desc">{t.description}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
