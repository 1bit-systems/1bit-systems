import { useCallback, useRef, useState, type JSX } from "react";
import { streamChat, type ChatMessage } from "../api/lemonade";

// Compact transcript + push-to-talk placeholder. Voice loop wires to
// cpp/echo via WebRTC in a later pass — today the button is a stub
// so the UX is wired but the audio path is not.
export function HaloTile(): JSX.Element {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "system", content: "you are halo, a calm-technical assistant" },
  ]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(async () => {
    const trimmed = draft.trim();
    if (!trimmed || busy) return;
    const userMsg: ChatMessage = { role: "user", content: trimmed };
    const next: ChatMessage[] = [...messages, userMsg];
    setMessages(next);
    setDraft("");
    setBusy(true);

    const ac = new AbortController();
    abortRef.current = ac;
    let assistantBuf = "";
    try {
      for await (const delta of streamChat(next, ac.signal)) {
        assistantBuf += delta;
        setMessages([...next, { role: "assistant", content: assistantBuf }]);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setMessages([
        ...next,
        { role: "assistant", content: `[stream error: ${msg}]` },
      ]);
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }, [draft, busy, messages]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return (
    <div className="tile halo-tile">
      <ol className="halo-transcript">
        {messages
          .filter((m) => m.role !== "system")
          .map((m, i) => (
            <li key={i} className={`msg msg-${m.role}`}>
              <span className="msg-role">{m.role}</span>
              <span className="msg-content">{m.content}</span>
            </li>
          ))}
      </ol>
      <div className="halo-input">
        <button
          type="button"
          className="ptt"
          title="push-to-talk (wires to cpp/echo)"
          disabled
        >
          ptt
        </button>
        <input
          type="text"
          value={draft}
          placeholder="ask halo..."
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          disabled={busy}
        />
        {busy ? (
          <button type="button" onClick={cancel}>
            stop
          </button>
        ) : (
          <button type="button" onClick={() => void send()}>
            send
          </button>
        )}
      </div>
    </div>
  );
}
