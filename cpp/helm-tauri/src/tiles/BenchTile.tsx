import { useEffect, useRef, useState, type JSX } from "react";
import { fetchSystemStats, type SystemStats } from "../api/lemonade";

const POLL_MS = 1000;
const HISTORY = 60;

interface Series {
  readonly label: string;
  readonly unit: string;
  readonly values: number[];
}

// Plain CSS sparkline. Drawing as a polyline inside an SVG keeps the
// bundle tiny — pulling in a chart lib would 5x the asset payload for
// no real win on a 4-line dashboard.
function Sparkline({ series }: { series: Series }): JSX.Element {
  const max = Math.max(1, ...series.values);
  const points = series.values
    .map((v, i) => {
      const x = (i / Math.max(1, HISTORY - 1)) * 100;
      const y = 100 - (v / max) * 100;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const last = series.values.at(-1) ?? 0;
  return (
    <div className="sparkline">
      <div className="sparkline-label">
        <span>{series.label}</span>
        <span className="sparkline-value">
          {last.toFixed(1)} {series.unit}
        </span>
      </div>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline points={points} />
      </svg>
    </div>
  );
}

export function BenchTile(): JSX.Element {
  const [tokps, setTokps] = useState<number[]>([]);
  const [promptps, setPromptps] = useState<number[]>([]);
  const [gpuMem, setGpuMem] = useState<number[]>([]);
  const [ram, setRam] = useState<number[]>([]);
  const [kv, setKv] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);
  const acRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async (): Promise<void> => {
      const ac = new AbortController();
      acRef.current = ac;
      try {
        const s: SystemStats = await fetchSystemStats(ac.signal);
        if (cancelled) return;
        setError(null);
        const push = (
          set: React.Dispatch<React.SetStateAction<number[]>>,
          v: number,
        ): void =>
          set((prev) => {
            const next = [...prev, v];
            return next.length > HISTORY ? next.slice(-HISTORY) : next;
          });
        push(setTokps, s.tokens_per_sec);
        push(setPromptps, s.prompt_tokens_per_sec);
        push(setGpuMem, s.gpu_mem_mb);
        push(setRam, s.system_ram_mb);
        push(setKv, s.kv_cache_mb);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
      acRef.current?.abort();
    };
  }, []);

  return (
    <div className="tile bench-tile">
      {error ? <p className="error">lemonade unreachable: {error}</p> : null}
      <Sparkline
        series={{ label: "decode", unit: "tok/s", values: tokps }}
      />
      <Sparkline
        series={{ label: "prompt", unit: "tok/s", values: promptps }}
      />
      <Sparkline series={{ label: "gpu", unit: "MB", values: gpuMem }} />
      <Sparkline series={{ label: "ram", unit: "MB", values: ram }} />
      <Sparkline series={{ label: "kv", unit: "MB", values: kv }} />
    </div>
  );
}
