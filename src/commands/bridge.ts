#!/usr/bin/env -S npx tsx
/**
 * 1bit NPU Bridge v2 — OpenAI-compatible chat + transcription + models.
 * Spawns npu_engine_all for chat. ffmpeg + whisper for transcription.
 * PORT=8081 npx tsx bridge_v2.ts
 */

import Fastify from "fastify";
import { spawn, execSync } from "child_process";
import { readFileSync, existsSync, mkdirSync, unlinkSync } from "fs";
import { createInterface } from "readline";
import { resolve, join } from "path";
import { tmpdir } from "os";
import { randomBytes } from "crypto";
import fastifyMultipart from "@fastify/multipart";

const HOME = process.env.HOME || "/home/bcloud";
const ENGINE = resolve(HOME, "npu-sandbox/npu-infer/build/npu_engine_all");
const WHISPER = "/usr/local/bin/whisper-cpp"; // install: make -C whisper.cpp

const MODELS = {
  qwen3_0_6b: {
    path: resolve(HOME, ".config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"),
    tok: resolve(HOME, ".config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json"),
  },
  qwen3_8b: {
    path: resolve(HOME, "models/Qwen3-8B-NPU2/model.q4nx"),
    tok: resolve(HOME, "models/Qwen3-8B-NPU2/tokenizer.json"),
  },
  llama: {
    path: resolve(HOME, ".config/flm/models/Llama-3.1-8B-NPU2/model.q4nx"),
    tok: resolve(HOME, ".config/flm/models/Llama-3.1-8B-NPU2/tokenizer.json"),
  },
  gemma4_e2b: {
    path: resolve(HOME, ".config/flm/models/Gemma4-E2B-IT-NPU2/model.q4nx"),
    tok: resolve(HOME, ".config/flm/models/Gemma4-E2B-IT-NPU2/tokenizer.json"),
  },
};

type Detok = Map<number, string>;
function loadDetok(tp: string): Detok | null {
  try {
    const raw = JSON.parse(readFileSync(tp, "utf-8"));
    const map = new Map<number, string>();
    const vocab = raw.model?.vocab;
    if (vocab && typeof vocab === "object") {
      for (const [tok, id] of Object.entries(vocab as Record<string, any>)) {
        if (typeof id === "number") map.set(id, tok);
      }
    }
    return map.size > 0 ? map : null;
  } catch { return null; }
}

const detokCache: Record<string, Detok | null> = {};
function getDetok(model: string): Detok | null {
  if (!(model in detokCache)) {
    detokCache[model] = loadDetok(MODELS[model as keyof typeof MODELS]?.tok || "");
  }
  return detokCache[model];
}

function tokenize(text: string, model: string): number[] {
  const d = getDetok(model);
  if (!d) return [];
  const ids: number[] = [];
  let pos = 0;
  const rev = new Map<string, number>();
  for (const [id, tok] of d) rev.set(tok, id);
  while (pos < text.length) {
    let bestId = 0, bestLen = 0;
    for (const [tok, id] of rev) {
      if (text.startsWith(tok, pos) && tok.length > bestLen) {
        bestLen = tok.length; bestId = id;
      }
    }
    if (bestLen > 0) { ids.push(bestId); pos += bestLen; }
    else { ids.push(text.charCodeAt(pos)); pos++; }
  }
  return ids;
}

// ===== SERVER =====
async function main() {
  const app = Fastify({ logger: false });
  await app.register(fastifyMultipart);

  // Health
  app.get("/health", async () => ({
    status: "ok",
    models: Object.keys(MODELS),
    engine: existsSync(ENGINE),
    whisper: existsSync(WHISPER),
  }));

  // Models list
  app.get("/v1/models", async () => {
    const avail = Object.keys(MODELS).filter(k => existsSync(MODELS[k as keyof typeof MODELS].path));
    return {
      object: "list",
      data: avail.map(id => ({ id, object: "model", created: Date.now(), owned_by: "1bit" })),
    };
  });

  // Chat completions
  app.post("/v1/chat/completions", async (req, reply) => {
    const { model = "qwen3_0_6b", messages, stream = false, max_tokens = 32 } = req.body as any;
    const cfg = MODELS[model as keyof typeof MODELS];
    if (!cfg || !existsSync(cfg.path)) {
      return reply.code(404).send({ error: `Unknown model: ${model}` });
    }

    // Build prompt from messages
    let prompt = "";
    for (const m of messages || []) {
      if (m.role === "system") prompt += `<|im_start|>system\n${m.content}<|im_end|>\n`;
      else if (m.role === "user") prompt += `<|im_start|>user\n${m.content}<|im_end|>\n`;
      else if (m.role === "assistant") prompt += `<|im_start|>assistant\n${m.content}<|im_end|>\n`;
    }
    prompt += "<|im_start|>assistant\n";

    // Tokenize (fallback: char codes if no tokenizer)
    const tokens = tokenize(prompt, model);
    const responseId = `chatcmpl-${Date.now()}`;

    if (stream) {
      reply.raw.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });

      // Send role chunk
      const roleChunk = {
        id: responseId, object: "chat.completion.chunk", created: Date.now(),
        model, choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
      };
      reply.raw.write(`data: ${JSON.stringify(roleChunk)}\n\n`);

      const engine = spawn(ENGINE, [cfg.path, String(max_tokens)], { stdio: ["ignore", "pipe", "pipe"] });
      const detok = getDetok(model) || new Map();

      const rl = createInterface({ input: engine.stdout! });
      rl.on("line", (line: string) => {
        let tokId = 0;
        const mt = line.match(/tok=(\d+)/);
        const old = line.match(/^\s+\[(\d+)\]\s+(\d+)\s+\((\d+)ms\)/);
        if (mt) tokId = parseInt(mt[1], 10);
        else if (old) tokId = parseInt(old[2], 10);
        else return;

        const text = detok.get(tokId) || String.fromCodePoint(tokId < 0x110000 ? tokId : 32);
        const chunk = {
          id: responseId, object: "chat.completion.chunk", created: Date.now(),
          model, choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
        };
        reply.raw.write(`data: ${JSON.stringify(chunk)}\n\n`);
      });

      engine.stderr!.on("data", (d: Buffer) => process.stderr.write(d));
      engine.on("close", () => {
        const done = {
          id: responseId, object: "chat.completion.chunk", created: Date.now(),
          model, choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        };
        reply.raw.write(`data: ${JSON.stringify(done)}\n\ndata: [DONE]\n\n`);
        reply.raw.end();
      });
    } else {
      try {
        const engine = spawn(ENGINE, [cfg.path, String(max_tokens)], { stdio: ["ignore", "pipe", "pipe"] });
        const detok = getDetok(model) || new Map();
        let output = "";

        const rl = createInterface({ input: engine.stdout! });
        for await (const line of rl) {
          let tokId = 0;
          const mt = line.match(/tok=(\d+)/);
          const old = line.match(/^\s+\[(\d+)\]\s+(\d+)\s+\((\d+)ms\)/);
          if (mt) tokId = parseInt(mt[1], 10);
          else if (old) tokId = parseInt(old[2], 10);
          else continue;
          output += detok.get(tokId) || "";
        }
        return {
          id: responseId, object: "chat.completion", created: Date.now(),
          model, choices: [{ index: 0, message: { role: "assistant", content: output }, finish_reason: "stop" }],
        };
      } catch (e: any) {
        return reply.code(500).send({ error: e.message });
      }
    }
  });

  // Transcription
  app.post("/v1/audio/transcriptions", async (req, reply) => {
    if (!existsSync(WHISPER)) {
      return reply.code(501).send({
        error: "whisper-cpp not installed.",
        install: "git clone https://github.com/ggerganov/whisper.cpp ~/whisper.cpp && cd ~/whisper.cpp && make && bash ./models/download-ggml-model.sh base.en && sudo cp main /usr/local/bin/whisper-cpp",
      });
    }
    try {
      const data = await req.file();
      if (!data) return reply.code(400).send({ error: "no audio file" });

      const tmpIn = join(tmpdir(), `1bit-${randomBytes(8).toString("hex")}.webm`);
      const tmpWav = join(tmpdir(), `1bit-${randomBytes(8).toString("hex")}.wav`);

      const buf = await data.toBuffer();
      require("fs").writeFileSync(tmpIn, buf);

      // Convert to 16kHz mono WAV
      execSync(`ffmpeg -y -i ${tmpIn} -ar 16000 -ac 1 -f wav ${tmpWav} 2>/dev/null`);

      // Run whisper.cpp
      const result = execSync(`${WHISPER} -m ~/whisper.cpp/models/ggml-base.en.bin -f ${tmpWav} --no-timestamps 2>/dev/null`, { encoding: "utf-8" });

      // Cleanup
      unlinkSync(tmpIn); unlinkSync(tmpWav);

      return { text: result.trim() };
    } catch (e: any) {
      return reply.code(500).send({ error: e.message });
    }
  });

  const PORT = parseInt(process.env.PORT || "8081", 10);
  // TTS — text-to-speech via espeak-ng
  const ESPEAK = "/usr/bin/espeak-ng";
  app.post("/v1/audio/speech", async (req, reply) => {
    const { input, voice = "en-us", speed = 1.0 } = (req.body || {}) as any;
    if (!input) return reply.code(400).send({ error: "missing input text" });
    try {
      const tmpWav = join(tmpdir(), `1bit-tts-${randomBytes(8).toString("hex")}.wav`);
      execSync(`${ESPEAK} -v ${voice} -s ${Math.round(speed * 175)} -w ${tmpWav} "${input.replace(/"/g, '\\"')}" 2>/dev/null`);
      const audio = readFileSync(tmpWav);
      unlinkSync(tmpWav);
      return reply.type("audio/wav").send(audio);
    } catch (e: any) {
      return reply.code(500).send({ error: e.message });
    }
  });

  const HOST = process.env.HOST || "0.0.0.0";
  await app.listen({ port: PORT, host: HOST });
  console.log(`[bridge] http://${HOST}:${PORT} | models: ${Object.keys(MODELS).join(", ")}`);
  console.log(`[bridge] engine: ${ENGINE} (${existsSync(ENGINE) ? "ok" : "MISSING"})`);
  console.log(`[bridge] whisper: ${WHISPER} (${existsSync(WHISPER) ? "ok" : "not installed"})`);
  console.log(`[bridge] espeak: ${ESPEAK} (${existsSync(ESPEAK) ? "ok" : "not installed"})`);
}

main().catch(e => { console.error(e); process.exit(1); });
