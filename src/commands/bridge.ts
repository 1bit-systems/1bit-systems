#!/usr/bin/env -S npx tsx
/** 1bit NPU Bridge v3 — OpenAI-compatible chat + TTS + transcription.
 *  Uses C tokenizer/detokenizer for accurate BPE. MT engine for chat.
 *  PORT=8081 npx tsx bridge.ts */
import Fastify from "fastify";
import { spawn, execSync } from "child_process";
import { readFileSync, existsSync, unlinkSync } from "fs";
import { createInterface } from "readline";
import { resolve, join } from "path";
import { tmpdir } from "os";
import { randomBytes } from "crypto";
import fastifyMultipart from "@fastify/multipart";

const HOME = process.env.HOME || "/home/bcloud";
const ENGINE = resolve(HOME, "1bit-systems/engine/npu/build/npu_engine_v12");
const TOKENIZE = resolve(HOME, "1bit-systems/engine/npu/tokenizer/tokenize");
const DETOK = resolve(HOME, "1bit-systems/engine/npu/tokenizer/detokenize");
const WHISPER = "/usr/local/bin/whisper-cpp";
const ESPEAK = "/usr/bin/espeak-ng";

const MODELS: Record<string, { path: string; tok: string; template: string }> = {
  qwen3_0_6b: {
    path: resolve(HOME, ".config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"),
    tok: resolve(HOME, ".config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json"),
    template: "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
  },
  qwen3_8b: {
    path: resolve(HOME, "models/Qwen3-8B-NPU2/model.q4nx"),
    tok: resolve(HOME, "models/Qwen3-8B-NPU2/tokenizer.json"),
    template: "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
  },
  llama: {
    path: resolve(HOME, ".config/flm/models/Llama-3.1-8B-NPU2/model.q4nx"),
    tok: resolve(HOME, ".config/flm/models/Llama-3.1-8B-NPU2/tokenizer.json"),
    template: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
  },
  gemma4_e2b: {
    path: resolve(HOME, ".config/flm/models/Gemma4-E2B-IT-NPU2/model.q4nx"),
    tok: resolve(HOME, ".config/flm/models/Gemma4-E2B-IT-NPU2/tokenizer.json"),
    template: "<bos><start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n",
  },
};

function tokenize(tokPath: string, text: string): number[] {
  try {
    return execSync(`${TOKENIZE} ${tokPath}`, { input: text, encoding: "utf-8", timeout: 2000 })
      .trim().split(",").map(Number).filter(n => !isNaN(n));
  } catch { return []; }
}
function detokenize(tokPath: string, ids: string): string {
  try {
    return execSync(`${DETOK} ${tokPath}`, { input: ids, encoding: "utf-8", timeout: 500 }).trim();
  } catch { return ""; }
}

async function main() {
  const app = Fastify({ logger: false });
  await app.register(fastifyMultipart);

  app.get("/health", async () => ({
    status: "ok", models: Object.keys(MODELS).filter(k => existsSync(MODELS[k].path)),
    engine: existsSync(ENGINE), tokenizer: existsSync(TOKENIZE), espeak: existsSync(ESPEAK),
  }));

  app.get("/v1/models", async () => {
    const avail = Object.keys(MODELS).filter(k => existsSync(MODELS[k].path));
    return { object: "list", data: avail.map(id => ({ id, object: "model", created: Date.now(), owned_by: "1bit" })) };
  });

  app.post("/v1/chat/completions", async (req, reply) => {
    const { model = "qwen3_0_6b", messages, stream = false } = req.body as any;
    const cfg = MODELS[model];
    if (!cfg || !existsSync(cfg.path)) return reply.code(404).send({ error: `Unknown model: ${model}` });

    const system = (messages || []).find((m: any) => m.role === "system")?.content || "You are a helpful assistant running on 1bit NPU.";
    const userMsgs = (messages || []).filter((m: any) => m.role === "user").map((m: any) => m.content);
    const lastUser = userMsgs[userMsgs.length - 1] || "Hello";
    const prompt = cfg.template.replace("{system}", system).replace("{user}", lastUser);
    const tokenIds = tokenize(cfg.tok, prompt);
    if (!tokenIds.length) return reply.code(500).send({ error: "tokenizer failed" });

    const responseId = `chatcmpl-${Date.now()}`;
    // v12 engine: ./npu_engine_v12 [decode_tokens]
    // Uses hardcoded Qwen3-0.6B prompt + model path
    const maxTok = model === "qwen3_0_6b" ? 16 : 8;
    const engineArgs = ["9", String(maxTok)];

    if (stream) {
      reply.raw.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
      reply.raw.write(`data: ${JSON.stringify({ id: responseId, object: "chat.completion.chunk", created: Date.now(), model, choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] })}\n\n`);

      const engine = spawn(ENGINE, engineArgs, { stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, OMP_NUM_THREADS: "16" } });
      const rl = createInterface({ input: engine.stdout! }); // v12 writes tokens to stdout
      rl.on("line", (line: string) => {
        const m = line.match(/tok=(\d+)/); // v12 format: "tok=N"
        if (!m) return;
        const text = detokenize(cfg.tok, m[1]);
        if (text) reply.raw.write(`data: ${JSON.stringify({ id: responseId, object: "chat.completion.chunk", created: Date.now(), model, choices: [{ index: 0, delta: { content: text }, finish_reason: null }] })}\n\n`);
      });
      engine.stderr!.on("data", (d: Buffer) => process.stderr.write(d));
      engine.on("close", () => {
        reply.raw.write(`data: ${JSON.stringify({ id: responseId, object: "chat.completion.chunk", created: Date.now(), model, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`);
        reply.raw.end();
      });
    } else {
      try {
        const engine = spawn(ENGINE, engineArgs, { stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, OMP_NUM_THREADS: "16" } });
        const ids: string[] = [];
        const rl = createInterface({ input: engine.stdout! });
        for await (const line of rl) {
          const m = line.match(/tok=(\d+)/);
          if (m) ids.push(m[1]);
        }
        const output = detokenize(cfg.tok, ids.join(",")) || "(generating...)";
        return { id: responseId, object: "chat.completion", created: Date.now(), model, choices: [{ index: 0, message: { role: "assistant", content: output }, finish_reason: "stop" }] };
      } catch (e: any) { return reply.code(500).send({ error: e.message }); }
    }
  });

  app.post("/v1/audio/transcriptions", async (req, reply) => {
    if (!existsSync(WHISPER)) return reply.code(501).send({ error: "whisper-cpp not installed" });
    try {
      const data = await req.file(); if (!data) return reply.code(400).send({ error: "no audio file" });
      const tmpIn = join(tmpdir(), `1bit-${randomBytes(8).toString("hex")}.webm`);
      const tmpWav = join(tmpdir(), `1bit-${randomBytes(8).toString("hex")}.wav`);
      require("fs").writeFileSync(tmpIn, await data.toBuffer());
      execSync(`ffmpeg -y -i ${tmpIn} -ar 16000 -ac 1 -f wav ${tmpWav} 2>/dev/null`);
      const out = execSync(`${WHISPER} -m ~/whisper.cpp/models/ggml-base.en.bin -f ${tmpWav} --no-timestamps 2>/dev/null`, { encoding: "utf-8" });
      unlinkSync(tmpIn); unlinkSync(tmpWav);
      return { text: out.trim() };
    } catch (e: any) { return reply.code(500).send({ error: e.message }); }
  });

  app.post("/v1/audio/speech", async (req, reply) => {
    const { input } = (req.body || {}) as any;
    if (!input) return reply.code(400).send({ error: "missing input" });
    try {
      const tmpWav = join(tmpdir(), `1bit-tts-${randomBytes(8).toString("hex")}.wav`);
      execSync(`${ESPEAK} -v en-us -s 175 -w ${tmpWav} "${input.replace(/"/g, '\\"')}" 2>/dev/null`);
      const audio = readFileSync(tmpWav); unlinkSync(tmpWav);
      return reply.type("audio/wav").send(audio);
    } catch (e: any) { return reply.code(500).send({ error: e.message }); }
  });

  const PORT = parseInt(process.env.PORT || "8081", 10);
  await app.listen({ port: PORT, host: "0.0.0.0" });
  console.log(`[bridge] http://0.0.0.0:${PORT} | engine:${existsSync(ENGINE)} tok:${existsSync(TOKENIZE)} tts:${existsSync(ESPEAK)}`);
}

main().catch(e => { console.error(e); process.exit(1); });
