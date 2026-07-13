#!/usr/bin/env npx tsx
/** 1bit NPU Server — OpenAI-compatible chat completions with autoregressive decode */

import Fastify from "fastify";
import cors from "@fastify/cors";
import fastifyStatic from "@fastify/static";
import { spawn, execSync } from "child_process";
import { readFileSync, existsSync, readdirSync } from "fs";
import { join } from "path";
import { randomUUID } from "crypto";

const MODELS_BASE = "/home/bcloud/.config/flm/models";
const MODELS_BASE_FALLBACK = "/home/bcloud/models";
const ENGINE = "/home/bcloud/npu-sandbox/npu-infer/build/npu_engine_mt";
const TOKENIZE_BIN = "/home/bcloud/1bit-systems/engine/npu/tokenizer/tokenize";
const UI_DIR = "/home/bcloud/npu-sandbox/npu-infer/1bit/ui";

// ── Model discovery ──────────────────────────────────────────────

interface ModelInfo {
  id: string; name: string; modelPath: string; tokenizerPath: string;
  status: string;
}

function findQ4NX(dir: string): string | null {
  try {
    const e = readdirSync(dir).find(f => f.endsWith(".q4nx"));
    return e ? join(dir, e) : null;
  } catch { return null; }
}

function discoverModels(): ModelInfo[] {
  const dirs = new Map<string,string>();
  for (const base of [MODELS_BASE, MODELS_BASE_FALLBACK]) {
    try { for (const e of readdirSync(base)) {
      const p = join(base, e); if (!dirs.has(e) && existsSync(p)) dirs.set(e, p);
    }} catch {}
  }
  const models: ModelInfo[] = [];
  for (const [name, dir] of dirs) {
    const qp = findQ4NX(dir);
    models.push({
      id: name, name, status: qp ? "ready" : "no_model_file",
      modelPath: qp || dir, tokenizerPath: dir,
    });
  }
  return models;
}

// ── Tokenizer: use C binary for encode, JSON reverse-lookup for decode ──

const tokenizerCache = new Map<string, { decode: (id: number) => string }>();

function getDecoder(modelDir: string) {
  const cached = tokenizerCache.get(modelDir);
  if (cached) return cached;

  const raw = JSON.parse(readFileSync(join(modelDir, "tokenizer.json"), "utf-8"));
  const idToToken = new Map<number, string>();
  const vocab = raw.model?.vocab || {};
  for (const [token, id] of Object.entries(vocab)) {
    idToToken.set(id as number, token);
  }
  for (const t of (raw.added_tokens || [])) {
    idToToken.set(t.id, t.content);
  }

  const decoder = {
    decode(id: number): string {
      return idToToken.get(id) || "";
    },
  };
  tokenizerCache.set(modelDir, decoder);
  return decoder;
}

function encode(tokPath: string, text: string): number[] {
  try {
    const out = execSync(`${TOKENIZE_BIN} "${tokPath}"`, {
      input: text, encoding: "utf-8", timeout: 3000,
    });
    return out.trim().split(",").map(Number).filter(n => !isNaN(n) && n >= 0);
  } catch { return []; }
}

// ── Run engine, return token IDs ──────────────────────────────────

function runEngine(modelPath: string, tokens: number[], genCount: number): Promise<number[]> {
  return new Promise((resolve, reject) => {
    const tokArgs = tokens.map(String).join(" ");
    // NPU_GEN passed inline since sudo strips env vars
    const genPrefix = genCount > 0 ? `NPU_GEN=${genCount} ` : "";
    const cmd = `${genPrefix}${ENGINE} ${modelPath} ${tokArgs}`;
    const proc = spawn("sudo", ["sh", "-c", cmd], {
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "", stderr = "";
    proc.stdout!.on("data", (d: Buffer) => { stdout += d.toString(); });
    proc.stderr!.on("data", (d: Buffer) => { stderr += d.toString(); });
    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Engine exit ${code}: ${stderr.slice(-300)}`));
        return;
      }
      const ids = stdout.trim().split(/\s+/).map(Number).filter(n => !isNaN(n) && n >= 0);
      resolve(ids);
    });
    proc.on("error", reject);
  });
}

// ── Server ────────────────────────────────────────────────────────

async function main() {
  const models = discoverModels();
  const defaultModel = models.find(m => m.status === "ready")?.id || "none";

  const app = Fastify({ logger: true });
  await app.register(cors, { origin: true });
  await app.register(fastifyStatic, { root: UI_DIR, prefix: "/ui/", decorateReply: false });

  app.get("/health", async () => ({ status: "ok", service: "1bit" }));

  app.get("/v1/models", async () => {
    const data = models.filter(m => m.status === "ready").map(m => ({
      id: m.id, object: "model", created: Math.floor(Date.now()/1000), owned_by: "1bit",
    }));
    return { object: "list", data };
  });

  app.post("/v1/chat/completions", async (req, reply) => {
    const body = req.body as any;
    const modelId = body.model || defaultModel;
    const messages = body.messages || [];
    const stream = body.stream === true;
    const maxTokens = Math.min(body.max_tokens || 16, 64);

    const model = models.find(m => m.id === modelId);
    if (!model || model.status !== "ready") {
      return reply.status(404).send({ error: `Model '${modelId}' not found` });
    }

    // Build Qwen chat template
    const system = messages.find((m: any) => m.role === "system")?.content || "";
    const userMsgs = messages.filter((m: any) => m.role === "user").map((m: any) => m.content);
    const lastUser = userMsgs[userMsgs.length - 1] || "";
    const prompt = system
      ? `<|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${lastUser}<|im_end|>\n<|im_start|>assistant\n`
      : `<|im_start|>user\n${lastUser}<|im_end|>\n<|im_start|>assistant\n`;

    const decoder = getDecoder(model.tokenizerPath);
    const tokJsonPath = join(model.tokenizerPath, "tokenizer.json");
    const inputTokens = encode(tokJsonPath, prompt);

    if (inputTokens.length === 0) {
      return reply.status(400).send({ error: "Tokenizer produced no tokens" });
    }
    if (inputTokens.length > 256) {
      inputTokens.splice(256); // truncate
    }

    try {
      // Single engine call with autoregressive decode
      const allIds = await runEngine(model.modelPath, inputTokens, maxTokens);
      // First len(inputTokens) predictions are the prefill output; the rest are generated
      const generatedIds = allIds.slice(inputTokens.length);
      if (generatedIds.length === 0) {
        throw new Error("Engine produced no generated tokens");
      }

      const completionId = `chatcmpl-${randomUUID()}`;
      const created = Math.floor(Date.now() / 1000);

      if (stream) {
        reply.raw.writeHead(200, {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        });
        reply.raw.write(`data: ${JSON.stringify({
          id: completionId, object: "chat.completion.chunk", created, model: model.id,
          choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
        })}\n\n`);

        for (let i = 0; i < generatedIds.length; i++) {
          const text = decoder.decode(generatedIds[i]);
          reply.raw.write(`data: ${JSON.stringify({
            id: completionId, object: "chat.completion.chunk", created, model: model.id,
            choices: [{ index: 0, delta: { content: text },
              finish_reason: i === generatedIds.length - 1 ? "stop" : null }],
          })}\n\n`);
        }
        reply.raw.write("data: [DONE]\n\n");
        reply.raw.end();
      } else {
        const content = generatedIds.map(id => decoder.decode(id)).join("");
        return {
          id: completionId, object: "chat.completion",
          created, model: model.id,
          choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
          usage: {
            prompt_tokens: inputTokens.length,
            completion_tokens: generatedIds.length,
            total_tokens: inputTokens.length + generatedIds.length,
          },
        };
      }
    } catch (e: any) {
      return reply.status(500).send({ error: "Inference failed", details: e.message });
    }
  });

  const PORT = parseInt(process.env.PORT || "8081", 10);
  await app.listen({ port: PORT, host: "0.0.0.0" });
  console.log(`[1bit] http://0.0.0.0:${PORT} | models: ${models.filter(m=>m.status==="ready").map(m=>m.id).join(", ")}`);
}

main().catch(e => { console.error(e); process.exit(1); });
