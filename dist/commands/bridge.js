/**
 * 1bit NPU API Bridge — OpenAI-compatible chat completions server.
 *
 * Start: npx tsx src/commands/bridge.ts
 * Usage: curl -N http://127.0.0.1:9090/v1/chat/completions \
 *            -H "Content-Type: application/json" \
 *            -d '{"model":"qwen3_0_6b","messages":[{"role":"user","content":"hello"}],"stream":true}'
 *
 * Pipeline: text → tokenize → NPU engine → detokenize → SSE stream
 */
import Fastify from "fastify";
import { spawn } from "child_process";
import { readFileSync, existsSync } from "fs";
import { createInterface } from "readline";
import { resolve } from "path";
const HOME = process.env.HOME || "/home/bcloud";
const ENGINE_DIR = resolve(HOME, "1bit-systems-new/engine/npu");
const TOKENIZER_DIR = resolve(ENGINE_DIR, "tokenizer");
// Map of model name -> engine binary + tokenizer + model paths
const MODELS = {
    qwen3_0_6b: {
        engine: resolve(ENGINE_DIR, "build/npu_engine_auto"),
        tokenizer: resolve(HOME, ".config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json"),
        modelPath: resolve(HOME, ".config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"),
        maxTokens: 128,
    },
    qwen3_vl_4b: {
        engine: resolve(ENGINE_DIR, "build/npu_engine_auto"),
        tokenizer: resolve(HOME, ".config/flm/models/Qwen3-VL-4B-Instruct-NPU2/tokenizer.json"),
        modelPath: resolve(HOME, ".config/flm/models/Qwen3-VL-4B-Instruct-NPU2/model.q4nx"),
        maxTokens: 64,
    },
    llama: {
        engine: resolve(ENGINE_DIR, "build/npu_engine_auto"),
        tokenizer: resolve(HOME, ".config/flm/models/Llama-3.1-8B-NPU2/tokenizer.json"),
        modelPath: resolve(HOME, ".config/flm/models/Llama-3.1-8B-NPU2/model.q4nx"),
        maxTokens: 64,
    },
    gemma4_e2b: {
        engine: resolve(ENGINE_DIR, "build/npu_engine_auto"),
        tokenizer: resolve(HOME, ".config/flm/models/Gemma4-E2B-IT-NPU2/tokenizer.json"),
        modelPath: resolve(HOME, ".config/flm/models/Gemma4-E2B-IT-NPU2/model.q4nx"),
        maxTokens: 64,
    },
};
// Simple reverse token lookup: load token -> string mapping from tokenizer.json
function loadDetokenizer(tokenizerPath) {
    const map = new Map();
    try {
        const json = JSON.parse(readFileSync(tokenizerPath, "utf-8"));
        // Try different tokenizer vocab structures
        let vocabSource = null;
        if (json.model?.vocab)
            vocabSource = json.model.vocab;
        if (json.added_tokens && Array.isArray(json.added_tokens)) {
            if (!vocabSource)
                vocabSource = {};
            for (const t of json.added_tokens) {
                if (t.content !== undefined && t.id !== undefined) {
                    map.set(t.id, t.content);
                }
            }
        }
        if (vocabSource && typeof vocabSource === "object") {
            if (Array.isArray(vocabSource)) {
                // Format: [{"id":0,"content":"token"}]
                for (const entry of vocabSource) {
                    if (entry.id !== undefined && entry.content !== undefined) {
                        const content = entry.content
                            .replace(/\\u[0-9a-fA-F]{4}/g, "?")
                            .replace(/\\n/g, "\n")
                            .replace(/\\t/g, "\t");
                        map.set(entry.id, content);
                    }
                }
            }
            else {
                // Format: {"token": id} — reverse it
                for (const [token, id] of Object.entries(vocabSource)) {
                    if (typeof id === "number") {
                        // Unescape the token string
                        let content = token
                            .replace(/\\u[0-9a-fA-F]{4}/g, "?")
                            .replace(/\\n/g, "\n")
                            .replace(/\\t/g, "\t");
                        map.set(id, content);
                    }
                }
            }
        }
    }
    catch (e) {
        console.error("[bridge] Failed to load detokenizer:", e.message);
    }
    return map;
}
async function main() {
    // Pre-load all detokenizers
    const detokenizers = new Map();
    for (const [name, cfg] of Object.entries(MODELS)) {
        if (existsSync(cfg.tokenizer)) {
            detokenizers.set(name, loadDetokenizer(cfg.tokenizer));
            console.log(`[bridge] loaded detokenizer for ${name}`);
        }
    }
    const app = Fastify({ logger: false });
    // Health check
    app.get("/health", async () => ({
        status: "ok",
        models: Object.keys(MODELS),
    }));
    // OpenAI-compatible model listing
    app.get("/v1/models", async () => ({
        object: "list",
        data: Object.entries(MODELS).map(([id]) => ({
            id,
            object: "model",
            created: Math.floor(Date.now() / 1000),
            owned_by: "1bit-systems",
            permission: [],
            root: id,
        })),
    }));
    // Chat completions
    app.post("/v1/chat/completions", async (req, reply) => {
        const body = req.body;
        const modelName = (body?.model || "qwen3_0_6b");
        const messages = body?.messages || [];
        const stream = body?.stream !== false;
        const maxTokens = Math.min(body?.max_tokens || 32, MODELS[modelName]?.maxTokens || 32);
        if (!MODELS[modelName]) {
            reply.code(404).send({ error: `Unknown model: ${modelName}` });
            return;
        }
        const cfg = MODELS[modelName];
        const detok = detokenizers.get(modelName) || new Map();
        if (!existsSync(cfg.engine)) {
            reply
                .code(500)
                .send({ error: `Engine not built: ${cfg.engine}. Run build_npu.sh.` });
            return;
        }
        // Build chat-formatted prompt using Qwen-style template
        let chatPrompt = "";
        for (const msg of messages) {
            if (msg.role === "system") {
                chatPrompt += `<|im_start|>system\n${msg.content}<|im_end|>\n`;
            }
            else if (msg.role === "user") {
                chatPrompt += `<|im_start|>user\n${msg.content}<|im_end|>\n`;
            }
            else if (msg.role === "assistant") {
                chatPrompt += `<|im_start|>assistant\n${msg.content}<|im_end|>\n`;
            }
        }
        chatPrompt += "<|im_start|>assistant\n";
        const promptForLog = chatPrompt.replace(/\n/g, "\\n").slice(0, 100);
        console.log(`[bridge] ${modelName}: prompt="${promptForLog}..." stream=${stream} max_tokens=${maxTokens}`);
        // Tokenize the prompt
        const tokenizerBin = resolve(TOKENIZER_DIR, "tokenize");
        if (!existsSync(tokenizerBin)) {
            reply
                .code(500)
                .send({
                error: "Tokenizer not built. Run make -C engine/npu/tokenizer/",
            });
            return;
        }
        const tokenizerProcess = spawn(tokenizerBin, [cfg.tokenizer]);
        tokenizerProcess.stdin.write(chatPrompt);
        tokenizerProcess.stdin.end();
        // Read token IDs from tokenizer
        const tokenIds = await new Promise((resolve_token, reject) => {
            let output = "";
            tokenizerProcess.stdout.on("data", (chunk) => {
                output += chunk.toString();
            });
            tokenizerProcess.on("close", (code) => {
                if (code === 0)
                    resolve_token(output.trim());
                else
                    reject(new Error(`Tokenizer exited with code ${code}`));
            });
            tokenizerProcess.on("error", reject);
        });
        const tokenCount = tokenIds.split(",").length;
        console.log(`[bridge] tokenized into ${tokenCount} tokens`);
        const responseId = `chatcmpl-${Date.now()}`;
        const created = Math.floor(Date.now() / 1000);
        // Set up SSE headers for streaming
        if (stream) {
            reply.raw.writeHead(200, {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                Connection: "keep-alive",
                "Access-Control-Allow-Origin": "*",
            });
            // Send role chunk
            const roleChunk = {
                id: responseId,
                object: "chat.completion.chunk",
                created,
                model: modelName,
                choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
            };
            reply.raw.write(`data: ${JSON.stringify(roleChunk)}\n\n`);
        }
        // Spawn the NPU engine (auto-detect: argv[1]=model_path argv[2]=decode_tokens argv[3]=input_tokens)
        const modelQ4nx = cfg.modelPath;
        const engineArgs = [
            modelQ4nx, // argv[1]: model.q4nx path
            String(maxTokens), // argv[2]: decode tokens count
            "-", // argv[3]: input tokens from stdin
        ];
        console.log(`[bridge] spawning: ${cfg.engine} ${engineArgs.join(" ")}`);
        const engine = spawn(cfg.engine, engineArgs, {
            cwd: ENGINE_DIR,
            stdio: ["pipe", "pipe", "pipe"],
        });
        // Pipe token IDs to engine stdin
        engine.stdin.write(tokenIds + "\n");
        engine.stdin.end();
        // Parse engine output and stream decoded tokens
        let accumulatedText = "";
        let tokenCount_generated = 0;
        const rl = createInterface({ input: engine.stdout });
        rl.on("line", (line) => {
            // Engine output format: "  [N] token_id (time_ms)"
            const match = line.match(/^\s+\[(\d+)\]\s+(\d+)\s+\((\d+)ms\)/);
            if (match) {
                const tokId = parseInt(match[2], 10);
                const tokText = detok.get(tokId);
                if (tokText) {
                    accumulatedText += tokText;
                    tokenCount_generated++;
                    if (stream) {
                        const chunk = {
                            id: responseId,
                            object: "chat.completion.chunk",
                            created,
                            model: modelName,
                            choices: [
                                {
                                    index: 0,
                                    delta: { content: tokText },
                                    finish_reason: null,
                                },
                            ],
                        };
                        reply.raw.write(`data: ${JSON.stringify(chunk)}\n\n`);
                    }
                }
            }
        });
        // Collect stderr for debugging
        let stderrData = "";
        engine.stderr.on("data", (chunk) => {
            stderrData += chunk.toString();
        });
        await new Promise((resolve_engine) => {
            engine.on("close", (code) => {
                console.log(`[bridge] engine exited code=${code}`);
                if (stderrData)
                    console.log(`[bridge] stderr: ${stderrData.slice(-200)}`);
                // Send final streaming chunks
                if (stream) {
                    const finalChunk = {
                        id: responseId,
                        object: "chat.completion.chunk",
                        created,
                        model: modelName,
                        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
                    };
                    reply.raw.write(`data: ${JSON.stringify(finalChunk)}\n\n`);
                    reply.raw.write("data: [DONE]\n\n");
                    reply.raw.end();
                }
                else {
                    // Non-streaming response
                    reply.send({
                        id: responseId,
                        object: "chat.completion",
                        created,
                        model: modelName,
                        choices: [
                            {
                                index: 0,
                                message: {
                                    role: "assistant",
                                    content: accumulatedText || " ",
                                },
                                finish_reason: "stop",
                            },
                        ],
                        usage: {
                            prompt_tokens: tokenCount,
                            completion_tokens: tokenCount_generated,
                            total_tokens: tokenCount + tokenCount_generated,
                        },
                    });
                }
                resolve_engine();
            });
            engine.on("error", (err) => {
                console.error(`[bridge] engine error: ${err.message}`);
                if (!stream)
                    reply.code(500).send({ error: err.message });
                resolve_engine();
            });
        });
    });
    // Start server
    const PORT = parseInt(process.env.PORT || "9090", 10);
    const HOST = process.env.HOST || "0.0.0.0";
    try {
        await app.listen({ port: PORT, host: HOST });
        console.log(`[bridge] listening on ${HOST}:${PORT}`);
        console.log(`[bridge] models: ${Object.keys(MODELS).join(", ")}`);
    }
    catch (err) {
        console.error(`[bridge] failed to start: ${err.message}`);
        process.exit(1);
    }
}
main();
//# sourceMappingURL=bridge.js.map