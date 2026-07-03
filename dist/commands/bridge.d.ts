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
export {};
//# sourceMappingURL=bridge.d.ts.map