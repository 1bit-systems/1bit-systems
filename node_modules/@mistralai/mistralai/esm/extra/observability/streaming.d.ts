/**
 * Streaming response helpers for OTEL tracing.
 *
 * Pure functions that parse SSE byte streams and accumulate CompletionChunk
 * deltas into a ChatCompletionResponse-shaped dict suitable for span enrichment.
 *
 * NOTE: The SSE bytes are re-parsed here even though EventStream already
 * parsed them during iteration.
 * TracedResponse sits below EventStream and can only accumulate raw bytes; it
 * has no access to the decoded events. Hooking into EventStream could eliminate
 * this double-parse, but EventStream is Speakeasy-generated code.
 */
import { CompletionChunk } from "../../models/components/completionchunk.js";
/**
 * Parse raw SSE text into a list of typed CompletionChunk models.
 *
 * Only CompletionChunk is handled. If new SSE-streamed response types
 * are added, parsing and typing here will need updating.
 */
export declare function parseSseChunks(rawSseText: string): CompletionChunk[];
/**
 * Accumulate streaming CompletionChunk deltas into a ChatCompletionResponse-shaped dict.
 */
export declare function accumulateChunksToResponseDict(chunks: CompletionChunk[]): Record<string, unknown>;
//# sourceMappingURL=streaming.d.ts.map