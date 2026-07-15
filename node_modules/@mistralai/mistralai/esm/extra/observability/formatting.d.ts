/**
 * Formatting helpers for converting Mistral API payloads to OTEL GenAI convention formats.
 *
 * These are pure functions with no OTEL dependencies - they transform objects to objects
 * matching the GenAI semantic convention schemas for input/output messages and tool definitions.
 * The caller is responsible for the final JSON serialization (single JSON.stringify on the whole
 * collection) before setting span attributes.
 *
 * Schemas:
 * - Input messages: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-input-messages.json
 * - Output messages: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-output-messages.json
 * - Tool definitions: https://github.com/Cirilla-zmh/semantic-conventions/blob/cc4d07e7e56b80e9aa5904a3d524c134699da37f/docs/gen-ai/gen-ai-tool-definitions.json
 */
/**
 * Format a single input message per the OTEL GenAI convention.
 *
 * Schema: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-input-messages.json
 * ChatMessage: {role (required), parts (required), name?}
 *
 * Conversation entry objects (e.g. function.result) don't carry a "role"
 * field - they are detected via their "type" and mapped to the closest OTEL role.
 */
export declare function formatInputMessage(message: Record<string, unknown>): Record<string, unknown>;
/**
 * Format a single output choice/message per the OTEL GenAI convention.
 *
 * Schema: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-output-messages.json
 * OutputMessage: {role (required), parts (required), finish_reason (required), name?}
 */
export declare function formatOutputMessage(choice: Record<string, unknown>): Record<string, unknown>;
/**
 * Flatten a Mistral tool definition to the OTEL GenAI convention schema.
 *
 * Mistral format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
 * OTEL format: {"type": "function", "name": ..., "description": ..., "parameters": ...}
 *
 * Schema, still under review: https://github.com/open-telemetry/semantic-conventions
 */
export declare function formatToolDefinition(tool: Record<string, unknown>): Record<string, unknown> | null;
//# sourceMappingURL=formatting.d.ts.map