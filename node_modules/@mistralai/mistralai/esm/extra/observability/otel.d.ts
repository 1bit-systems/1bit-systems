/**
 * OTEL conventions for gen AI may be found at:
 *
 * https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
 * https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
 */
import { type Context, type Span, type Tracer, type TracerProvider } from "@opentelemetry/api";
import * as semConvAttributes from "@opentelemetry/semantic-conventions/incubating";
export type { Context, Span, Tracer };
export { semConvAttributes };
export declare const OTEL_SERVICE_NAME = "mistralai_sdk";
export declare const MISTRAL_SDK_OTEL_TRACER_NAME = "mistralai_sdk_tracer";
/**
 * Route SDK tracing spans to a specific OpenTelemetry tracer provider.
 */
export declare function registerTracerProvider(tracerProvider?: TracerProvider): void;
export declare const MistralAIAttributes: {
    readonly MISTRAL_AI_OCR_USAGE_PAGES_PROCESSED: "mistral_ai.ocr.usage.pages_processed";
    readonly MISTRAL_AI_OCR_USAGE_DOC_SIZE_BYTES: "mistral_ai.ocr.usage.doc_size_bytes";
    readonly MISTRAL_AI_ERROR_CODE: "mistral_ai.error.code";
};
export declare const TracingErrors: {
    readonly FAILED_TO_CREATE_SPAN_FOR_REQUEST: "Failed to create span for request.";
    readonly FAILED_TO_ENRICH_SPAN_WITH_RESPONSE: "Failed to enrich span with response.";
    readonly FAILED_TO_HANDLE_ERROR_IN_SPAN: "Failed to handle error in span.";
    readonly FAILED_TO_END_SPAN: "Failed to end span.";
};
export declare function enrichSpanFromRequest(span: Span, operationId: string, url: URL, method: string, host: string, body: string | null): Span;
export declare function enrichSpanFromResponse(tracer: Tracer, span: Span, operationId: string, responseData: Record<string, unknown>): void;
/**
 * Get a tracer from the registered or global TracerProvider.
 *
 * The SDK does not set up its own TracerProvider. It relies on the application
 * to register one explicitly or configure OpenTelemetry's global provider.
 *
 * If no TracerProvider is configured, the ProxyTracerProvider (default) will
 * return a NoOp tracer, effectively disabling tracing. Once the application
 * sets up a real TracerProvider, subsequent spans will be recorded.
 */
export declare function getOrCreateOtelTracer(): Tracer;
export declare function getSpanContext(span: Span): Context;
export declare function runWithContext<T>(context: Context, fn: () => T): T;
export declare function recordRequestError(context: Context, error: unknown): Promise<void>;
export declare function getTracedRequestAndSpan(tracer: Tracer, operationId: string, request: Request): Promise<{
    request: Request;
    span: Span;
    body: string | null;
}>;
export declare function getTracedResponse(tracer: Tracer, span: Span, operationId: string, response: Response): Promise<Response>;
export declare function getResponseAndError(span: Span, response: Response | null, error: unknown): Promise<{
    response: Response | null;
    error: unknown;
}>;
/**
 * Create a traced span using a callback pattern.
 * This is the TypeScript equivalent of Python's context manager.
 */
export declare function traceAsync<T>(name: string, fn: (span: Span) => Promise<T>): Promise<T>;
//# sourceMappingURL=otel.d.ts.map