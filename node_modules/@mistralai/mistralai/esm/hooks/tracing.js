import { HTTPClient } from "../lib/http.js";
let observabilityModule;
async function getObservabilityModule() {
    if (observabilityModule !== undefined) {
        return observabilityModule;
    }
    try {
        observabilityModule = await import("../extra/observability/otel.js");
    }
    catch {
        // OpenTelemetry is an optional peer; without it, tracing is a no-op.
        observabilityModule = null;
    }
    return observabilityModule;
}
export const TRACING_SPAN_KEY = "_tracingSpan";
export const TRACING_BODY_KEY = "_tracingBody";
export const TRACING_TRACER_KEY = "_tracingTracer";
// Runs the actual HTTP send inside the GenAI span context so lower-level
// fetch/http auto-instrumentation parents its spans correctly.
class TracingHTTPClient extends HTTPClient {
    wrappedClient;
    requestContexts;
    constructor(wrappedClient, requestContexts) {
        super();
        this.wrappedClient = wrappedClient;
        this.requestContexts = requestContexts;
    }
    async request(request) {
        const activeContext = this.requestContexts.get(request);
        if (!activeContext) {
            return this.wrappedClient.request(request);
        }
        let observability = null;
        try {
            observability = await getObservabilityModule();
            if (!observability) {
                return await this.wrappedClient.request(request);
            }
            return await observability.runWithContext(activeContext, () => this.wrappedClient.request(request));
        }
        catch (error) {
            if (observability) {
                await observability.recordRequestError(activeContext, error);
            }
            throw error;
        }
        finally {
            this.requestContexts.delete(request);
        }
    }
    clone() {
        return new TracingHTTPClient(this.wrappedClient.clone(), this.requestContexts);
    }
}
export class TracingHook {
    #requestContexts = new WeakMap();
    sdkInit(opts) {
        return {
            ...opts,
            client: new TracingHTTPClient(opts.client, this.#requestContexts),
        };
    }
    async beforeRequest(hookCtx, request) {
        const ctx = hookCtx;
        const observability = await getObservabilityModule();
        if (!observability) {
            return request;
        }
        const tracer = observability.getOrCreateOtelTracer();
        const { request: tracedRequest, span, body } = await observability.getTracedRequestAndSpan(tracer, hookCtx.operationID, request);
        ctx[TRACING_TRACER_KEY] = tracer;
        ctx[TRACING_SPAN_KEY] = span;
        ctx[TRACING_BODY_KEY] = body;
        this.#requestContexts.set(tracedRequest, observability.getSpanContext(span));
        return tracedRequest;
    }
    async afterSuccess(hookCtx, response) {
        const ctx = hookCtx;
        const span = ctx[TRACING_SPAN_KEY];
        const tracer = ctx[TRACING_TRACER_KEY];
        if (!span || !tracer) {
            return response;
        }
        const observability = await getObservabilityModule();
        if (!observability) {
            return response;
        }
        return observability.getTracedResponse(tracer, span, hookCtx.operationID, response);
    }
    async afterError(hookCtx, response, error) {
        const ctx = hookCtx;
        const span = ctx[TRACING_SPAN_KEY];
        if (!span) {
            return { response, error };
        }
        const observability = await getObservabilityModule();
        if (!observability) {
            return { response, error };
        }
        return observability.getResponseAndError(span, response, error);
    }
}
//# sourceMappingURL=tracing.js.map