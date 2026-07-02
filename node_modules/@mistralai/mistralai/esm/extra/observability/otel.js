/**
 * OTEL conventions for gen AI may be found at:
 *
 * https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
 * https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
 */
import { context as contextApi, propagation, trace, SpanStatusCode, } from "@opentelemetry/api";
import * as semConvAttributes from "@opentelemetry/semantic-conventions/incubating";
import { formatInputMessage, formatOutputMessage, formatToolDefinition, } from "./formatting.js";
import { accumulateChunksToResponseDict, parseSseChunks } from "./streaming.js";
export { semConvAttributes };
export const OTEL_SERVICE_NAME = "mistralai_sdk";
export const MISTRAL_SDK_OTEL_TRACER_NAME = `${OTEL_SERVICE_NAME}_tracer`;
const MISTRAL_AGENT_TRACE_PUBLIC_ATTRIBUTE = "agent.trace.public";
let registeredTracerProvider;
/**
 * Route SDK tracing spans to a specific OpenTelemetry tracer provider.
 */
export function registerTracerProvider(tracerProvider) {
    registeredTracerProvider = tracerProvider;
}
// Safe check for environment variable in both Node.js and browser
const getDebugTracing = () => {
    try {
        if (typeof globalThis !== "undefined" && "process" in globalThis) {
            const proc = globalThis.process;
            return proc?.env?.["MISTRAL_SDK_DEBUG_TRACING"]?.toLowerCase() === "true";
        }
    }
    catch {
        // Ignore
    }
    return false;
};
const MISTRAL_SDK_DEBUG_TRACING = getDebugTracing();
const DEBUG_HINT = "To see detailed tracing logs, set MISTRAL_SDK_DEBUG_TRACING=true.";
export const MistralAIAttributes = {
    MISTRAL_AI_OCR_USAGE_PAGES_PROCESSED: "mistral_ai.ocr.usage.pages_processed",
    MISTRAL_AI_OCR_USAGE_DOC_SIZE_BYTES: "mistral_ai.ocr.usage.doc_size_bytes",
    MISTRAL_AI_ERROR_CODE: "mistral_ai.error.code",
};
export const TracingErrors = {
    FAILED_TO_CREATE_SPAN_FOR_REQUEST: "Failed to create span for request.",
    FAILED_TO_ENRICH_SPAN_WITH_RESPONSE: "Failed to enrich span with response.",
    FAILED_TO_HANDLE_ERROR_IN_SPAN: "Failed to handle error in span.",
    FAILED_TO_END_SPAN: "Failed to end span.",
};
function parseTimeToMillis(ts) {
    return new Date(ts).getTime();
}
function inferGenAiOperationName(operationId) {
    if (operationId.includes("chat_completion") || operationId === "stream_chat") {
        return semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_CHAT;
    }
    if ((operationId.includes("agents_create") || operationId.includes("agents_update")) &&
        !operationId.includes("alias")) {
        return semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_CREATE_AGENT;
    }
    if (operationId.includes("agents_completion") || operationId === "stream_agents") {
        return semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_INVOKE_AGENT;
    }
    if (operationId.includes("conversations") &&
        (operationId.includes("start") ||
            operationId.includes("append") ||
            operationId.includes("restart"))) {
        return semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_INVOKE_AGENT;
    }
    if (operationId.includes("fim")) {
        return semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_TEXT_COMPLETION;
    }
    if (operationId.includes("embeddings")) {
        return semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_EMBEDDINGS;
    }
    if (operationId.includes("ocr_post")) {
        return semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_GENERATE_CONTENT;
    }
    return null;
}
function isKnownGenAiOperation(operationId) {
    return inferGenAiOperationName(operationId) !== null;
}
function isJsonContentType(contentType) {
    const mediaType = contentType?.split(";")[0]?.trim().toLowerCase();
    return mediaType === "application/json" ||
        mediaType === "text/json" ||
        mediaType?.endsWith("+json") === true;
}
function buildGenaiSpanName(genAiOp, body) {
    if (genAiOp === semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_CREATE_AGENT ||
        genAiOp === semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_INVOKE_AGENT) {
        const agentName = body["name"];
        return agentName ? `${genAiOp} ${agentName}` : genAiOp;
    }
    if (genAiOp === semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_EXECUTE_TOOL) {
        const toolName = body["name"];
        return toolName ? `${genAiOp} ${toolName}` : genAiOp;
    }
    const model = body["model"];
    return model ? `${genAiOp} ${model}` : genAiOp;
}
function setAvailableAttributes(span, attributes) {
    for (const [key, value] of Object.entries(attributes)) {
        if (value != null && value !== "") {
            span.setAttribute(key, value);
        }
    }
}
function setMistralAgentTraceMarker(span) {
    // Mistral-internal marker consumed by downstream trace processors.
    span.setAttribute(MISTRAL_AGENT_TRACE_PUBLIC_ATTRIBUTE, "");
}
function setHttpAttributes(span, url, method, host) {
    let port = url.port ? parseInt(url.port, 10) : -1;
    if (port === -1) {
        if (url.protocol === "https:") {
            port = 443;
        }
        else if (url.protocol === "http:") {
            port = 80;
        }
    }
    span.setAttributes({
        [semConvAttributes.ATTR_HTTP_REQUEST_METHOD]: method,
        [semConvAttributes.ATTR_URL_FULL]: url.toString(),
        [semConvAttributes.ATTR_SERVER_ADDRESS]: host,
        [semConvAttributes.ATTR_SERVER_PORT]: port,
    });
}
function enrichRequestGenaiAttrs(span, genAiOp, requestBody) {
    span.updateName(buildGenaiSpanName(genAiOp, requestBody));
    const attributes = {
        [semConvAttributes.ATTR_GEN_AI_REQUEST_CHOICE_COUNT]: requestBody["n"],
        [semConvAttributes.ATTR_GEN_AI_REQUEST_ENCODING_FORMATS]: requestBody["encoding_formats"],
        [semConvAttributes.ATTR_GEN_AI_REQUEST_FREQUENCY_PENALTY]: requestBody["frequency_penalty"],
        [semConvAttributes.ATTR_GEN_AI_REQUEST_MAX_TOKENS]: requestBody["max_tokens"],
        [semConvAttributes.ATTR_GEN_AI_REQUEST_MODEL]: requestBody["model"],
        [semConvAttributes.ATTR_GEN_AI_REQUEST_PRESENCE_PENALTY]: requestBody["presence_penalty"],
        [semConvAttributes.ATTR_GEN_AI_REQUEST_SEED]: requestBody["random_seed"],
        [semConvAttributes.ATTR_GEN_AI_REQUEST_STOP_SEQUENCES]: requestBody["stop"],
        [semConvAttributes.ATTR_GEN_AI_REQUEST_TEMPERATURE]: requestBody["temperature"],
        [semConvAttributes.ATTR_GEN_AI_REQUEST_TOP_P]: requestBody["top_p"],
        [semConvAttributes.ATTR_GEN_AI_REQUEST_TOP_K]: requestBody["top_k"],
    };
    const inputMessages = (requestBody["messages"] || requestBody["inputs"]);
    if (typeof inputMessages === "string") {
        attributes[semConvAttributes.ATTR_GEN_AI_INPUT_MESSAGES] = JSON.stringify([
            formatInputMessage({ role: "user", content: inputMessages }),
        ]);
    }
    else if (Array.isArray(inputMessages)) {
        attributes[semConvAttributes.ATTR_GEN_AI_INPUT_MESSAGES] = JSON.stringify(inputMessages.map((msg) => formatInputMessage(msg)));
    }
    const tools = requestBody["tools"];
    if (tools) {
        const formattedTools = tools
            .map((tool) => formatToolDefinition(tool))
            .filter((t) => t !== null);
        if (formattedTools.length > 0) {
            attributes[semConvAttributes.ATTR_GEN_AI_TOOL_DEFINITIONS] = JSON.stringify(formattedTools);
        }
    }
    // TODO: For agent start conversation, add agent id and version attributes here ?
    setAvailableAttributes(span, attributes);
}
export function enrichSpanFromRequest(span, operationId, url, method, host, body) {
    setHttpAttributes(span, url, method, host);
    const genAiOp = inferGenAiOperationName(operationId);
    if (genAiOp === null) {
        return span;
    }
    span.setAttributes({
        [semConvAttributes.ATTR_GEN_AI_OPERATION_NAME]: genAiOp,
        [semConvAttributes.ATTR_GEN_AI_PROVIDER_NAME]: semConvAttributes.GEN_AI_PROVIDER_NAME_VALUE_MISTRAL_AI,
    });
    if (body) {
        try {
            const requestBody = JSON.parse(body);
            enrichRequestGenaiAttrs(span, genAiOp, requestBody);
        }
        catch {
            // Ignore parse errors
        }
    }
    return span;
}
function enrichResponseGenaiAttrs(span, genAiOp, responseData) {
    const attributes = {};
    if (genAiOp !== semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_CREATE_AGENT) {
        attributes[semConvAttributes.ATTR_GEN_AI_RESPONSE_ID] = responseData["id"];
    }
    attributes[semConvAttributes.ATTR_GEN_AI_RESPONSE_MODEL] = responseData["model"];
    const choices = (responseData["choices"] || []);
    const finishReasons = choices
        .map((c) => c["finish_reason"])
        .filter((r) => !!r);
    if (finishReasons.length > 0) {
        attributes[semConvAttributes.ATTR_GEN_AI_RESPONSE_FINISH_REASONS] = finishReasons;
    }
    if (choices.length > 0) {
        attributes[semConvAttributes.ATTR_GEN_AI_OUTPUT_MESSAGES] = JSON.stringify(choices.map((choice) => formatOutputMessage(choice)));
    }
    const usage = responseData["usage"];
    if (usage) {
        attributes[semConvAttributes.ATTR_GEN_AI_USAGE_INPUT_TOKENS] = usage["prompt_tokens"] || 0;
        attributes[semConvAttributes.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = usage["completion_tokens"] || 0;
    }
    setAvailableAttributes(span, attributes);
}
function enrichCreateAgent(span, responseData) {
    const agentAttributes = {
        [semConvAttributes.ATTR_GEN_AI_AGENT_DESCRIPTION]: responseData["description"],
        [semConvAttributes.ATTR_GEN_AI_AGENT_ID]: responseData["id"],
        [semConvAttributes.ATTR_GEN_AI_AGENT_NAME]: responseData["name"],
        "gen_ai.agent.version": String(responseData["version"]),
        [semConvAttributes.ATTR_GEN_AI_REQUEST_MODEL]: responseData["model"],
        [semConvAttributes.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS]: responseData["instructions"],
    };
    setAvailableAttributes(span, agentAttributes);
}
function stringifyToolAttribute(value) {
    if (typeof value === "string") {
        return value;
    }
    if (value) {
        return JSON.stringify(value);
    }
    return null;
}
function createToolExecutionChildSpan(tracer, parentContext, output) {
    const startMs = parseTimeToMillis(output["created_at"]);
    const endMs = parseTimeToMillis(output["completed_at"]);
    const opName = semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_EXECUTE_TOOL;
    const spanName = buildGenaiSpanName(opName, output);
    const childSpan = tracer.startSpan(spanName, { startTime: startMs }, parentContext);
    setMistralAgentTraceMarker(childSpan);
    const toolArguments = output["arguments"];
    const toolResult = output["info"];
    const toolAttributes = {
        [semConvAttributes.ATTR_GEN_AI_OPERATION_NAME]: opName,
        [semConvAttributes.ATTR_GEN_AI_PROVIDER_NAME]: semConvAttributes.GEN_AI_PROVIDER_NAME_VALUE_MISTRAL_AI,
        [semConvAttributes.ATTR_GEN_AI_TOOL_CALL_ID]: output["id"],
        [semConvAttributes.ATTR_GEN_AI_TOOL_CALL_ARGUMENTS]: stringifyToolAttribute(toolArguments),
        [semConvAttributes.ATTR_GEN_AI_TOOL_CALL_RESULT]: stringifyToolAttribute(toolResult),
        [semConvAttributes.ATTR_GEN_AI_TOOL_NAME]: output["name"],
        [semConvAttributes.ATTR_GEN_AI_TOOL_TYPE]: "extension",
    };
    setAvailableAttributes(childSpan, toolAttributes);
    childSpan.end(endMs);
}
function createMessageOutputChildSpan(tracer, parentContext, output) {
    const startMs = parseTimeToMillis(output["created_at"]);
    const endMs = parseTimeToMillis(output["completed_at"]);
    const opName = semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_CHAT;
    const spanName = buildGenaiSpanName(opName, output);
    const childSpan = tracer.startSpan(spanName, { startTime: startMs }, parentContext);
    setMistralAgentTraceMarker(childSpan);
    const choiceWrapper = {
        message: output,
        finish_reason: output["finish_reason"] || "",
    };
    const messageAttributes = {
        [semConvAttributes.ATTR_GEN_AI_OPERATION_NAME]: opName,
        [semConvAttributes.ATTR_GEN_AI_PROVIDER_NAME]: semConvAttributes.GEN_AI_PROVIDER_NAME_VALUE_MISTRAL_AI,
        [semConvAttributes.ATTR_GEN_AI_RESPONSE_ID]: output["id"],
        [semConvAttributes.ATTR_GEN_AI_AGENT_ID]: output["agent_id"],
        [semConvAttributes.ATTR_GEN_AI_RESPONSE_MODEL]: output["model"],
        [semConvAttributes.ATTR_GEN_AI_OUTPUT_MESSAGES]: JSON.stringify([formatOutputMessage(choiceWrapper)]),
    };
    setAvailableAttributes(childSpan, messageAttributes);
    childSpan.end(endMs);
}
function enrichInvokeAgent(tracer, span, responseData) {
    const conversationAttributes = {
        [semConvAttributes.ATTR_GEN_AI_CONVERSATION_ID]: responseData["conversation_id"],
    };
    setAvailableAttributes(span, conversationAttributes);
    const outputs = (responseData["outputs"] || []);
    const parentContext = trace.setSpan(contextApi.active(), span);
    for (const output of outputs) {
        const outputType = output["type"];
        if (!outputType)
            continue;
        if (outputType === "function.call") {
            continue;
        }
        if (outputType === "tool.execution") {
            createToolExecutionChildSpan(tracer, parentContext, output);
        }
        else if (outputType === "message.output") {
            createMessageOutputChildSpan(tracer, parentContext, output);
        }
    }
}
function enrichOcr(span, responseData) {
    const usageInfo = responseData["usage_info"];
    if (!usageInfo)
        return;
    const ocrAttributes = {
        [MistralAIAttributes.MISTRAL_AI_OCR_USAGE_PAGES_PROCESSED]: usageInfo["pages_processed"],
        [MistralAIAttributes.MISTRAL_AI_OCR_USAGE_DOC_SIZE_BYTES]: usageInfo["doc_size_bytes"],
    };
    setAvailableAttributes(span, ocrAttributes);
}
export function enrichSpanFromResponse(tracer, span, operationId, responseData) {
    const genAiOp = inferGenAiOperationName(operationId);
    if (genAiOp === null) {
        return;
    }
    enrichResponseGenaiAttrs(span, genAiOp, responseData);
    if (genAiOp === semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_CREATE_AGENT) {
        enrichCreateAgent(span, responseData);
    }
    else if (genAiOp === semConvAttributes.GEN_AI_OPERATION_NAME_VALUE_INVOKE_AGENT) {
        enrichInvokeAgent(tracer, span, responseData);
    }
    if (operationId === "ocr_v1_ocr_post") {
        enrichOcr(span, responseData);
    }
}
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
export function getOrCreateOtelTracer() {
    const tracerProvider = registeredTracerProvider ?? trace.getTracerProvider();
    return tracerProvider.getTracer(MISTRAL_SDK_OTEL_TRACER_NAME);
}
export function getSpanContext(span) {
    return trace.setSpan(contextApi.active(), span);
}
export function runWithContext(context, fn) {
    return contextApi.with(context, fn);
}
export async function recordRequestError(context, error) {
    const span = trace.getSpan(context);
    if (span) {
        await getResponseAndError(span, null, error);
    }
}
function warn(error, details) {
    if (MISTRAL_SDK_DEBUG_TRACING) {
        console.warn(error, details);
    }
    else {
        console.warn(error, DEBUG_HINT);
    }
}
export async function getTracedRequestAndSpan(tracer, operationId, request) {
    const span = tracer.startSpan(operationId);
    if (!span.isRecording()) {
        return { request, span, body: null };
    }
    try {
        setMistralAgentTraceMarker(span);
        // Propagate gen_ai.conversation.id from OTEL baggage if present
        const baggage = propagation.getBaggage(contextApi.active());
        const conversationId = baggage?.getEntry(semConvAttributes.ATTR_GEN_AI_CONVERSATION_ID)?.value;
        if (conversationId) {
            span.setAttribute(semConvAttributes.ATTR_GEN_AI_CONVERSATION_ID, conversationId);
        }
        let body = null;
        if (request.body &&
            isKnownGenAiOperation(operationId) &&
            isJsonContentType(request.headers.get("content-type"))) {
            try {
                const cloned = request.clone();
                body = await cloned.text();
            }
            catch {
                // Ignore - body might not be cloneable
            }
        }
        const headers = new Headers(request.headers);
        const headerCarrier = {};
        propagation.inject(trace.setSpan(contextApi.active(), span), headerCarrier);
        for (const [key, value] of Object.entries(headerCarrier)) {
            headers.set(key, value);
        }
        const newRequest = new Request(request, { headers });
        const url = new URL(request.url);
        enrichSpanFromRequest(span, operationId, url, request.method, headers.get("host") || url.host, body);
        return { request: newRequest, span, body };
    }
    catch (err) {
        warn(TracingErrors.FAILED_TO_CREATE_SPAN_FOR_REQUEST, err);
        try {
            span.end();
        }
        catch {
            // Ignore
        }
        return { request, span, body: null };
    }
}
export async function getTracedResponse(tracer, span, operationId, response) {
    if (!span.isRecording()) {
        return response;
    }
    try {
        span.setStatus({ code: SpanStatusCode.OK });
        span.setAttribute(semConvAttributes.ATTR_HTTP_RESPONSE_STATUS_CODE, response.status);
        const knownGenAiOperation = isKnownGenAiOperation(operationId);
        const responseContentType = response.headers.get("content-type");
        const isStreamResponse = knownGenAiOperation &&
            !response.bodyUsed &&
            responseContentType?.toLowerCase().includes("text/event-stream") === true;
        if (isStreamResponse && response.body) {
            return createTracedStreamResponse(response, span, tracer, operationId);
        }
        if (knownGenAiOperation &&
            !response.bodyUsed &&
            isJsonContentType(responseContentType)) {
            const clonedResponse = response.clone();
            try {
                const responseData = await clonedResponse.json();
                enrichSpanFromResponse(tracer, span, operationId, responseData);
            }
            catch {
                // Ignore parse errors
            }
        }
        endSpan(span);
        return response;
    }
    catch (err) {
        warn(TracingErrors.FAILED_TO_ENRICH_SPAN_WITH_RESPONSE, err);
        endSpan(span);
        return response;
    }
}
function createTracedStreamResponse(response, span, tracer, operationId) {
    const originalBody = response.body;
    const chunks = [];
    const decoder = new TextDecoder();
    const reader = originalBody.getReader();
    let finalized = false;
    const finalize = () => {
        if (finalized) {
            return;
        }
        const remaining = decoder.decode();
        if (remaining) {
            chunks.push(remaining);
        }
        finalized = true;
        finalizeStreamSpan(chunks, span, tracer, operationId);
    };
    const tracedStream = new ReadableStream({
        async pull(controller) {
            try {
                const { done, value } = await reader.read();
                if (done) {
                    finalize();
                    controller.close();
                    return;
                }
                chunks.push(decoder.decode(value, { stream: true }));
                controller.enqueue(value);
            }
            catch (err) {
                finalize();
                controller.error(err);
                try {
                    await reader.cancel(err);
                }
                catch {
                    // Ignore cancellation errors from already-failed streams.
                }
            }
        },
        async cancel(reason) {
            finalize();
            await reader.cancel(reason);
        },
    }, { highWaterMark: 0 });
    return new Response(tracedStream, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
    });
}
function finalizeStreamSpan(chunks, span, tracer, operationId) {
    try {
        const fullText = chunks.join("");
        const parsedChunks = parseSseChunks(fullText);
        if (parsedChunks.length > 0) {
            const responseData = accumulateChunksToResponseDict(parsedChunks);
            enrichSpanFromResponse(tracer, span, operationId, responseData);
        }
    }
    catch (err) {
        warn("Failed to enrich span with streaming response.", err);
    }
    endSpan(span);
}
export async function getResponseAndError(span, response, error) {
    if (!span.isRecording()) {
        return { response, error };
    }
    try {
        if (error) {
            span.recordException(error);
            span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
        }
        if (!response) {
            endSpan(span);
            return { response, error };
        }
        try {
            const responseBody = await response.clone().json();
            const body = responseBody;
            if (body["object"] === "error") {
                const errorMsg = body["message"] || "";
                const errorType = body["type"] || "";
                if (errorMsg) {
                    span.setStatus({ code: SpanStatusCode.ERROR, message: errorMsg });
                    span.addEvent("exception", {
                        "exception.type": errorType || "api_error",
                        "exception.message": errorMsg,
                    });
                    const attrs = {
                        [semConvAttributes.ATTR_HTTP_RESPONSE_STATUS_CODE]: response.status,
                    };
                    if (errorType) {
                        attrs[semConvAttributes.ATTR_ERROR_TYPE] = errorType;
                    }
                    if (body["code"]) {
                        attrs[MistralAIAttributes.MISTRAL_AI_ERROR_CODE] = body["code"];
                    }
                    setAvailableAttributes(span, attrs);
                }
            }
        }
        catch {
            // Non-JSON error bodies are valid API responses; leave the span as-is.
        }
        endSpan(span);
    }
    catch (err) {
        warn(TracingErrors.FAILED_TO_HANDLE_ERROR_IN_SPAN, err);
        try {
            span.end();
        }
        catch {
            // Ignore
        }
    }
    return { response, error };
}
function endSpan(span) {
    try {
        span.end();
    }
    catch (err) {
        warn(TracingErrors.FAILED_TO_END_SPAN, err);
    }
}
/**
 * Create a traced span using a callback pattern.
 * This is the TypeScript equivalent of Python's context manager.
 */
export async function traceAsync(name, fn) {
    const tracer = getOrCreateOtelTracer();
    const span = tracer.startSpan(name);
    try {
        return await fn(span);
    }
    finally {
        span.end();
    }
}
//# sourceMappingURL=otel.js.map