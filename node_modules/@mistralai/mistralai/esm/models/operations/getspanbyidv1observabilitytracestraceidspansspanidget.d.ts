import * as z from "zod/v4";
export type GetSpanByIdV1ObservabilityTracesTraceIdSpansSpanIdGetRequest = {
    traceId: string;
    spanId: string;
    from?: Date | null | undefined;
    to?: Date | null | undefined;
};
/** @internal */
export type GetSpanByIdV1ObservabilityTracesTraceIdSpansSpanIdGetRequest$Outbound = {
    trace_id: string;
    span_id: string;
    from?: string | null | undefined;
    to?: string | null | undefined;
};
/** @internal */
export declare const GetSpanByIdV1ObservabilityTracesTraceIdSpansSpanIdGetRequest$outboundSchema: z.ZodType<GetSpanByIdV1ObservabilityTracesTraceIdSpansSpanIdGetRequest$Outbound, GetSpanByIdV1ObservabilityTracesTraceIdSpansSpanIdGetRequest>;
export declare function getSpanByIdV1ObservabilityTracesTraceIdSpansSpanIdGetRequestToJSON(getSpanByIdV1ObservabilityTracesTraceIdSpansSpanIdGetRequest: GetSpanByIdV1ObservabilityTracesTraceIdSpansSpanIdGetRequest): string;
//# sourceMappingURL=getspanbyidv1observabilitytracestraceidspansspanidget.d.ts.map