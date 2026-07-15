import * as z from "zod/v4";
export type GetTraceByIdV1ObservabilityTracesTraceIdGetRequest = {
    traceId: string;
};
/** @internal */
export type GetTraceByIdV1ObservabilityTracesTraceIdGetRequest$Outbound = {
    trace_id: string;
};
/** @internal */
export declare const GetTraceByIdV1ObservabilityTracesTraceIdGetRequest$outboundSchema: z.ZodType<GetTraceByIdV1ObservabilityTracesTraceIdGetRequest$Outbound, GetTraceByIdV1ObservabilityTracesTraceIdGetRequest>;
export declare function getTraceByIdV1ObservabilityTracesTraceIdGetRequestToJSON(getTraceByIdV1ObservabilityTracesTraceIdGetRequest: GetTraceByIdV1ObservabilityTracesTraceIdGetRequest): string;
//# sourceMappingURL=gettracebyidv1observabilitytracestraceidget.d.ts.map