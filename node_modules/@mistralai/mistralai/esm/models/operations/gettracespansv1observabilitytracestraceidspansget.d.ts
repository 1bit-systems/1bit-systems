import * as z from "zod/v4";
export type GetTraceSpansV1ObservabilityTracesTraceIdSpansGetRequest = {
    traceId: string;
    from?: Date | null | undefined;
    to?: Date | null | undefined;
    pageSize?: number | undefined;
    cursor?: string | null | undefined;
};
/** @internal */
export type GetTraceSpansV1ObservabilityTracesTraceIdSpansGetRequest$Outbound = {
    trace_id: string;
    from?: string | null | undefined;
    to?: string | null | undefined;
    page_size: number;
    cursor?: string | null | undefined;
};
/** @internal */
export declare const GetTraceSpansV1ObservabilityTracesTraceIdSpansGetRequest$outboundSchema: z.ZodType<GetTraceSpansV1ObservabilityTracesTraceIdSpansGetRequest$Outbound, GetTraceSpansV1ObservabilityTracesTraceIdSpansGetRequest>;
export declare function getTraceSpansV1ObservabilityTracesTraceIdSpansGetRequestToJSON(getTraceSpansV1ObservabilityTracesTraceIdSpansGetRequest: GetTraceSpansV1ObservabilityTracesTraceIdSpansGetRequest): string;
//# sourceMappingURL=gettracespansv1observabilitytracestraceidspansget.d.ts.map