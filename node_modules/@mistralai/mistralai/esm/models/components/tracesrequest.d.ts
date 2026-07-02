import * as z from "zod/v4";
export type TracesRequest = {
    searchExpression?: string | null | undefined;
};
/** @internal */
export type TracesRequest$Outbound = {
    search_expression?: string | null | undefined;
};
/** @internal */
export declare const TracesRequest$outboundSchema: z.ZodType<TracesRequest$Outbound, TracesRequest>;
export declare function tracesRequestToJSON(tracesRequest: TracesRequest): string;
//# sourceMappingURL=tracesrequest.d.ts.map