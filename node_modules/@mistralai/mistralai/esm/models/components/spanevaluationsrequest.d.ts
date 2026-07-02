import * as z from "zod/v4";
export type SpanEvaluationsRequest = {
    searchExpression?: string | null | undefined;
};
/** @internal */
export type SpanEvaluationsRequest$Outbound = {
    search_expression?: string | null | undefined;
};
/** @internal */
export declare const SpanEvaluationsRequest$outboundSchema: z.ZodType<SpanEvaluationsRequest$Outbound, SpanEvaluationsRequest>;
export declare function spanEvaluationsRequestToJSON(spanEvaluationsRequest: SpanEvaluationsRequest): string;
//# sourceMappingURL=spanevaluationsrequest.d.ts.map