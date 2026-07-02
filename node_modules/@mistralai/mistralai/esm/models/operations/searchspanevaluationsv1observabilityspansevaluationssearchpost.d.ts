import * as z from "zod/v4";
import * as components from "../components/index.js";
export type SearchSpanEvaluationsV1ObservabilitySpansEvaluationsSearchPostRequest = {
    from?: Date | null | undefined;
    to?: Date | null | undefined;
    pageSize?: number | undefined;
    cursor?: string | null | undefined;
    spanEvaluationsRequest: components.SpanEvaluationsRequest;
};
/** @internal */
export type SearchSpanEvaluationsV1ObservabilitySpansEvaluationsSearchPostRequest$Outbound = {
    from?: string | null | undefined;
    to?: string | null | undefined;
    page_size: number;
    cursor?: string | null | undefined;
    SpanEvaluationsRequest: components.SpanEvaluationsRequest$Outbound;
};
/** @internal */
export declare const SearchSpanEvaluationsV1ObservabilitySpansEvaluationsSearchPostRequest$outboundSchema: z.ZodType<SearchSpanEvaluationsV1ObservabilitySpansEvaluationsSearchPostRequest$Outbound, SearchSpanEvaluationsV1ObservabilitySpansEvaluationsSearchPostRequest>;
export declare function searchSpanEvaluationsV1ObservabilitySpansEvaluationsSearchPostRequestToJSON(searchSpanEvaluationsV1ObservabilitySpansEvaluationsSearchPostRequest: SearchSpanEvaluationsV1ObservabilitySpansEvaluationsSearchPostRequest): string;
//# sourceMappingURL=searchspanevaluationsv1observabilityspansevaluationssearchpost.d.ts.map