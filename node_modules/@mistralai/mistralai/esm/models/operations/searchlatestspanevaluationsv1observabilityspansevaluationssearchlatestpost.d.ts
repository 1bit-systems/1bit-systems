import * as z from "zod/v4";
import * as components from "../components/index.js";
export type SearchLatestSpanEvaluationsV1ObservabilitySpansEvaluationsSearchLatestPostRequest = {
    from?: Date | null | undefined;
    to?: Date | null | undefined;
    pageSize?: number | undefined;
    cursor?: string | null | undefined;
    spanEvaluationsRequest: components.SpanEvaluationsRequest;
};
/** @internal */
export type SearchLatestSpanEvaluationsV1ObservabilitySpansEvaluationsSearchLatestPostRequest$Outbound = {
    from?: string | null | undefined;
    to?: string | null | undefined;
    page_size: number;
    cursor?: string | null | undefined;
    SpanEvaluationsRequest: components.SpanEvaluationsRequest$Outbound;
};
/** @internal */
export declare const SearchLatestSpanEvaluationsV1ObservabilitySpansEvaluationsSearchLatestPostRequest$outboundSchema: z.ZodType<SearchLatestSpanEvaluationsV1ObservabilitySpansEvaluationsSearchLatestPostRequest$Outbound, SearchLatestSpanEvaluationsV1ObservabilitySpansEvaluationsSearchLatestPostRequest>;
export declare function searchLatestSpanEvaluationsV1ObservabilitySpansEvaluationsSearchLatestPostRequestToJSON(searchLatestSpanEvaluationsV1ObservabilitySpansEvaluationsSearchLatestPostRequest: SearchLatestSpanEvaluationsV1ObservabilitySpansEvaluationsSearchLatestPostRequest): string;
//# sourceMappingURL=searchlatestspanevaluationsv1observabilityspansevaluationssearchlatestpost.d.ts.map