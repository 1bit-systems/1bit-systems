import * as z from "zod/v4";
import * as components from "../components/index.js";
export type SearchSpansV1ObservabilitySpansSearchPostRequest = {
    from?: Date | null | undefined;
    to?: Date | null | undefined;
    pageSize?: number | undefined;
    cursor?: string | null | undefined;
    spansRequest: components.SpansRequest;
};
/** @internal */
export type SearchSpansV1ObservabilitySpansSearchPostRequest$Outbound = {
    from?: string | null | undefined;
    to?: string | null | undefined;
    page_size: number;
    cursor?: string | null | undefined;
    SpansRequest: components.SpansRequest$Outbound;
};
/** @internal */
export declare const SearchSpansV1ObservabilitySpansSearchPostRequest$outboundSchema: z.ZodType<SearchSpansV1ObservabilitySpansSearchPostRequest$Outbound, SearchSpansV1ObservabilitySpansSearchPostRequest>;
export declare function searchSpansV1ObservabilitySpansSearchPostRequestToJSON(searchSpansV1ObservabilitySpansSearchPostRequest: SearchSpansV1ObservabilitySpansSearchPostRequest): string;
//# sourceMappingURL=searchspansv1observabilityspanssearchpost.d.ts.map