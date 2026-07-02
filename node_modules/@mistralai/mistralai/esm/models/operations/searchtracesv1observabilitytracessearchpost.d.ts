import * as z from "zod/v4";
import * as components from "../components/index.js";
export type SearchTracesV1ObservabilityTracesSearchPostRequest = {
    from?: Date | null | undefined;
    to?: Date | null | undefined;
    pageSize?: number | undefined;
    cursor?: string | null | undefined;
    tracesRequest: components.TracesRequest;
};
/** @internal */
export type SearchTracesV1ObservabilityTracesSearchPostRequest$Outbound = {
    from?: string | null | undefined;
    to?: string | null | undefined;
    page_size: number;
    cursor?: string | null | undefined;
    TracesRequest: components.TracesRequest$Outbound;
};
/** @internal */
export declare const SearchTracesV1ObservabilityTracesSearchPostRequest$outboundSchema: z.ZodType<SearchTracesV1ObservabilityTracesSearchPostRequest$Outbound, SearchTracesV1ObservabilityTracesSearchPostRequest>;
export declare function searchTracesV1ObservabilityTracesSearchPostRequestToJSON(searchTracesV1ObservabilityTracesSearchPostRequest: SearchTracesV1ObservabilityTracesSearchPostRequest): string;
//# sourceMappingURL=searchtracesv1observabilitytracessearchpost.d.ts.map