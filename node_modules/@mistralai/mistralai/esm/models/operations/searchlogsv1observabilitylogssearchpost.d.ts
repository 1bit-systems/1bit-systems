import * as z from "zod/v4";
import * as components from "../components/index.js";
export type SearchLogsV1ObservabilityLogsSearchPostRequest = {
    from?: Date | null | undefined;
    to?: Date | null | undefined;
    pageSize?: number | undefined;
    cursor?: string | null | undefined;
    logsRequest: components.LogsRequest;
};
/** @internal */
export type SearchLogsV1ObservabilityLogsSearchPostRequest$Outbound = {
    from?: string | null | undefined;
    to?: string | null | undefined;
    page_size: number;
    cursor?: string | null | undefined;
    LogsRequest: components.LogsRequest$Outbound;
};
/** @internal */
export declare const SearchLogsV1ObservabilityLogsSearchPostRequest$outboundSchema: z.ZodType<SearchLogsV1ObservabilityLogsSearchPostRequest$Outbound, SearchLogsV1ObservabilityLogsSearchPostRequest>;
export declare function searchLogsV1ObservabilityLogsSearchPostRequestToJSON(searchLogsV1ObservabilityLogsSearchPostRequest: SearchLogsV1ObservabilityLogsSearchPostRequest): string;
//# sourceMappingURL=searchlogsv1observabilitylogssearchpost.d.ts.map