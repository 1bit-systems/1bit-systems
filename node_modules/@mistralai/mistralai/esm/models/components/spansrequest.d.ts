import * as z from "zod/v4";
export type SpansRequest = {
    searchExpression?: string | null | undefined;
};
/** @internal */
export type SpansRequest$Outbound = {
    search_expression?: string | null | undefined;
};
/** @internal */
export declare const SpansRequest$outboundSchema: z.ZodType<SpansRequest$Outbound, SpansRequest>;
export declare function spansRequestToJSON(spansRequest: SpansRequest): string;
//# sourceMappingURL=spansrequest.d.ts.map