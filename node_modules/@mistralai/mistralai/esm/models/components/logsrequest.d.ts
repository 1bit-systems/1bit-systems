import * as z from "zod/v4";
import { ClosedEnum } from "../../types/enums.js";
export declare const Order: {
    readonly Asc: "asc";
    readonly Desc: "desc";
};
export type Order = ClosedEnum<typeof Order>;
export type LogsRequest = {
    searchExpression?: string | null | undefined;
    order?: Order | undefined;
};
/** @internal */
export declare const Order$outboundSchema: z.ZodEnum<typeof Order>;
/** @internal */
export type LogsRequest$Outbound = {
    search_expression?: string | null | undefined;
    order: string;
};
/** @internal */
export declare const LogsRequest$outboundSchema: z.ZodType<LogsRequest$Outbound, LogsRequest>;
export declare function logsRequestToJSON(logsRequest: LogsRequest): string;
//# sourceMappingURL=logsrequest.d.ts.map