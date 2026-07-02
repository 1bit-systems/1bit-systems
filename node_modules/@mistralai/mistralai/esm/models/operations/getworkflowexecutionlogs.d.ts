import * as z from "zod/v4";
import { ClosedEnum } from "../../types/enums.js";
/**
 * First-page sort order: 'asc' (oldest first) or 'desc'. Ignored when `cursor` is set.
 */
export declare const GetWorkflowExecutionLogsOrder: {
    readonly Asc: "asc";
    readonly Desc: "desc";
};
/**
 * First-page sort order: 'asc' (oldest first) or 'desc'. Ignored when `cursor` is set.
 */
export type GetWorkflowExecutionLogsOrder = ClosedEnum<typeof GetWorkflowExecutionLogsOrder>;
export type GetWorkflowExecutionLogsRequest = {
    executionId: string;
    /**
     * Filter logs by workflow run ID
     */
    runId?: string | null | undefined;
    /**
     * Filter logs by activity ID
     */
    activityId?: string | null | undefined;
    /**
     * Only return logs at or after this timestamp
     */
    after?: Date | null | undefined;
    /**
     * Only return logs before this timestamp
     */
    before?: Date | null | undefined;
    /**
     * First-page sort order: 'asc' (oldest first) or 'desc'. Ignored when `cursor` is set.
     */
    order?: GetWorkflowExecutionLogsOrder | undefined;
    /**
     * Pagination cursor from a previous response's `next_cursor`; carries the window and order
     */
    cursor?: string | null | undefined;
    /**
     * Maximum number of logs to return
     */
    limit?: number | undefined;
};
/** @internal */
export declare const GetWorkflowExecutionLogsOrder$outboundSchema: z.ZodEnum<typeof GetWorkflowExecutionLogsOrder>;
/** @internal */
export type GetWorkflowExecutionLogsRequest$Outbound = {
    execution_id: string;
    run_id?: string | null | undefined;
    activity_id?: string | null | undefined;
    after?: string | null | undefined;
    before?: string | null | undefined;
    order: string;
    cursor?: string | null | undefined;
    limit: number;
};
/** @internal */
export declare const GetWorkflowExecutionLogsRequest$outboundSchema: z.ZodType<GetWorkflowExecutionLogsRequest$Outbound, GetWorkflowExecutionLogsRequest>;
export declare function getWorkflowExecutionLogsRequestToJSON(getWorkflowExecutionLogsRequest: GetWorkflowExecutionLogsRequest): string;
//# sourceMappingURL=getworkflowexecutionlogs.d.ts.map