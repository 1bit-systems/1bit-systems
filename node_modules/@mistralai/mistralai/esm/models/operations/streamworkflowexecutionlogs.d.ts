import * as z from "zod/v4";
import { OpenEnum } from "../../types/enums.js";
import { Result as SafeParseResult } from "../../types/fp.js";
import * as components from "../components/index.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
export type StreamWorkflowExecutionLogsRequest = {
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
     * Start a fresh stream at this timestamp (ignored when resuming via last_event_id)
     */
    after?: Date | null | undefined;
    /**
     * Resume from this cursor (a prior response's SSE id)
     */
    lastEventId?: string | null | undefined;
};
export declare const Event: {
    readonly Log: "log";
    readonly Error: "error";
};
export type Event = OpenEnum<typeof Event>;
export type StreamWorkflowExecutionLogsData = components.ExecutionLogRecord | components.StreamError;
/**
 * Stream of Server-Sent Events (SSE): `log` events carry an ExecutionLogRecord; `error` events carry a StreamError payload.
 */
export type StreamWorkflowExecutionLogsResponseBody = {
    event?: Event | undefined;
    id?: string | undefined;
    data?: components.ExecutionLogRecord | components.StreamError | undefined;
};
/** @internal */
export type StreamWorkflowExecutionLogsRequest$Outbound = {
    execution_id: string;
    run_id?: string | null | undefined;
    activity_id?: string | null | undefined;
    after?: string | null | undefined;
    last_event_id?: string | null | undefined;
};
/** @internal */
export declare const StreamWorkflowExecutionLogsRequest$outboundSchema: z.ZodType<StreamWorkflowExecutionLogsRequest$Outbound, StreamWorkflowExecutionLogsRequest>;
export declare function streamWorkflowExecutionLogsRequestToJSON(streamWorkflowExecutionLogsRequest: StreamWorkflowExecutionLogsRequest): string;
/** @internal */
export declare const Event$inboundSchema: z.ZodType<Event, unknown>;
/** @internal */
export declare const StreamWorkflowExecutionLogsData$inboundSchema: z.ZodType<StreamWorkflowExecutionLogsData, unknown>;
export declare function streamWorkflowExecutionLogsDataFromJSON(jsonString: string): SafeParseResult<StreamWorkflowExecutionLogsData, SDKValidationError>;
/** @internal */
export declare const StreamWorkflowExecutionLogsResponseBody$inboundSchema: z.ZodType<StreamWorkflowExecutionLogsResponseBody, unknown>;
export declare function streamWorkflowExecutionLogsResponseBodyFromJSON(jsonString: string): SafeParseResult<StreamWorkflowExecutionLogsResponseBody, SDKValidationError>;
//# sourceMappingURL=streamworkflowexecutionlogs.d.ts.map