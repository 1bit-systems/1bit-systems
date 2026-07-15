import { EventStream } from "../lib/event-streams.js";
import { ClientSDK, RequestOptions } from "../lib/sdks.js";
import * as components from "../models/components/index.js";
import * as operations from "../models/operations/index.js";
export declare class Executions extends ClientSDK {
    /**
     * Get Workflow Execution
     */
    getWorkflowExecution(request: operations.GetWorkflowExecutionV1WorkflowsExecutionsExecutionIdGetRequest, options?: RequestOptions): Promise<components.WorkflowExecutionResponse>;
    /**
     * Get Workflow Execution History
     */
    getWorkflowExecutionHistory(request: operations.GetWorkflowExecutionHistoryV1WorkflowsExecutionsExecutionIdHistoryGetRequest, options?: RequestOptions): Promise<any>;
    /**
     * Signal Workflow Execution
     */
    signalWorkflowExecution(request: operations.SignalWorkflowExecutionV1WorkflowsExecutionsExecutionIdSignalsPostRequest, options?: RequestOptions): Promise<components.SignalWorkflowResponse>;
    /**
     * Query Workflow Execution
     */
    queryWorkflowExecution(request: operations.QueryWorkflowExecutionV1WorkflowsExecutionsExecutionIdQueriesPostRequest, options?: RequestOptions): Promise<components.QueryWorkflowResponse>;
    /**
     * Terminate Workflow Execution
     */
    terminateWorkflowExecution(request: operations.TerminateWorkflowExecutionV1WorkflowsExecutionsExecutionIdTerminatePostRequest, options?: RequestOptions): Promise<void>;
    /**
     * Batch Terminate Workflow Executions
     */
    batchTerminateWorkflowExecutions(request: components.BatchExecutionBody, options?: RequestOptions): Promise<components.BatchExecutionResponse>;
    /**
     * Cancel Workflow Execution
     */
    cancelWorkflowExecution(request: operations.CancelWorkflowExecutionV1WorkflowsExecutionsExecutionIdCancelPostRequest, options?: RequestOptions): Promise<void>;
    /**
     * Batch Cancel Workflow Executions
     */
    batchCancelWorkflowExecutions(request: components.BatchExecutionBody, options?: RequestOptions): Promise<components.BatchExecutionResponse>;
    /**
     * Reset Workflow
     */
    resetWorkflow(request: operations.ResetWorkflowV1WorkflowsExecutionsExecutionIdResetPostRequest, options?: RequestOptions): Promise<void>;
    /**
     * Update Workflow Execution
     */
    updateWorkflowExecution(request: operations.UpdateWorkflowExecutionV1WorkflowsExecutionsExecutionIdUpdatesPostRequest, options?: RequestOptions): Promise<components.UpdateWorkflowResponse>;
    /**
     * Get Workflow Execution Trace Otel
     */
    getWorkflowExecutionTraceOtel(request: operations.GetWorkflowExecutionTraceOtelRequest, options?: RequestOptions): Promise<components.WorkflowExecutionTraceOTelResponse>;
    /**
     * Get Workflow Execution Trace Summary
     */
    getWorkflowExecutionTraceSummary(request: operations.GetWorkflowExecutionTraceSummaryRequest, options?: RequestOptions): Promise<components.WorkflowExecutionTraceSummaryResponse>;
    /**
     * Get Workflow Execution Trace Events
     */
    getWorkflowExecutionTraceEvents(request: operations.GetWorkflowExecutionTraceEventsRequest, options?: RequestOptions): Promise<components.WorkflowExecutionTraceEventsResponse>;
    /**
     * Stream
     */
    stream(request: operations.StreamV1WorkflowsExecutionsExecutionIdStreamGetRequest, options?: RequestOptions): Promise<EventStream<operations.StreamV1WorkflowsExecutionsExecutionIdStreamGetResponseBody>>;
    /**
     * Get Workflow Execution Logs
     *
     * @remarks
     * Retrieve logs for a workflow execution from Dora.
     *
     * First page sets the window via `after`/`before` (default: execution start through now, both
     * widened by a margin so the bounds still prune partitions); later pages pass `cursor`, which
     * carries both the window and the sort order (so `after`/`before`/`order` are then ignored —
     * the order is fixed at the first page so a client can't flip direction mid-pagination).
     */
    getWorkflowExecutionLogs(request: operations.GetWorkflowExecutionLogsRequest, options?: RequestOptions): Promise<components.ExecutionLogSearchResponse>;
    /**
     * Stream Workflow Execution Logs
     *
     * @remarks
     * Stream logs for a workflow execution via SSE.
     *
     * If `last_event_id` is set it resumes from that cursor and takes precedence over `after`;
     * otherwise `after` sets a fresh stream's start point (omit both to tail from the execution start).
     */
    streamWorkflowExecutionLogs(request: operations.StreamWorkflowExecutionLogsRequest, options?: RequestOptions): Promise<EventStream<operations.StreamWorkflowExecutionLogsResponseBody>>;
}
//# sourceMappingURL=executions.d.ts.map