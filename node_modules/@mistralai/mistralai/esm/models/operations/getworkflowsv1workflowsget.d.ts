import * as z from "zod/v4";
import { ClosedEnum } from "../../types/enums.js";
import { Result as SafeParseResult } from "../../types/fp.js";
import * as components from "../components/index.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
/**
 * Filter by workflow status
 */
export type GetWorkflowsV1WorkflowsGetStatus = components.WorkflowExecutionStatus | Array<components.WorkflowExecutionStatus>;
/**
 * Filter by deployment activity. active=only active, inactive=only inactive, None=no filter
 */
export declare const DeploymentStatus: {
    readonly Active: "active";
    readonly Inactive: "inactive";
};
/**
 * Filter by deployment activity. active=only active, inactive=only inactive, None=no filter
 */
export type DeploymentStatus = ClosedEnum<typeof DeploymentStatus>;
/**
 * Sort direction
 */
export declare const GetWorkflowsV1WorkflowsGetOrder: {
    readonly Asc: "asc";
    readonly Desc: "desc";
};
/**
 * Sort direction
 */
export type GetWorkflowsV1WorkflowsGetOrder = ClosedEnum<typeof GetWorkflowsV1WorkflowsGetOrder>;
export type GetWorkflowsV1WorkflowsGetRequest = {
    /**
     * Filter by workflow status
     */
    status?: components.WorkflowExecutionStatus | Array<components.WorkflowExecutionStatus> | null | undefined;
    /**
     * Whether to include shared workflows
     */
    includeShared?: boolean | undefined;
    /**
     * Whether to only return workflows available in chat assistant
     */
    availableInChatAssistant?: boolean | null | undefined;
    /**
     * Filter by deployment name(s)
     */
    deploymentName?: Array<string> | null | undefined;
    /**
     * Filter by deployment activity. active=only active, inactive=only inactive, None=no filter
     */
    deploymentStatus?: DeploymentStatus | null | undefined;
    /**
     * Filter by archived state. False=exclude archived, True=only archived, None=include all
     */
    archived?: boolean | null | undefined;
    /**
     * Filter to workflows tagged with all listed tags (AND).
     */
    tags?: Array<string> | null | undefined;
    /**
     * Field to sort by
     */
    sortBy?: "display_name" | null | undefined;
    /**
     * Sort direction
     */
    order?: GetWorkflowsV1WorkflowsGetOrder | undefined;
    /**
     * The cursor for pagination
     */
    cursor?: string | null | undefined;
    /**
     * The maximum number of workflows to return
     */
    limit?: number | undefined;
    /**
     * Deprecated: use deployment_status instead
     *
     * @deprecated field: This will be removed in a future release, please migrate away from it as soon as possible.
     */
    activeOnly?: boolean | undefined;
};
export type GetWorkflowsV1WorkflowsGetResponse = {
    result: components.WorkflowListResponse;
};
/** @internal */
export type GetWorkflowsV1WorkflowsGetStatus$Outbound = string | Array<string>;
/** @internal */
export declare const GetWorkflowsV1WorkflowsGetStatus$outboundSchema: z.ZodType<GetWorkflowsV1WorkflowsGetStatus$Outbound, GetWorkflowsV1WorkflowsGetStatus>;
export declare function getWorkflowsV1WorkflowsGetStatusToJSON(getWorkflowsV1WorkflowsGetStatus: GetWorkflowsV1WorkflowsGetStatus): string;
/** @internal */
export declare const DeploymentStatus$outboundSchema: z.ZodEnum<typeof DeploymentStatus>;
/** @internal */
export declare const GetWorkflowsV1WorkflowsGetOrder$outboundSchema: z.ZodEnum<typeof GetWorkflowsV1WorkflowsGetOrder>;
/** @internal */
export type GetWorkflowsV1WorkflowsGetRequest$Outbound = {
    status?: string | Array<string> | null | undefined;
    include_shared: boolean;
    available_in_chat_assistant?: boolean | null | undefined;
    deployment_name?: Array<string> | null | undefined;
    deployment_status?: string | null | undefined;
    archived?: boolean | null | undefined;
    tags?: Array<string> | null | undefined;
    sort_by?: "display_name" | null | undefined;
    order: string;
    cursor?: string | null | undefined;
    limit: number;
    active_only: boolean;
};
/** @internal */
export declare const GetWorkflowsV1WorkflowsGetRequest$outboundSchema: z.ZodType<GetWorkflowsV1WorkflowsGetRequest$Outbound, GetWorkflowsV1WorkflowsGetRequest>;
export declare function getWorkflowsV1WorkflowsGetRequestToJSON(getWorkflowsV1WorkflowsGetRequest: GetWorkflowsV1WorkflowsGetRequest): string;
/** @internal */
export declare const GetWorkflowsV1WorkflowsGetResponse$inboundSchema: z.ZodType<GetWorkflowsV1WorkflowsGetResponse, unknown>;
export declare function getWorkflowsV1WorkflowsGetResponseFromJSON(jsonString: string): SafeParseResult<GetWorkflowsV1WorkflowsGetResponse, SDKValidationError>;
//# sourceMappingURL=getworkflowsv1workflowsget.d.ts.map