import * as z from "zod/v4";
export type ListDeploymentsV1WorkflowsDeploymentsGetRequest = {
    activeOnly?: boolean | undefined;
    /**
     * Filter deployments by hardened status
     */
    isHardened?: boolean | null | undefined;
    workflowName?: string | null | undefined;
    /**
     * Filter deployments by name or ID prefix
     */
    search?: string | null | undefined;
    /**
     * Maximum number of deployments to return
     */
    limit?: number | null | undefined;
    /**
     * Cursor from a previous response for pagination
     */
    cursor?: string | null | undefined;
    /**
     * Workspace ID to scope the request to. Defaults to the caller's context.
     */
    workspaceId?: string | null | undefined;
};
/** @internal */
export type ListDeploymentsV1WorkflowsDeploymentsGetRequest$Outbound = {
    active_only: boolean;
    is_hardened?: boolean | null | undefined;
    workflow_name?: string | null | undefined;
    search?: string | null | undefined;
    limit?: number | null | undefined;
    cursor?: string | null | undefined;
    workspace_id?: string | null | undefined;
};
/** @internal */
export declare const ListDeploymentsV1WorkflowsDeploymentsGetRequest$outboundSchema: z.ZodType<ListDeploymentsV1WorkflowsDeploymentsGetRequest$Outbound, ListDeploymentsV1WorkflowsDeploymentsGetRequest>;
export declare function listDeploymentsV1WorkflowsDeploymentsGetRequestToJSON(listDeploymentsV1WorkflowsDeploymentsGetRequest: ListDeploymentsV1WorkflowsDeploymentsGetRequest): string;
//# sourceMappingURL=listdeploymentsv1workflowsdeploymentsget.d.ts.map