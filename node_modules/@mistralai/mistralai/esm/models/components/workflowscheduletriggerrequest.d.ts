import * as z from "zod/v4";
import { ScheduleOverlapPolicy } from "./scheduleoverlappolicy.js";
export type WorkflowScheduleTriggerRequest = {
    /**
     * Optional overlap policy override to use for the immediate trigger.
     */
    overlap?: ScheduleOverlapPolicy | null | undefined;
};
/** @internal */
export type WorkflowScheduleTriggerRequest$Outbound = {
    overlap?: number | null | undefined;
};
/** @internal */
export declare const WorkflowScheduleTriggerRequest$outboundSchema: z.ZodType<WorkflowScheduleTriggerRequest$Outbound, WorkflowScheduleTriggerRequest>;
export declare function workflowScheduleTriggerRequestToJSON(workflowScheduleTriggerRequest: WorkflowScheduleTriggerRequest): string;
//# sourceMappingURL=workflowscheduletriggerrequest.d.ts.map