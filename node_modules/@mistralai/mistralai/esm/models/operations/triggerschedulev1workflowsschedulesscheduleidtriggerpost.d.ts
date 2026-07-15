import * as z from "zod/v4";
import * as components from "../components/index.js";
export type TriggerScheduleV1WorkflowsSchedulesScheduleIdTriggerPostRequest = {
    scheduleId: string;
    workflowScheduleTriggerRequest?: components.WorkflowScheduleTriggerRequest | null | undefined;
};
/** @internal */
export type TriggerScheduleV1WorkflowsSchedulesScheduleIdTriggerPostRequest$Outbound = {
    schedule_id: string;
    WorkflowScheduleTriggerRequest?: components.WorkflowScheduleTriggerRequest$Outbound | null | undefined;
};
/** @internal */
export declare const TriggerScheduleV1WorkflowsSchedulesScheduleIdTriggerPostRequest$outboundSchema: z.ZodType<TriggerScheduleV1WorkflowsSchedulesScheduleIdTriggerPostRequest$Outbound, TriggerScheduleV1WorkflowsSchedulesScheduleIdTriggerPostRequest>;
export declare function triggerScheduleV1WorkflowsSchedulesScheduleIdTriggerPostRequestToJSON(triggerScheduleV1WorkflowsSchedulesScheduleIdTriggerPostRequest: TriggerScheduleV1WorkflowsSchedulesScheduleIdTriggerPostRequest): string;
//# sourceMappingURL=triggerschedulev1workflowsschedulesscheduleidtriggerpost.d.ts.map