import * as z from "zod/v4";
import { OpenEnum } from "../../types/enums.js";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
export declare const GetTraceStatusCode: {
    readonly Error: "Error";
    readonly Unset: "Unset";
};
export type GetTraceStatusCode = OpenEnum<typeof GetTraceStatusCode>;
export type GetTrace = {
    customerId: string;
    organizationId: string;
    workspaceId: string;
    userId: string;
    traceId: string;
    rootSpanId: string;
    rootSpanName: string;
    startTime: Date;
    endTime: Date;
    durationNs: number;
    serviceName: string;
    environment: string;
    conversationId: string;
    workflowName: string;
    agentId: string;
    agentName: string;
    statusCode: GetTraceStatusCode;
    errorCount: number;
    spanCount: number;
    genAiSpanCount: number;
    llmCallCount: number;
    toolCallCount: number;
    retrievalCount: number;
    evaluationCount: number;
    inputTokens: number;
    outputTokens: number;
    cacheReadInputTokens: number;
    cacheCreationInputTokens: number;
    modelsUsed: Array<string>;
    toolsUsed: Array<string>;
    firstTurnLastInputMessage: string;
    firstTurnLastOutputMessage: string;
    lastTurnLastInputMessage: string;
    lastTurnLastOutputMessage: string;
};
/** @internal */
export declare const GetTraceStatusCode$inboundSchema: z.ZodType<GetTraceStatusCode, unknown>;
/** @internal */
export declare const GetTrace$inboundSchema: z.ZodType<GetTrace, unknown>;
export declare function getTraceFromJSON(jsonString: string): SafeParseResult<GetTrace, SDKValidationError>;
//# sourceMappingURL=gettrace.d.ts.map