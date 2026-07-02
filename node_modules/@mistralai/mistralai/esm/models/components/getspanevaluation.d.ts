import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
export type GetSpanEvaluation = {
    customerId: string;
    organizationId: string;
    workspaceId: string;
    userId: string;
    traceId: string;
    spanId: string;
    responseId: string;
    conversationId: string;
    timestamp: Date;
    evaluationName: string;
    scoreValue: number;
    scoreLabel: string;
    explanation: string;
    metadata: {
        [k: string]: string;
    };
};
/** @internal */
export declare const GetSpanEvaluation$inboundSchema: z.ZodType<GetSpanEvaluation, unknown>;
export declare function getSpanEvaluationFromJSON(jsonString: string): SafeParseResult<GetSpanEvaluation, SDKValidationError>;
//# sourceMappingURL=getspanevaluation.d.ts.map