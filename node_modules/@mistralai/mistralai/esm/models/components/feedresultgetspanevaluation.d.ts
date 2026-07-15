import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { GetSpanEvaluation } from "./getspanevaluation.js";
export type FeedResultGetSpanEvaluation = {
    results?: Array<GetSpanEvaluation> | undefined;
    next?: string | null | undefined;
    cursor?: string | null | undefined;
};
/** @internal */
export declare const FeedResultGetSpanEvaluation$inboundSchema: z.ZodType<FeedResultGetSpanEvaluation, unknown>;
export declare function feedResultGetSpanEvaluationFromJSON(jsonString: string): SafeParseResult<FeedResultGetSpanEvaluation, SDKValidationError>;
//# sourceMappingURL=feedresultgetspanevaluation.d.ts.map