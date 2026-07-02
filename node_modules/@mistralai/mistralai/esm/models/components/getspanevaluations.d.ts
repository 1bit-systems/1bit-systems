import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { FeedResultGetSpanEvaluation } from "./feedresultgetspanevaluation.js";
export type GetSpanEvaluations = {
    spanEvaluations: FeedResultGetSpanEvaluation;
};
/** @internal */
export declare const GetSpanEvaluations$inboundSchema: z.ZodType<GetSpanEvaluations, unknown>;
export declare function getSpanEvaluationsFromJSON(jsonString: string): SafeParseResult<GetSpanEvaluations, SDKValidationError>;
//# sourceMappingURL=getspanevaluations.d.ts.map