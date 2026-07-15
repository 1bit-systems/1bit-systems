import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
export type GetSpanEvaluationFieldOptions = {
    options: Array<string> | null;
};
/** @internal */
export declare const GetSpanEvaluationFieldOptions$inboundSchema: z.ZodType<GetSpanEvaluationFieldOptions, unknown>;
export declare function getSpanEvaluationFieldOptionsFromJSON(jsonString: string): SafeParseResult<GetSpanEvaluationFieldOptions, SDKValidationError>;
//# sourceMappingURL=getspanevaluationfieldoptions.d.ts.map