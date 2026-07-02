import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
export type GetSpanFieldOptions = {
    options: Array<string> | null;
};
/** @internal */
export declare const GetSpanFieldOptions$inboundSchema: z.ZodType<GetSpanFieldOptions, unknown>;
export declare function getSpanFieldOptionsFromJSON(jsonString: string): SafeParseResult<GetSpanFieldOptions, SDKValidationError>;
//# sourceMappingURL=getspanfieldoptions.d.ts.map