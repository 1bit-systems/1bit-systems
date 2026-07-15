import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
export type GetLogFieldOptions = {
    options: Array<string> | null;
};
/** @internal */
export declare const GetLogFieldOptions$inboundSchema: z.ZodType<GetLogFieldOptions, unknown>;
export declare function getLogFieldOptionsFromJSON(jsonString: string): SafeParseResult<GetLogFieldOptions, SDKValidationError>;
//# sourceMappingURL=getlogfieldoptions.d.ts.map