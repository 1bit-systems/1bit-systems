import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
export type GetTraceFieldOptions = {
    options: Array<string> | null;
};
/** @internal */
export declare const GetTraceFieldOptions$inboundSchema: z.ZodType<GetTraceFieldOptions, unknown>;
export declare function getTraceFieldOptionsFromJSON(jsonString: string): SafeParseResult<GetTraceFieldOptions, SDKValidationError>;
//# sourceMappingURL=gettracefieldoptions.d.ts.map