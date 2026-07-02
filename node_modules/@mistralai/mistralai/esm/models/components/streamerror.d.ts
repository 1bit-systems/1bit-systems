import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
export type StreamError = {
    error: string;
};
/** @internal */
export declare const StreamError$inboundSchema: z.ZodType<StreamError, unknown>;
export declare function streamErrorFromJSON(jsonString: string): SafeParseResult<StreamError, SDKValidationError>;
//# sourceMappingURL=streamerror.d.ts.map