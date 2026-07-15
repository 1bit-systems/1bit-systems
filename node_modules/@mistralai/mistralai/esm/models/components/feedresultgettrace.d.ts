import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { GetTrace } from "./gettrace.js";
export type FeedResultGetTrace = {
    results?: Array<GetTrace> | undefined;
    next?: string | null | undefined;
    cursor?: string | null | undefined;
};
/** @internal */
export declare const FeedResultGetTrace$inboundSchema: z.ZodType<FeedResultGetTrace, unknown>;
export declare function feedResultGetTraceFromJSON(jsonString: string): SafeParseResult<FeedResultGetTrace, SDKValidationError>;
//# sourceMappingURL=feedresultgettrace.d.ts.map