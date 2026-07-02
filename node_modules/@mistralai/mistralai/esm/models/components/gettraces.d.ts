import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { FeedResultGetTrace } from "./feedresultgettrace.js";
export type GetTraces = {
    traces: FeedResultGetTrace;
};
/** @internal */
export declare const GetTraces$inboundSchema: z.ZodType<GetTraces, unknown>;
export declare function getTracesFromJSON(jsonString: string): SafeParseResult<GetTraces, SDKValidationError>;
//# sourceMappingURL=gettraces.d.ts.map