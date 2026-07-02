import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { FeedResultGetSpan } from "./feedresultgetspan.js";
export type GetSpans = {
    spans: FeedResultGetSpan;
};
/** @internal */
export declare const GetSpans$inboundSchema: z.ZodType<GetSpans, unknown>;
export declare function getSpansFromJSON(jsonString: string): SafeParseResult<GetSpans, SDKValidationError>;
//# sourceMappingURL=getspans.d.ts.map