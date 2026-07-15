import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { GetSpan } from "./getspan.js";
export type FeedResultGetSpan = {
    results?: Array<GetSpan> | undefined;
    next?: string | null | undefined;
    cursor?: string | null | undefined;
};
/** @internal */
export declare const FeedResultGetSpan$inboundSchema: z.ZodType<FeedResultGetSpan, unknown>;
export declare function feedResultGetSpanFromJSON(jsonString: string): SafeParseResult<FeedResultGetSpan, SDKValidationError>;
//# sourceMappingURL=feedresultgetspan.d.ts.map