import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { GetLog } from "./getlog.js";
export type FeedResultGetLog = {
    results?: Array<GetLog> | undefined;
    next?: string | null | undefined;
    cursor?: string | null | undefined;
};
/** @internal */
export declare const FeedResultGetLog$inboundSchema: z.ZodType<FeedResultGetLog, unknown>;
export declare function feedResultGetLogFromJSON(jsonString: string): SafeParseResult<FeedResultGetLog, SDKValidationError>;
//# sourceMappingURL=feedresultgetlog.d.ts.map