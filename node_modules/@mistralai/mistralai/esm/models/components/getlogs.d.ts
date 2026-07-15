import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { FeedResultGetLog } from "./feedresultgetlog.js";
export type GetLogs = {
    logs: FeedResultGetLog;
};
/** @internal */
export declare const GetLogs$inboundSchema: z.ZodType<GetLogs, unknown>;
export declare function getLogsFromJSON(jsonString: string): SafeParseResult<GetLogs, SDKValidationError>;
//# sourceMappingURL=getlogs.d.ts.map