import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { ExecutionLogRecord } from "./executionlogrecord.js";
export type ExecutionLogSearchResponse = {
    results: Array<ExecutionLogRecord>;
    nextCursor?: string | null | undefined;
};
/** @internal */
export declare const ExecutionLogSearchResponse$inboundSchema: z.ZodType<ExecutionLogSearchResponse, unknown>;
export declare function executionLogSearchResponseFromJSON(jsonString: string): SafeParseResult<ExecutionLogSearchResponse, SDKValidationError>;
//# sourceMappingURL=executionlogsearchresponse.d.ts.map