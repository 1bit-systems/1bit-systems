import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
export type ExecutionLogRecord = {
    timestamp: Date;
    traceId: string;
    spanId: string;
    severityText: string;
    body: string;
    logAttributes: {
        [k: string]: string;
    };
};
/** @internal */
export declare const ExecutionLogRecord$inboundSchema: z.ZodType<ExecutionLogRecord, unknown>;
export declare function executionLogRecordFromJSON(jsonString: string): SafeParseResult<ExecutionLogRecord, SDKValidationError>;
//# sourceMappingURL=executionlogrecord.d.ts.map