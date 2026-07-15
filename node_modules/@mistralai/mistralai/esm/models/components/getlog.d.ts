import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
export type GetLog = {
    customerId: string;
    organizationId: string;
    workspaceId: string;
    userId: string;
    timestamp: Date;
    traceId: string;
    spanId: string;
    traceFlags: number;
    severityText: string;
    severityNumber: number;
    serviceName: string;
    body: string;
    eventName: string;
    resourceSchemaUrl: string;
    resourceAttributes: {
        [k: string]: string;
    };
    scopeSchemaUrl: string;
    scopeName: string;
    scopeVersion: string;
    scopeAttributes: {
        [k: string]: string;
    };
    logAttributes: {
        [k: string]: string;
    };
};
/** @internal */
export declare const GetLog$inboundSchema: z.ZodType<GetLog, unknown>;
export declare function getLogFromJSON(jsonString: string): SafeParseResult<GetLog, SDKValidationError>;
//# sourceMappingURL=getlog.d.ts.map