import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { OtelFieldDefinition } from "./otelfielddefinition.js";
export type GetTraceFields = {
    fieldDefinitions: Array<OtelFieldDefinition>;
};
/** @internal */
export declare const GetTraceFields$inboundSchema: z.ZodType<GetTraceFields, unknown>;
export declare function getTraceFieldsFromJSON(jsonString: string): SafeParseResult<GetTraceFields, SDKValidationError>;
//# sourceMappingURL=gettracefields.d.ts.map