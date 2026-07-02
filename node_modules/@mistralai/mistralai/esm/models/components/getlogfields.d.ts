import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { OtelFieldDefinition } from "./otelfielddefinition.js";
export type GetLogFields = {
    fieldDefinitions: Array<OtelFieldDefinition>;
};
/** @internal */
export declare const GetLogFields$inboundSchema: z.ZodType<GetLogFields, unknown>;
export declare function getLogFieldsFromJSON(jsonString: string): SafeParseResult<GetLogFields, SDKValidationError>;
//# sourceMappingURL=getlogfields.d.ts.map