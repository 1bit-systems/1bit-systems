import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { OtelFieldDefinition } from "./otelfielddefinition.js";
export type GetSpanFields = {
    fieldDefinitions: Array<OtelFieldDefinition>;
};
/** @internal */
export declare const GetSpanFields$inboundSchema: z.ZodType<GetSpanFields, unknown>;
export declare function getSpanFieldsFromJSON(jsonString: string): SafeParseResult<GetSpanFields, SDKValidationError>;
//# sourceMappingURL=getspanfields.d.ts.map