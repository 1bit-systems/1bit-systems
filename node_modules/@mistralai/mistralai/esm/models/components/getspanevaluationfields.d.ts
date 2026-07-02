import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { OtelFieldDefinition } from "./otelfielddefinition.js";
export type GetSpanEvaluationFields = {
    fieldDefinitions: Array<OtelFieldDefinition>;
};
/** @internal */
export declare const GetSpanEvaluationFields$inboundSchema: z.ZodType<GetSpanEvaluationFields, unknown>;
export declare function getSpanEvaluationFieldsFromJSON(jsonString: string): SafeParseResult<GetSpanEvaluationFields, SDKValidationError>;
//# sourceMappingURL=getspanevaluationfields.d.ts.map