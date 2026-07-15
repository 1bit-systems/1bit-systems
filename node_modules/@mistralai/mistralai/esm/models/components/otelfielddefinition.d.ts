import * as z from "zod/v4";
import { OpenEnum } from "../../types/enums.js";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
export declare const OtelFieldDefinitionType: {
    readonly Enum: "ENUM";
    readonly Text: "TEXT";
    readonly Int: "INT";
    readonly Float: "FLOAT";
    readonly Bool: "BOOL";
    readonly Timestamp: "TIMESTAMP";
    readonly Array: "ARRAY";
    readonly Map: "MAP";
};
export type OtelFieldDefinitionType = OpenEnum<typeof OtelFieldDefinitionType>;
export declare const OtelFieldDefinitionSupportedOperator: {
    readonly Eq: "eq";
    readonly Neq: "neq";
    readonly Lt: "lt";
    readonly Lte: "lte";
    readonly Gt: "gt";
    readonly Gte: "gte";
    readonly Like: "like";
    readonly Ilike: "ilike";
    readonly NotLike: "not_like";
    readonly NotIlike: "not_ilike";
    readonly Between: "between";
    readonly NotBetween: "not_between";
    readonly In: "in";
    readonly NotIn: "not_in";
    readonly Exists: "exists";
    readonly NotExists: "not_exists";
    readonly Regexp: "regexp";
    readonly NotRegexp: "not_regexp";
    readonly Contains: "contains";
    readonly NotContains: "not_contains";
    readonly Has: "has";
    readonly HasAny: "hasAny";
    readonly HasAll: "hasAll";
    readonly HasToken: "hasToken";
};
export type OtelFieldDefinitionSupportedOperator = OpenEnum<typeof OtelFieldDefinitionSupportedOperator>;
export type OtelFieldDefinition = {
    name: string;
    label: string;
    type: OtelFieldDefinitionType;
    group?: string | null | undefined;
    supportedOperators: Array<OtelFieldDefinitionSupportedOperator>;
};
/** @internal */
export declare const OtelFieldDefinitionType$inboundSchema: z.ZodType<OtelFieldDefinitionType, unknown>;
/** @internal */
export declare const OtelFieldDefinitionSupportedOperator$inboundSchema: z.ZodType<OtelFieldDefinitionSupportedOperator, unknown>;
/** @internal */
export declare const OtelFieldDefinition$inboundSchema: z.ZodType<OtelFieldDefinition, unknown>;
export declare function otelFieldDefinitionFromJSON(jsonString: string): SafeParseResult<OtelFieldDefinition, SDKValidationError>;
//# sourceMappingURL=otelfielddefinition.d.ts.map