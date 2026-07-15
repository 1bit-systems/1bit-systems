import * as z from "zod/v4";
export type GetSpanFieldOptionsV1ObservabilitySpansFieldsFieldNameOptionsGetRequest = {
    fieldName: string;
    from?: Date | null | undefined;
    to?: Date | null | undefined;
};
/** @internal */
export type GetSpanFieldOptionsV1ObservabilitySpansFieldsFieldNameOptionsGetRequest$Outbound = {
    field_name: string;
    from?: string | null | undefined;
    to?: string | null | undefined;
};
/** @internal */
export declare const GetSpanFieldOptionsV1ObservabilitySpansFieldsFieldNameOptionsGetRequest$outboundSchema: z.ZodType<GetSpanFieldOptionsV1ObservabilitySpansFieldsFieldNameOptionsGetRequest$Outbound, GetSpanFieldOptionsV1ObservabilitySpansFieldsFieldNameOptionsGetRequest>;
export declare function getSpanFieldOptionsV1ObservabilitySpansFieldsFieldNameOptionsGetRequestToJSON(getSpanFieldOptionsV1ObservabilitySpansFieldsFieldNameOptionsGetRequest: GetSpanFieldOptionsV1ObservabilitySpansFieldsFieldNameOptionsGetRequest): string;
//# sourceMappingURL=getspanfieldoptionsv1observabilityspansfieldsfieldnameoptionsget.d.ts.map