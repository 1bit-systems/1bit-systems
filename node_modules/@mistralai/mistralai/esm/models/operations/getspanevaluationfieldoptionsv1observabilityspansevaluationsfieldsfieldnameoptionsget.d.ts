import * as z from "zod/v4";
export type GetSpanEvaluationFieldOptionsV1ObservabilitySpansEvaluationsFieldsFieldNameOptionsGetRequest = {
    fieldName: string;
    from?: Date | null | undefined;
    to?: Date | null | undefined;
};
/** @internal */
export type GetSpanEvaluationFieldOptionsV1ObservabilitySpansEvaluationsFieldsFieldNameOptionsGetRequest$Outbound = {
    field_name: string;
    from?: string | null | undefined;
    to?: string | null | undefined;
};
/** @internal */
export declare const GetSpanEvaluationFieldOptionsV1ObservabilitySpansEvaluationsFieldsFieldNameOptionsGetRequest$outboundSchema: z.ZodType<GetSpanEvaluationFieldOptionsV1ObservabilitySpansEvaluationsFieldsFieldNameOptionsGetRequest$Outbound, GetSpanEvaluationFieldOptionsV1ObservabilitySpansEvaluationsFieldsFieldNameOptionsGetRequest>;
export declare function getSpanEvaluationFieldOptionsV1ObservabilitySpansEvaluationsFieldsFieldNameOptionsGetRequestToJSON(getSpanEvaluationFieldOptionsV1ObservabilitySpansEvaluationsFieldsFieldNameOptionsGetRequest: GetSpanEvaluationFieldOptionsV1ObservabilitySpansEvaluationsFieldsFieldNameOptionsGetRequest): string;
//# sourceMappingURL=getspanevaluationfieldoptionsv1observabilityspansevaluationsfieldsfieldnameoptionsget.d.ts.map