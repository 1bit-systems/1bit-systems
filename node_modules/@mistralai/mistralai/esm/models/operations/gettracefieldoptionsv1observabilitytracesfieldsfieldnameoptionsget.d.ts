import * as z from "zod/v4";
export type GetTraceFieldOptionsV1ObservabilityTracesFieldsFieldNameOptionsGetRequest = {
    fieldName: string;
    from?: Date | null | undefined;
    to?: Date | null | undefined;
};
/** @internal */
export type GetTraceFieldOptionsV1ObservabilityTracesFieldsFieldNameOptionsGetRequest$Outbound = {
    field_name: string;
    from?: string | null | undefined;
    to?: string | null | undefined;
};
/** @internal */
export declare const GetTraceFieldOptionsV1ObservabilityTracesFieldsFieldNameOptionsGetRequest$outboundSchema: z.ZodType<GetTraceFieldOptionsV1ObservabilityTracesFieldsFieldNameOptionsGetRequest$Outbound, GetTraceFieldOptionsV1ObservabilityTracesFieldsFieldNameOptionsGetRequest>;
export declare function getTraceFieldOptionsV1ObservabilityTracesFieldsFieldNameOptionsGetRequestToJSON(getTraceFieldOptionsV1ObservabilityTracesFieldsFieldNameOptionsGetRequest: GetTraceFieldOptionsV1ObservabilityTracesFieldsFieldNameOptionsGetRequest): string;
//# sourceMappingURL=gettracefieldoptionsv1observabilitytracesfieldsfieldnameoptionsget.d.ts.map