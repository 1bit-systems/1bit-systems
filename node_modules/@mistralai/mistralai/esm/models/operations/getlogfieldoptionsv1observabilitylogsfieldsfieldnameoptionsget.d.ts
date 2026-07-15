import * as z from "zod/v4";
export type GetLogFieldOptionsV1ObservabilityLogsFieldsFieldNameOptionsGetRequest = {
    fieldName: string;
    from?: Date | null | undefined;
    to?: Date | null | undefined;
};
/** @internal */
export type GetLogFieldOptionsV1ObservabilityLogsFieldsFieldNameOptionsGetRequest$Outbound = {
    field_name: string;
    from?: string | null | undefined;
    to?: string | null | undefined;
};
/** @internal */
export declare const GetLogFieldOptionsV1ObservabilityLogsFieldsFieldNameOptionsGetRequest$outboundSchema: z.ZodType<GetLogFieldOptionsV1ObservabilityLogsFieldsFieldNameOptionsGetRequest$Outbound, GetLogFieldOptionsV1ObservabilityLogsFieldsFieldNameOptionsGetRequest>;
export declare function getLogFieldOptionsV1ObservabilityLogsFieldsFieldNameOptionsGetRequestToJSON(getLogFieldOptionsV1ObservabilityLogsFieldsFieldNameOptionsGetRequest: GetLogFieldOptionsV1ObservabilityLogsFieldsFieldNameOptionsGetRequest): string;
//# sourceMappingURL=getlogfieldoptionsv1observabilitylogsfieldsfieldnameoptionsget.d.ts.map