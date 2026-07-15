import { ClientSDK, RequestOptions } from "../lib/sdks.js";
import * as components from "../models/components/index.js";
import * as operations from "../models/operations/index.js";
export declare class Spans extends ClientSDK {
    /**
     * Search spans
     */
    searchSpans(request: operations.SearchSpansV1ObservabilitySpansSearchPostRequest, options?: RequestOptions): Promise<components.GetSpans>;
    /**
     * Search span evaluations
     */
    searchSpanEvaluations(request: operations.SearchSpanEvaluationsV1ObservabilitySpansEvaluationsSearchPostRequest, options?: RequestOptions): Promise<components.GetSpanEvaluations>;
    /**
     * Search latest span evaluations
     */
    searchLatestSpanEvaluations(request: operations.SearchLatestSpanEvaluationsV1ObservabilitySpansEvaluationsSearchLatestPostRequest, options?: RequestOptions): Promise<components.GetSpanEvaluations>;
    /**
     * Get span field definitions
     */
    listSpanFields(options?: RequestOptions): Promise<components.GetSpanFields>;
    /**
     * Get span evaluation field definitions
     */
    listSpanEvalFields(options?: RequestOptions): Promise<components.GetSpanEvaluationFields>;
    /**
     * Get options for a span field
     */
    fetchSpanFieldOptions(request: operations.GetSpanFieldOptionsV1ObservabilitySpansFieldsFieldNameOptionsGetRequest, options?: RequestOptions): Promise<components.GetSpanFieldOptions>;
    /**
     * Get options for a span evaluation field
     */
    fetchSpanEvalFieldOptions(request: operations.GetSpanEvaluationFieldOptionsV1ObservabilitySpansEvaluationsFieldsFieldNameOptionsGetRequest, options?: RequestOptions): Promise<components.GetSpanEvaluationFieldOptions>;
}
//# sourceMappingURL=spans.d.ts.map