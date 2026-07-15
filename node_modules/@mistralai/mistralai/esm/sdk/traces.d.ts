import { ClientSDK, RequestOptions } from "../lib/sdks.js";
import * as components from "../models/components/index.js";
import * as operations from "../models/operations/index.js";
export declare class Traces extends ClientSDK {
    /**
     * Search traces
     */
    search(request: operations.SearchTracesV1ObservabilityTracesSearchPostRequest, options?: RequestOptions): Promise<components.GetTraces>;
    /**
     * Get trace field definitions
     */
    getTraceFields(options?: RequestOptions): Promise<components.GetTraceFields>;
    /**
     * Get trace by id
     */
    getTraceById(request: operations.GetTraceByIdV1ObservabilityTracesTraceIdGetRequest, options?: RequestOptions): Promise<components.GetTrace>;
    /**
     * Get trace spans
     */
    getTraceSpans(request: operations.GetTraceSpansV1ObservabilityTracesTraceIdSpansGetRequest, options?: RequestOptions): Promise<components.GetSpans>;
    /**
     * Get options for a trace field
     */
    fetchOptions(request: operations.GetTraceFieldOptionsV1ObservabilityTracesFieldsFieldNameOptionsGetRequest, options?: RequestOptions): Promise<components.GetTraceFieldOptions>;
    /**
     * Get span by id
     */
    getSpanById(request: operations.GetSpanByIdV1ObservabilityTracesTraceIdSpansSpanIdGetRequest, options?: RequestOptions): Promise<components.GetSpan>;
}
//# sourceMappingURL=traces.d.ts.map