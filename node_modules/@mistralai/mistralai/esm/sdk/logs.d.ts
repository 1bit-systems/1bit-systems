import { ClientSDK, RequestOptions } from "../lib/sdks.js";
import * as components from "../models/components/index.js";
import * as operations from "../models/operations/index.js";
export declare class Logs extends ClientSDK {
    /**
     * Search logs
     */
    search(request: operations.SearchLogsV1ObservabilityLogsSearchPostRequest, options?: RequestOptions): Promise<components.GetLogs>;
    /**
     * Get log field definitions
     */
    list(options?: RequestOptions): Promise<components.GetLogFields>;
    /**
     * Get options for a log field
     */
    fetchOptions(request: operations.GetLogFieldOptionsV1ObservabilityLogsFieldsFieldNameOptionsGetRequest, options?: RequestOptions): Promise<components.GetLogFieldOptions>;
}
//# sourceMappingURL=logs.d.ts.map