import { MistralCore } from "../core.js";
import { RequestOptions } from "../lib/sdks.js";
import * as components from "../models/components/index.js";
import { ConnectionError, InvalidRequestError, RequestAbortedError, RequestTimeoutError, UnexpectedClientError } from "../models/errors/httpclienterrors.js";
import * as errors from "../models/errors/index.js";
import { MistralError } from "../models/errors/mistralerror.js";
import { ResponseValidationError } from "../models/errors/responsevalidationerror.js";
import { SDKValidationError } from "../models/errors/sdkvalidationerror.js";
import * as operations from "../models/operations/index.js";
import { APIPromise } from "../types/async.js";
import { Result } from "../types/fp.js";
/**
 * Get Workflow Execution Logs
 *
 * @remarks
 * Retrieve logs for a workflow execution from Dora.
 *
 * First page sets the window via `after`/`before` (default: execution start through now, both
 * widened by a margin so the bounds still prune partitions); later pages pass `cursor`, which
 * carries both the window and the sort order (so `after`/`before`/`order` are then ignored —
 * the order is fixed at the first page so a client can't flip direction mid-pagination).
 */
export declare function workflowsExecutionsGetWorkflowExecutionLogs(client: MistralCore, request: operations.GetWorkflowExecutionLogsRequest, options?: RequestOptions): APIPromise<Result<components.ExecutionLogSearchResponse, errors.HTTPValidationError | MistralError | ResponseValidationError | ConnectionError | RequestAbortedError | RequestTimeoutError | InvalidRequestError | UnexpectedClientError | SDKValidationError>>;
//# sourceMappingURL=workflowsExecutionsGetWorkflowExecutionLogs.d.ts.map