import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
/**
 * Token usage details for the completion.
 */
export type CompletionTokensDetails = {
    reasoningTokens: number;
};
/** @internal */
export declare const CompletionTokensDetails$inboundSchema: z.ZodType<CompletionTokensDetails, unknown>;
export declare function completionTokensDetailsFromJSON(jsonString: string): SafeParseResult<CompletionTokensDetails, SDKValidationError>;
//# sourceMappingURL=completiontokensdetails.d.ts.map