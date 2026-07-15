import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { AuthStatus } from "./authstatus.js";
import { HTTPStatus } from "./httpstatus.js";
export type CredentialsStatus = {
    statusType: AuthStatus;
    lastCheckedAt?: Date | null | undefined;
    errorHttpCode?: HTTPStatus | null | undefined;
    errorMessage?: string | null | undefined;
};
/** @internal */
export declare const CredentialsStatus$inboundSchema: z.ZodType<CredentialsStatus, unknown>;
export declare function credentialsStatusFromJSON(jsonString: string): SafeParseResult<CredentialsStatus, SDKValidationError>;
//# sourceMappingURL=credentialsstatus.d.ts.map