import * as z from "zod/v4";
import { Result as SafeParseResult } from "../../types/fp.js";
import { SDKValidationError } from "../errors/sdkvalidationerror.js";
import { ConsumerType } from "./consumertype.js";
import { CredentialsStatus } from "./credentialsstatus.js";
import { OutboundAuthenticationType } from "./outboundauthenticationtype.js";
export type AuthenticationConfiguration = {
    name: string;
    authenticationType: OutboundAuthenticationType;
    scope: ConsumerType;
    status?: CredentialsStatus | null | undefined;
    isDefault: boolean;
};
/** @internal */
export declare const AuthenticationConfiguration$inboundSchema: z.ZodType<AuthenticationConfiguration, unknown>;
export declare function authenticationConfigurationFromJSON(jsonString: string): SafeParseResult<AuthenticationConfiguration, SDKValidationError>;
//# sourceMappingURL=authenticationconfiguration.d.ts.map