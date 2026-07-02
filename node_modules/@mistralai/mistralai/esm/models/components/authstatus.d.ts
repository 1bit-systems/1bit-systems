import * as z from "zod/v4";
import { OpenEnum } from "../../types/enums.js";
export declare const AuthStatus: {
    readonly Valid: "valid";
    readonly Invalid: "invalid";
    readonly Error: "error";
};
export type AuthStatus = OpenEnum<typeof AuthStatus>;
/** @internal */
export declare const AuthStatus$inboundSchema: z.ZodType<AuthStatus, unknown>;
//# sourceMappingURL=authstatus.d.ts.map