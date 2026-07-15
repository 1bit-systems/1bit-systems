import * as z from "zod/v4";
import * as components from "../components/index.js";
export type ConnectorGetAuthUrlV1Request = {
    appReturnUrl?: string | null | undefined;
    /**
     * Auth method type to use for the authorization URL. Required when the connector supports multiple interactive auth methods; otherwise the sole method is selected automatically. Use this to pick a specific method (e.g. 'oauth2' vs 'github_app').
     */
    methodType?: components.OutboundAuthenticationType | undefined;
    credentialsName?: string | null | undefined;
    /**
     * Only valid with method_type=oauth2. When true, returns a GitHub App installation URL (https://github.com/apps/<slug>/installations/new) if the connector has the proper configuration The Github application needs to have 'Request user authorization (OAuth) during installation' enabled to perform the proper auth loop.
     */
    githubInstallationLink?: boolean | undefined;
    connectorIdOrName: string;
};
/** @internal */
export type ConnectorGetAuthUrlV1Request$Outbound = {
    app_return_url?: string | null | undefined;
    method_type?: string | undefined;
    credentials_name?: string | null | undefined;
    github_installation_link: boolean;
    connector_id_or_name: string;
};
/** @internal */
export declare const ConnectorGetAuthUrlV1Request$outboundSchema: z.ZodType<ConnectorGetAuthUrlV1Request$Outbound, ConnectorGetAuthUrlV1Request>;
export declare function connectorGetAuthUrlV1RequestToJSON(connectorGetAuthUrlV1Request: ConnectorGetAuthUrlV1Request): string;
//# sourceMappingURL=connectorgetauthurlv1.d.ts.map