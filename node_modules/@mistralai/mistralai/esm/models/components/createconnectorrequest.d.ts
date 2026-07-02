import * as z from "zod/v4";
import { AuthData, AuthData$Outbound } from "./authdata.js";
import { ExtendedOAuthServerMetadata, ExtendedOAuthServerMetadata$Outbound } from "./extendedoauthservermetadata.js";
import { ResourceVisibility } from "./resourcevisibility.js";
export type CreateConnectorRequest = {
    protocol?: "mcp" | undefined;
    /**
     * The name of the connector. Should be 64 char length maximum, alphanumeric, only underscores/dashes.
     */
    name: string;
    /**
     * Optional human-readable title for the connector.
     */
    title?: string | null | undefined;
    /**
     * The description of the connector.
     */
    description: string;
    /**
     * The optional url of the icon you want to associate to the connector.
     */
    iconUrl?: string | null | undefined;
    visibility?: ResourceVisibility | undefined;
    /**
     * The url of the MCP server.
     */
    server: string;
    /**
     * Optional organization-level headers to be sent with the request to the mcp server.
     */
    headers?: {
        [k: string]: any;
    } | null | undefined;
    /**
     * Optional additional authentication data for the connector.
     */
    authData?: AuthData | null | undefined;
    /**
     * Optional OAuth2 authorization server metadata (authorization_endpoint, token_endpoint, etc.). When provided, skips .well-known discovery and uses these endpoints directly.
     */
    oauth2ServerMetadata?: ExtendedOAuthServerMetadata | null | undefined;
    /**
     * Optional URL to fetch OAuth2 authorization server metadata from (RFC 8414). When provided, the metadata is fetched from this URL and used instead of .well-known discovery. Mutually exclusive with oauth2_server_metadata.
     */
    oauth2ServerMetadataUrl?: string | null | undefined;
    /**
     * Optional system prompt for the connector.
     */
    systemPrompt?: string | null | undefined;
};
/** @internal */
export type CreateConnectorRequest$Outbound = {
    protocol: "mcp";
    name: string;
    title?: string | null | undefined;
    description: string;
    icon_url?: string | null | undefined;
    visibility?: string | undefined;
    server: string;
    headers?: {
        [k: string]: any;
    } | null | undefined;
    auth_data?: AuthData$Outbound | null | undefined;
    oauth2_server_metadata?: ExtendedOAuthServerMetadata$Outbound | null | undefined;
    oauth2_server_metadata_url?: string | null | undefined;
    system_prompt?: string | null | undefined;
};
/** @internal */
export declare const CreateConnectorRequest$outboundSchema: z.ZodType<CreateConnectorRequest$Outbound, CreateConnectorRequest>;
export declare function createConnectorRequestToJSON(createConnectorRequest: CreateConnectorRequest): string;
//# sourceMappingURL=createconnectorrequest.d.ts.map