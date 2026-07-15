import * as z from "zod/v4";
/**
 * Custom superset of RFC 8414 OAuth 2.0 Authorization Server Metadata.
 *
 * @remarks
 *
 * Stored at connector creation time (provided for HTTP connectors, discovered via .well-known for MCP).
 * Mirrors the shape of .well-known/oauth-authorization-server responses.
 */
export type ExtendedOAuthServerMetadata = {
    issuer: string;
    authorizationEndpoint: string;
    tokenEndpoint: string;
    registrationEndpoint?: string | null | undefined;
    scopesSupported?: Array<string> | null | undefined;
    responseTypesSupported?: Array<string> | undefined;
    responseModesSupported?: Array<string> | null | undefined;
    grantTypesSupported?: Array<string> | null | undefined;
    tokenEndpointAuthMethodsSupported?: Array<string> | null | undefined;
    tokenEndpointAuthSigningAlgValuesSupported?: Array<string> | null | undefined;
    serviceDocumentation?: string | null | undefined;
    uiLocalesSupported?: Array<string> | null | undefined;
    opPolicyUri?: string | null | undefined;
    opTosUri?: string | null | undefined;
    revocationEndpoint?: string | null | undefined;
    revocationEndpointAuthMethodsSupported?: Array<string> | null | undefined;
    revocationEndpointAuthSigningAlgValuesSupported?: Array<string> | null | undefined;
    introspectionEndpoint?: string | null | undefined;
    introspectionEndpointAuthMethodsSupported?: Array<string> | null | undefined;
    introspectionEndpointAuthSigningAlgValuesSupported?: Array<string> | null | undefined;
    codeChallengeMethodsSupported?: Array<string> | null | undefined;
    clientIdMetadataDocumentSupported?: boolean | null | undefined;
    xResourceUrl?: string | null | undefined;
};
/** @internal */
export type ExtendedOAuthServerMetadata$Outbound = {
    issuer: string;
    authorization_endpoint: string;
    token_endpoint: string;
    registration_endpoint?: string | null | undefined;
    scopes_supported?: Array<string> | null | undefined;
    response_types_supported?: Array<string> | undefined;
    response_modes_supported?: Array<string> | null | undefined;
    grant_types_supported?: Array<string> | null | undefined;
    token_endpoint_auth_methods_supported?: Array<string> | null | undefined;
    token_endpoint_auth_signing_alg_values_supported?: Array<string> | null | undefined;
    service_documentation?: string | null | undefined;
    ui_locales_supported?: Array<string> | null | undefined;
    op_policy_uri?: string | null | undefined;
    op_tos_uri?: string | null | undefined;
    revocation_endpoint?: string | null | undefined;
    revocation_endpoint_auth_methods_supported?: Array<string> | null | undefined;
    revocation_endpoint_auth_signing_alg_values_supported?: Array<string> | null | undefined;
    introspection_endpoint?: string | null | undefined;
    introspection_endpoint_auth_methods_supported?: Array<string> | null | undefined;
    introspection_endpoint_auth_signing_alg_values_supported?: Array<string> | null | undefined;
    code_challenge_methods_supported?: Array<string> | null | undefined;
    client_id_metadata_document_supported?: boolean | null | undefined;
    x_resource_url?: string | null | undefined;
};
/** @internal */
export declare const ExtendedOAuthServerMetadata$outboundSchema: z.ZodType<ExtendedOAuthServerMetadata$Outbound, ExtendedOAuthServerMetadata>;
export declare function extendedOAuthServerMetadataToJSON(extendedOAuthServerMetadata: ExtendedOAuthServerMetadata): string;
//# sourceMappingURL=extendedoauthservermetadata.d.ts.map