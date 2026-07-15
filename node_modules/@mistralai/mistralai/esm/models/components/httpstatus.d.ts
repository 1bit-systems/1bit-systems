import * as z from "zod/v4";
import { OpenEnum } from "../../types/enums.js";
/**
 * HTTP status codes and reason phrases
 *
 * @remarks
 *
 * Status codes from the following RFCs are all observed:
 *
 *     * RFC 7231: Hypertext Transfer Protocol (HTTP/1.1), obsoletes 2616
 *     * RFC 6585: Additional HTTP Status Codes
 *     * RFC 3229: Delta encoding in HTTP
 *     * RFC 4918: HTTP Extensions for WebDAV, obsoletes 2518
 *     * RFC 5842: Binding Extensions to WebDAV
 *     * RFC 7238: Permanent Redirect
 *     * RFC 2295: Transparent Content Negotiation in HTTP
 *     * RFC 2774: An HTTP Extension Framework
 *     * RFC 7725: An HTTP Status Code to Report Legal Obstacles
 *     * RFC 7540: Hypertext Transfer Protocol Version 2 (HTTP/2)
 *     * RFC 2324: Hyper Text Coffee Pot Control Protocol (HTCPCP/1.0)
 *     * RFC 8297: An HTTP Status Code for Indicating Hints
 *     * RFC 8470: Using Early Data in HTTP
 */
export declare const HTTPStatus: {
    readonly OneHundred: 100;
    readonly OneHundredAndOne: 101;
    readonly OneHundredAndTwo: 102;
    readonly OneHundredAndThree: 103;
    readonly TwoHundred: 200;
    readonly TwoHundredAndOne: 201;
    readonly TwoHundredAndTwo: 202;
    readonly TwoHundredAndThree: 203;
    readonly TwoHundredAndFour: 204;
    readonly TwoHundredAndFive: 205;
    readonly TwoHundredAndSix: 206;
    readonly TwoHundredAndSeven: 207;
    readonly TwoHundredAndEight: 208;
    readonly TwoHundredAndTwentySix: 226;
    readonly ThreeHundred: 300;
    readonly ThreeHundredAndOne: 301;
    readonly ThreeHundredAndTwo: 302;
    readonly ThreeHundredAndThree: 303;
    readonly ThreeHundredAndFour: 304;
    readonly ThreeHundredAndFive: 305;
    readonly ThreeHundredAndSeven: 307;
    readonly ThreeHundredAndEight: 308;
    readonly FourHundred: 400;
    readonly FourHundredAndOne: 401;
    readonly FourHundredAndTwo: 402;
    readonly FourHundredAndThree: 403;
    readonly FourHundredAndFour: 404;
    readonly FourHundredAndFive: 405;
    readonly FourHundredAndSix: 406;
    readonly FourHundredAndSeven: 407;
    readonly FourHundredAndEight: 408;
    readonly FourHundredAndNine: 409;
    readonly FourHundredAndTen: 410;
    readonly FourHundredAndEleven: 411;
    readonly FourHundredAndTwelve: 412;
    readonly FourHundredAndThirteen: 413;
    readonly FourHundredAndFourteen: 414;
    readonly FourHundredAndFifteen: 415;
    readonly FourHundredAndSixteen: 416;
    readonly FourHundredAndSeventeen: 417;
    readonly FourHundredAndEighteen: 418;
    readonly FourHundredAndTwentyOne: 421;
    readonly FourHundredAndTwentyTwo: 422;
    readonly FourHundredAndTwentyThree: 423;
    readonly FourHundredAndTwentyFour: 424;
    readonly FourHundredAndTwentyFive: 425;
    readonly FourHundredAndTwentySix: 426;
    readonly FourHundredAndTwentyEight: 428;
    readonly FourHundredAndTwentyNine: 429;
    readonly FourHundredAndThirtyOne: 431;
    readonly FourHundredAndFiftyOne: 451;
    readonly FiveHundred: 500;
    readonly FiveHundredAndOne: 501;
    readonly FiveHundredAndTwo: 502;
    readonly FiveHundredAndThree: 503;
    readonly FiveHundredAndFour: 504;
    readonly FiveHundredAndFive: 505;
    readonly FiveHundredAndSix: 506;
    readonly FiveHundredAndSeven: 507;
    readonly FiveHundredAndEight: 508;
    readonly FiveHundredAndTen: 510;
    readonly FiveHundredAndEleven: 511;
};
/**
 * HTTP status codes and reason phrases
 *
 * @remarks
 *
 * Status codes from the following RFCs are all observed:
 *
 *     * RFC 7231: Hypertext Transfer Protocol (HTTP/1.1), obsoletes 2616
 *     * RFC 6585: Additional HTTP Status Codes
 *     * RFC 3229: Delta encoding in HTTP
 *     * RFC 4918: HTTP Extensions for WebDAV, obsoletes 2518
 *     * RFC 5842: Binding Extensions to WebDAV
 *     * RFC 7238: Permanent Redirect
 *     * RFC 2295: Transparent Content Negotiation in HTTP
 *     * RFC 2774: An HTTP Extension Framework
 *     * RFC 7725: An HTTP Status Code to Report Legal Obstacles
 *     * RFC 7540: Hypertext Transfer Protocol Version 2 (HTTP/2)
 *     * RFC 2324: Hyper Text Coffee Pot Control Protocol (HTCPCP/1.0)
 *     * RFC 8297: An HTTP Status Code for Indicating Hints
 *     * RFC 8470: Using Early Data in HTTP
 */
export type HTTPStatus = OpenEnum<typeof HTTPStatus>;
/** @internal */
export declare const HTTPStatus$inboundSchema: z.ZodType<HTTPStatus, unknown>;
//# sourceMappingURL=httpstatus.d.ts.map