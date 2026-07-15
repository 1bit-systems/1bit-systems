import { ClientSDK } from "../lib/sdks.js";
import { Campaigns } from "./campaigns.js";
import { ChatCompletionEvents } from "./chatcompletionevents.js";
import { Datasets } from "./datasets.js";
import { Judges } from "./judges.js";
import { Logs } from "./logs.js";
import { Spans } from "./spans.js";
import { Traces } from "./traces.js";
export declare class Observability extends ClientSDK {
    private _chatCompletionEvents?;
    get chatCompletionEvents(): ChatCompletionEvents;
    private _judges?;
    get judges(): Judges;
    private _campaigns?;
    get campaigns(): Campaigns;
    private _datasets?;
    get datasets(): Datasets;
    private _logs?;
    get logs(): Logs;
    private _traces?;
    get traces(): Traces;
    private _spans?;
    get spans(): Spans;
}
//# sourceMappingURL=observability.d.ts.map