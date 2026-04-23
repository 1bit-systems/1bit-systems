//! MCP protocol glue. `handle()` is pure-ish — it constructs the
//! `PatreonClient` lazily inside `dispatch()` so `tools/list` and
//! `initialize` work with no token configured.
//!
//! We stick to the 2025-06-18 MCP revision the rest of the tree targets.

use serde_json::{Value, json};

use crate::client::{PatreonClient, PatreonError};
use crate::patreon::PostDraft;

/// Static tool catalogue. Kept in one place so `tools/list` and the
/// dispatcher agree without runtime registration.
pub fn tools_json() -> Value {
    json!([
        {
            "name": "patreon_campaigns",
            "description": "List the authenticated creator's campaigns (with tiers + goals inlined).",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "patreon_members",
            "description": "List members (patrons + declined + former) of a campaign. Returns one page; pass 'cursor' from the previous response to continue.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "campaign_id": { "type": "string" },
                    "cursor":      { "type": "string" }
                },
                "required": ["campaign_id"]
            }
        },
        {
            "name": "patreon_post_create",
            "description": "Create a campaign post. Requires a Creator-scope OAuth token with w:campaigns.posts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "campaign_id": { "type": "string" },
                    "title":       { "type": "string" },
                    "content":     { "type": "string" },
                    "post_type":   { "type": "string", "enum": ["public", "public_patrons", "patrons_only"] }
                },
                "required": ["campaign_id", "title", "content"]
            }
        }
    ])
}

/// JSON-RPC dispatch. Returns a single response `Value`. Network work
/// happens only under `tools/call`.
pub async fn handle(req: &Value) -> Value {
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    match method {
        "initialize" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "1bit-mcp-patreon", "version": env!("CARGO_PKG_VERSION") }
            }
        }),
        "tools/list" => json!({ "jsonrpc": "2.0", "id": id, "result": { "tools": tools_json() } }),
        "tools/call" => {
            let name = req
                .pointer("/params/name")
                .and_then(Value::as_str)
                .unwrap_or("");
            let args = req
                .pointer("/params/arguments")
                .cloned()
                .unwrap_or(json!({}));
            let result = dispatch(name, args).await;
            json!({ "jsonrpc": "2.0", "id": id, "result": result })
        }
        _ => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32601, "message": format!("method not found: {method}") }
        }),
    }
}

async fn dispatch(name: &str, args: Value) -> Value {
    match name {
        "patreon_campaigns" => {
            run_with_client(|c| async move {
                c.campaigns()
                    .await
                    .map(|v| serde_json::to_string(&v).unwrap_or_default())
            })
            .await
        }
        "patreon_members" => {
            let Some(cid) = args.get("campaign_id").and_then(Value::as_str) else {
                return error_text("missing required 'campaign_id'");
            };
            let cursor = args
                .get("cursor")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let cid = cid.to_owned();
            run_with_client(move |c| async move {
                c.members(&cid, cursor.as_deref())
                    .await
                    .map(|v| serde_json::to_string(&v).unwrap_or_default())
            })
            .await
        }
        "patreon_post_create" => {
            let Some(cid) = args
                .get("campaign_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
            else {
                return error_text("missing required 'campaign_id'");
            };
            let Some(title) = args.get("title").and_then(Value::as_str).map(str::to_owned) else {
                return error_text("missing required 'title'");
            };
            let Some(content) = args
                .get("content")
                .and_then(Value::as_str)
                .map(str::to_owned)
            else {
                return error_text("missing required 'content'");
            };
            let post_type = args
                .get("post_type")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let draft = PostDraft {
                title,
                content,
                post_type,
            };
            run_with_client(move |c| async move {
                c.create_post(&cid, &draft)
                    .await
                    .map(|v| serde_json::to_string(&v).unwrap_or_default())
            })
            .await
        }
        _ => error_text(&format!("unknown tool: {name}")),
    }
}

async fn run_with_client<F, Fut>(f: F) -> Value
where
    F: FnOnce(PatreonClient) -> Fut,
    Fut: std::future::Future<Output = Result<String, PatreonError>>,
{
    match PatreonClient::from_env() {
        Ok(c) => match f(c).await {
            Ok(s) => text_result(&s),
            Err(e) => error_text(&e.to_string()),
        },
        Err(e) => error_text(&e.to_string()),
    }
}

fn text_result(s: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": s }], "isError": false })
}

fn error_text(s: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": s }], "isError": true })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn initialize_returns_protocol_version() {
        let req = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize" });
        let resp = handle(&req).await;
        assert_eq!(resp["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(resp["result"]["serverInfo"]["name"], "1bit-mcp-patreon");
    }

    #[tokio::test]
    async fn tools_list_advertises_three_tools() {
        let req = json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" });
        let resp = handle(&req).await;
        let arr = resp["result"]["tools"].as_array().expect("array");
        let names: Vec<&str> = arr
            .iter()
            .map(|v| v["name"].as_str().unwrap_or(""))
            .collect();
        assert!(names.contains(&"patreon_campaigns"));
        assert!(names.contains(&"patreon_members"));
        assert!(names.contains(&"patreon_post_create"));
    }

    #[tokio::test]
    async fn unknown_method_returns_minus_32601() {
        let req = json!({ "jsonrpc": "2.0", "id": 3, "method": "eval/exec" });
        let resp = handle(&req).await;
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn unknown_tool_is_error_block() {
        let req = json!({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": { "name": "patreon_nonexistent", "arguments": {} }
        });
        let resp = handle(&req).await;
        assert_eq!(resp["result"]["isError"], true);
        assert!(
            resp["result"]["content"][0]["text"]
                .as_str()
                .unwrap_or("")
                .contains("unknown tool")
        );
    }

    #[tokio::test]
    async fn members_without_campaign_id_is_error_block() {
        let req = json!({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": { "name": "patreon_members", "arguments": {} }
        });
        let resp = handle(&req).await;
        assert_eq!(resp["result"]["isError"], true);
        assert!(
            resp["result"]["content"][0]["text"]
                .as_str()
                .unwrap_or("")
                .contains("campaign_id")
        );
    }

    #[tokio::test]
    async fn post_create_missing_args_is_error_block() {
        let req = json!({
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": { "name": "patreon_post_create", "arguments": { "campaign_id": "c1" } }
        });
        let resp = handle(&req).await;
        assert_eq!(resp["result"]["isError"], true);
    }
}
