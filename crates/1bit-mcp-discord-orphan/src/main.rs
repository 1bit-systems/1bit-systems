//! 1bit-mcp-discord — tiny MCP stdio server exposing Discord classify +
//! mention helpers as tools.
//!
//! Runs as a child process under `1bit-mcp-clients::StdioClient`. Keeps
//! the MCP surface narrow: three tools, pure functions reused from
//! `onebit_agents::watch::discord`. No Discord gateway connection here
//! — posting is still handled by the `1bit-watch-discord` daemon.

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};

use onebit_agents::watch::discord::{
    Classification, classify, is_direct_mention, parse_channel_whitelist, strip_mention,
};

fn tools() -> Value {
    json!([
        {
            "name": "discord_classify",
            "description": "Classify a Discord message body as bug_report / feature_request / question / chat.",
            "inputSchema": {
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"]
            }
        },
        {
            "name": "discord_is_direct_mention",
            "description": "Return whether a message mentions the given bot_id via <@id> or <@!id>.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string" },
                    "bot_id": { "type": "integer", "minimum": 1 }
                },
                "required": ["content", "bot_id"]
            }
        },
        {
            "name": "discord_parse_channel_whitelist",
            "description": "Parse a comma-separated list of Discord channel IDs; returns sanitized u64[].",
            "inputSchema": {
                "type": "object",
                "properties": { "raw": { "type": "string" } },
                "required": ["raw"]
            }
        }
    ])
}

fn handle(req: &Value) -> Value {
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    match method {
        "initialize" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "1bit-mcp-discord", "version": env!("CARGO_PKG_VERSION") }
            }
        }),
        "tools/list" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": { "tools": tools() }
        }),
        "tools/call" => {
            let name = req
                .pointer("/params/name")
                .and_then(Value::as_str)
                .unwrap_or("");
            let args = req
                .pointer("/params/arguments")
                .cloned()
                .unwrap_or(json!({}));
            let result = dispatch(name, &args);
            json!({ "jsonrpc": "2.0", "id": id, "result": result })
        }
        _ => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32601, "message": format!("method not found: {method}") }
        }),
    }
}

fn dispatch(name: &str, args: &Value) -> Value {
    match name {
        "discord_classify" => {
            let text = args.get("text").and_then(Value::as_str).unwrap_or("");
            let c: Classification = classify(text);
            text_result(&format!("{}:{}", c.as_str(), c.specialist().as_str()))
        }
        "discord_is_direct_mention" => {
            let content = args.get("content").and_then(Value::as_str).unwrap_or("");
            let bot_id = args.get("bot_id").and_then(Value::as_u64).unwrap_or(0);
            text_result(&is_direct_mention(content, bot_id).to_string())
        }
        "discord_parse_channel_whitelist" => {
            let raw = args.get("raw").and_then(Value::as_str).unwrap_or("");
            let ids = parse_channel_whitelist(raw);
            text_result(&ids.iter().map(u64::to_string).collect::<Vec<_>>().join(","))
        }
        "discord_strip_mention" => {
            let content = args.get("content").and_then(Value::as_str).unwrap_or("");
            let bot_id = args.get("bot_id").and_then(Value::as_u64).unwrap_or(0);
            text_result(&strip_mention(content, bot_id))
        }
        _ => json!({
            "content": [{ "type": "text", "text": format!("unknown tool: {name}") }],
            "isError": true
        }),
    }
}

fn text_result(s: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": s }], "isError": false })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let stdin = tokio::io::stdin();
    let mut stdin = BufReader::new(stdin);
    let mut stdout = tokio::io::stdout();
    let mut line = String::new();
    loop {
        line.clear();
        let n = stdin.read_line(&mut line).await?;
        if n == 0 {
            break;
        }
        let req: Value = match serde_json::from_str(line.trim_end()) {
            Ok(v) => v,
            Err(e) => {
                let err = json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": { "code": -32700, "message": format!("parse error: {e}") }
                });
                writeln_frame(&mut stdout, &err).await?;
                continue;
            }
        };
        let resp = handle(&req);
        writeln_frame(&mut stdout, &resp).await?;
    }
    Ok(())
}

async fn writeln_frame<W: tokio::io::AsyncWriteExt + Unpin>(
    w: &mut W,
    v: &Value,
) -> std::io::Result<()> {
    let mut s = serde_json::to_string(v).map_err(std::io::Error::other)?;
    s.push('\n');
    w.write_all(s.as_bytes()).await?;
    w.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_list_advertises_three_tools() {
        let t = tools();
        let arr = t.as_array().expect("array");
        assert!(arr.len() >= 3);
        let names: Vec<&str> = arr
            .iter()
            .map(|v| v["name"].as_str().unwrap_or(""))
            .collect();
        assert!(names.contains(&"discord_classify"));
        assert!(names.contains(&"discord_is_direct_mention"));
        assert!(names.contains(&"discord_parse_channel_whitelist"));
    }

    #[test]
    fn initialize_returns_protocol_version() {
        let req = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize" });
        let resp = handle(&req);
        assert_eq!(resp["result"]["protocolVersion"], "2025-06-18");
    }

    #[test]
    fn tools_call_classify_routes_bug_report() {
        let req = json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": { "name": "discord_classify", "arguments": { "text": "got a panic in decode" } }
        });
        let resp = handle(&req);
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("bug_report:"));
    }

    #[test]
    fn tools_call_unknown_method_returns_minus_32601() {
        let req = json!({ "jsonrpc": "2.0", "id": 3, "method": "eval/exec" });
        let resp = handle(&req);
        assert_eq!(resp["error"]["code"], -32601);
    }
}
