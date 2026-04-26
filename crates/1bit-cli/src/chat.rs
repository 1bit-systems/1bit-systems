// `1bit chat` — one-shot REPL against 1bit-server :8180. No session history,
// (deleted: was 1bit-router orphan crate)
// commit de53544). Good enough for demos + smoke tests.

use anyhow::{Context, Result, bail};
use serde_json::json;
use std::io::{BufRead, Write};
use std::time::Duration;

const DEFAULT_URL: &str = "http://127.0.0.1:8180/v1/chat/completions";
const DEFAULT_MODEL: &str = "1bit-monster-2b";

pub async fn run(url: Option<String>, model: Option<String>, max_tokens: u32) -> Result<()> {
    let url = url
        .unwrap_or_else(|| std::env::var("HALO_CHAT_URL").unwrap_or_else(|_| DEFAULT_URL.into()));
    let model = model.unwrap_or_else(|| DEFAULT_MODEL.into());

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    println!("1bit chat · {url} · model={model} · Ctrl-D to exit");
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    loop {
        print!("> ");
        stdout.flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                println!();
                break;
            }
            Ok(_) => {}
            Err(e) => bail!("stdin read: {e}"),
        }
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }

        let body = json!({
            "model": model,
            "messages": [{ "role": "user", "content": prompt }],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": false,
        });
        let res = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        if !res.status().is_success() {
            println!(
                "  ← {} {}",
                res.status(),
                res.text().await.unwrap_or_default()
            );
            continue;
        }
        let v: serde_json::Value = res.json().await?;
        let reply = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("(empty)");
        let usage = &v["usage"];
        println!("{}", reply);
        println!(
            "  (prompt={} completion={} total={})",
            usage["prompt_tokens"].as_u64().unwrap_or(0),
            usage["completion_tokens"].as_u64().unwrap_or(0),
            usage["total_tokens"].as_u64().unwrap_or(0)
        );
    }
    Ok(())
}
