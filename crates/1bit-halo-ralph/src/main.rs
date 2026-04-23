// 1bit-halo-ralph — minimal ralph-loop agent for 1bit-halo-server (or any
// OpenAI-compatible chat endpoint).
//
// One task, N iterations, optional test-command feedback. Prints model output
// to stdout; test-command output is appended as the next user turn so the
// model sees its own failures.
//
// Rule A: Rust, no Python at runtime. Caller-side tool.

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

const DEFAULT_SYSTEM: &str = "You are a precise, terse coding assistant. \
Every turn, emit exactly one concrete action: a diff, a shell command, or a \
specific file change. No preamble, no apology, no commentary. If the previous \
turn included a failing test output, address it directly.";

#[derive(Parser, Debug)]
#[command(
    name = "1bit-halo-ralph",
    about = "Minimal ralph-loop agent pointed at an OpenAI-compatible endpoint",
    version
)]
struct Args {
    /// Task prompt for the agent (wrap in quotes).
    #[arg(long)]
    task: String,

    /// OpenAI-compatible base URL.
    #[arg(
        long,
        default_value = "http://localhost:8180/v1",
        env = "RALPH_BASE_URL"
    )]
    base_url: String,

    /// Model id.
    #[arg(long, default_value = "1bit-halo-v2", env = "RALPH_MODEL")]
    model: String,

    /// Bearer token, if the endpoint requires one.
    #[arg(long, env = "RALPH_API_KEY")]
    api_key: Option<String>,

    /// Maximum loop iterations.
    #[arg(long, default_value_t = 5)]
    max_iter: u32,

    /// Optional shell command to run each iteration. Zero exit = success (loop
    /// breaks). Nonzero exit = failure (stdout+stderr appended as next user
    /// turn and loop continues).
    #[arg(long)]
    test_cmd: Option<String>,

    /// Override the system prompt.
    #[arg(long)]
    system: Option<String>,

    /// Sampling temperature.
    #[arg(long, default_value_t = 0.3)]
    temperature: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    temperature: f32,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()?;
    let endpoint = format!("{}/chat/completions", args.base_url.trim_end_matches('/'));

    let system = args.system.as_deref().unwrap_or(DEFAULT_SYSTEM);
    let mut history: Vec<Message> = vec![
        Message {
            role: "system".into(),
            content: system.into(),
        },
        Message {
            role: "user".into(),
            content: args.task.clone(),
        },
    ];

    for iter in 1..=args.max_iter {
        println!(
            "── ralph iter {}/{} ────────────────────",
            iter, args.max_iter
        );

        let req = ChatRequest {
            model: &args.model,
            messages: &history,
            temperature: args.temperature,
            stream: false,
        };
        let mut builder = client.post(&endpoint).json(&req);
        if let Some(key) = &args.api_key {
            builder = builder.bearer_auth(key);
        }
        let resp = builder
            .send()
            .await
            .with_context(|| format!("POST {endpoint} failed"))?
            .error_for_status()?
            .json::<ChatResponse>()
            .await
            .context("malformed chat response")?;

        let content = resp
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        println!("{content}\n");
        history.push(Message {
            role: "assistant".into(),
            content,
        });

        let Some(cmd) = args.test_cmd.as_deref() else {
            break;
        };

        println!("── test · {cmd}");
        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()
            .await
            .with_context(|| format!("failed to spawn test-cmd: {cmd}"))?;

        if output.status.success() {
            println!("✓ tests passed at iter {iter}");
            return Ok(());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let feedback = format!(
            "Previous action failed. Test command `{cmd}` exited with code {code}.\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}",
            code = output.status.code().unwrap_or(-1),
        );
        eprintln!("✗ tests failed at iter {iter}");
        history.push(Message {
            role: "user".into(),
            content: feedback,
        });
    }

    eprintln!("── ralph gave up after {} iterations", args.max_iter);
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_roundtrips_through_serde() {
        let m = Message {
            role: "user".into(),
            content: "hello".into(),
        };
        let wire = serde_json::to_string(&m).unwrap();
        let back: Message = serde_json::from_str(&wire).unwrap();
        assert_eq!(back.role, "user");
        assert_eq!(back.content, "hello");
    }

    #[test]
    fn chat_request_shape_matches_openai() {
        let msgs = vec![Message {
            role: "user".into(),
            content: "ping".into(),
        }];
        let req = ChatRequest {
            model: "1bit-halo-v2",
            messages: &msgs,
            temperature: 0.3,
            stream: false,
        };
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&req).unwrap()).unwrap();
        assert_eq!(v["model"], "1bit-halo-v2");
        assert_eq!(v["stream"], false);
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "ping");
        assert!((v["temperature"].as_f64().unwrap() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn cli_parses_minimal_task() {
        use clap::Parser;
        let args = Args::parse_from(["1bit-halo-ralph", "--task", "fix the tests"]);
        assert_eq!(args.task, "fix the tests");
        assert_eq!(args.model, "1bit-halo-v2");
        assert_eq!(args.base_url, "http://localhost:8180/v1");
        assert_eq!(args.max_iter, 5);
        assert!(args.test_cmd.is_none());
    }

    #[test]
    fn cli_respects_overrides() {
        use clap::Parser;
        let args = Args::parse_from([
            "1bit-halo-ralph",
            "--task",
            "optimize",
            "--model",
            "onebit-ft",
            "--max-iter",
            "10",
            "--test-cmd",
            "cargo test",
        ]);
        assert_eq!(args.model, "onebit-ft");
        assert_eq!(args.max_iter, 10);
        assert_eq!(args.test_cmd.as_deref(), Some("cargo test"));
    }
}
