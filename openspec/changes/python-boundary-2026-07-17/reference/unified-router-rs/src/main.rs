// Unified NPU+GPU Router — std-only Rust port of unified-router.py
//
// Routes requests to NPU (qwen3-0.6b-FLM) or GPU (MLX/Qwen3-8B) based on policy.
// Sits in front of lemond so apps see one endpoint.
//
// Zero external crates: only the Rust standard library. Single static binary.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

// ── Routing policy ──────────────────────────────────────────────────
const SMALL_MODEL: &str = "qwen3-0.6b-FLM"; // NPU (210ms/tok, efficient)
const BIG_MODEL: &str = "mlx-engine.mlx-community/Qwen3-8B-4bit"; // GPU (fast compute)
const ROUTER_NAME: &str = "user.Unified";

const GPU_KEYWORDS: &[&str] = &[
    "code", "explain", "analyze", "write", "implement", "debug", "refactor", "function",
    "algorithm", "bug", "error", "review", "optimize", "design", "architecture", "test",
];

// ── Minimal JSON value + parser + serializer (std-only) ─────────────
#[derive(Clone, Debug)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    fn as_array(&self) -> Option<&Vec<Json>> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    fn new(b: &'a [u8]) -> Self {
        P { b, i: 0 }
    }
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn parse(&mut self) -> Option<Json> {
        self.ws();
        if self.i >= self.b.len() {
            return None;
        }
        match self.b[self.i] {
            b'{' => self.obj(),
            b'[' => self.arr(),
            b'"' => self.string().map(Json::Str),
            b't' | b'f' => self.boolean(),
            b'n' => self.null(),
            _ => self.number(),
        }
    }
    fn obj(&mut self) -> Option<Json> {
        self.i += 1; // {
        let mut m = BTreeMap::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b'}' {
            self.i += 1;
            return Some(Json::Obj(m));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            if self.i >= self.b.len() || self.b[self.i] != b':' {
                return None;
            }
            self.i += 1;
            let v = self.parse()?;
            m.insert(k, v);
            self.ws();
            if self.i >= self.b.len() {
                return None;
            }
            match self.b[self.i] {
                b',' => {
                    self.i += 1;
                }
                b'}' => {
                    self.i += 1;
                    return Some(Json::Obj(m));
                }
                _ => return None,
            }
        }
    }
    fn arr(&mut self) -> Option<Json> {
        self.i += 1; // [
        let mut a = Vec::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b']' {
            self.i += 1;
            return Some(Json::Arr(a));
        }
        loop {
            let v = self.parse()?;
            a.push(v);
            self.ws();
            if self.i >= self.b.len() {
                return None;
            }
            match self.b[self.i] {
                b',' => {
                    self.i += 1;
                }
                b']' => {
                    self.i += 1;
                    return Some(Json::Arr(a));
                }
                _ => return None,
            }
        }
    }
    fn string(&mut self) -> Option<String> {
        if self.i >= self.b.len() || self.b[self.i] != b'"' {
            return None;
        }
        self.i += 1;
        let mut s = String::new();
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'"' => return Some(s),
                b'\\' => {
                    if self.i >= self.b.len() {
                        return None;
                    }
                    let e = self.b[self.i];
                    self.i += 1;
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b't' => s.push('\t'),
                        b'r' => s.push('\r'),
                        b'b' => s.push('\u{0008}'),
                        b'f' => s.push('\u{000C}'),
                        b'u' => {
                            if self.i + 4 > self.b.len() {
                                return None;
                            }
                            let hex = std::str::from_utf8(&self.b[self.i..self.i + 4]).ok()?;
                            let cp = u32::from_str_radix(hex, 16).ok()?;
                            self.i += 4;
                            if let Some(ch) = char::from_u32(cp) {
                                s.push(ch);
                            }
                        }
                        _ => return None,
                    }
                }
                _ => s.push(c as char),
            }
        }
        None
    }
    fn boolean(&mut self) -> Option<Json> {
        if self.b[self.i..].starts_with(b"true") {
            self.i += 4;
            Some(Json::Bool(true))
        } else if self.b[self.i..].starts_with(b"false") {
            self.i += 5;
            Some(Json::Bool(false))
        } else {
            None
        }
    }
    fn null(&mut self) -> Option<Json> {
        if self.b[self.i..].starts_with(b"null") {
            self.i += 4;
            Some(Json::Null)
        } else {
            None
        }
    }
    fn number(&mut self) -> Option<Json> {
        let start = self.i;
        while self.i < self.b.len()
            && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        {
            self.i += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).ok()?;
        s.parse::<f64>().ok().map(Json::Num)
    }
}

fn parse_json(bytes: &[u8]) -> Option<Json> {
    P::new(bytes).parse()
}

fn ser(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                out.push_str(&format!("{}", *n as i64));
            } else {
                out.push_str(&format!("{}", n));
            }
        }
        Json::Str(s) => ser_str(s, out),
        Json::Arr(a) => {
            out.push('[');
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                ser(e, out);
            }
            out.push(']');
        }
        Json::Obj(m) => {
            out.push('{');
            for (i, (k, val)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                ser_str(k, out);
                out.push(':');
                ser(val, out);
            }
            out.push('}');
        }
    }
}

fn ser_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

// ── Routing decision (faithful to should_route_to_gpu) ──────────────
fn text_triggers_gpu(text: &str) -> bool {
    if text.len() > 800 {
        return true;
    }
    let lower = text.to_lowercase();
    GPU_KEYWORDS.iter().any(|kw| lower.contains(kw))
}

fn should_route_to_gpu(body: &Json) -> bool {
    if let Some(Json::Arr(messages)) = body.get("messages") {
        for msg in messages {
            match msg.get("content") {
                Some(Json::Str(s)) => {
                    if text_triggers_gpu(s) {
                        return true;
                    }
                }
                Some(Json::Arr(parts)) => {
                    for part in parts {
                        if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                                if text_triggers_gpu(t) {
                                    return true;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    // Tool calls → GPU (needs more capability)
    if let Some(tools) = body.get("tools") {
        if let Some(a) = tools.as_array() {
            if !a.is_empty() {
                return true;
            }
        } else if !matches!(tools, Json::Null) {
            return true;
        }
    }
    false
}

fn obj_set(v: &mut Json, key: &str, val: Json) {
    if let Json::Obj(m) = v {
        m.insert(key.to_string(), val);
    }
}

// ── Backend URL parsing ─────────────────────────────────────────────
#[derive(Clone)]
struct Backend {
    host: String,
    port: u16,
}

fn parse_backend(url: &str) -> Backend {
    let u = url.trim_end_matches('/');
    let u = u.strip_prefix("http://").unwrap_or(u);
    let u = u.strip_prefix("https://").unwrap_or(u);
    let (host, port) = match u.split_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(13305)),
        None => (u.to_string(), 80),
    };
    Backend { host, port }
}

// ── Minimal HTTP forward to backend ─────────────────────────────────
struct HttpResp {
    status: u16,
    content_type: String,
    body: Vec<u8>,
}

fn forward(be: &Backend, method: &str, path: &str, body: &[u8]) -> HttpResp {
    let addr = format!("{}:{}", be.host, be.port);
    let mut stream = match TcpStream::connect(&addr) {
        Ok(s) => s,
        Err(e) => return err_resp(502, &format!("connect {}: {}", addr, e)),
    };
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(600)));

    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
        be.host,
        body.len()
    );
    let mut raw = req.into_bytes();
    raw.extend_from_slice(body);
    if let Err(e) = stream.write_all(&raw) {
        return err_resp(502, &format!("write: {}", e));
    }
    req = String::new();
    let _ = req;

    let mut resp = Vec::new();
    if let Err(e) = stream.read_to_end(&mut resp) {
        return err_resp(502, &format!("read: {}", e));
    }
    parse_http_response(&resp)
}

fn parse_http_response(resp: &[u8]) -> HttpResp {
    // Split headers / body on first \r\n\r\n
    let sep = resp
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or(resp.len());
    let head = String::from_utf8_lossy(&resp[..sep]);
    let mut lines = head.lines();
    let status_line = lines.next().unwrap_or("HTTP/1.1 502");
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(502);
    let mut content_type = "application/json".to_string();
    let mut chunked = false;
    for line in lines {
        let lower = line.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-type:") {
            content_type = v.trim().to_string();
        }
        if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
            chunked = true;
        }
    }
    let body_start = (sep + 4).min(resp.len());
    let body = if chunked {
        dechunk(&resp[body_start..])
    } else {
        resp[body_start..].to_vec()
    };
    HttpResp {
        status,
        content_type,
        body,
    }
}

fn dechunk(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let mut j = i;
        while j < data.len() && data[j] != b'\r' {
            j += 1;
        }
        let size_str = String::from_utf8_lossy(&data[i..j]);
        let size = usize::from_str_radix(size_str.trim(), 16).unwrap_or(0);
        if size == 0 {
            break;
        }
        i = j + 2; // skip \r\n
        if i + size > data.len() {
            out.extend_from_slice(&data[i..]);
            break;
        }
        out.extend_from_slice(&data[i..i + size]);
        i += size + 2; // data + \r\n
    }
    out
}

fn err_resp(status: u16, msg: &str) -> HttpResp {
    let mut s = String::new();
    ser(
        &Json::Obj({
            let mut m = BTreeMap::new();
            m.insert("error".to_string(), Json::Str(msg.to_string()));
            m
        }),
        &mut s,
    );
    HttpResp {
        status,
        content_type: "application/json".to_string(),
        body: s.into_bytes(),
    }
}

// ── Client-facing HTTP write ────────────────────────────────────────
fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    extra: &[(&str, &str)],
    body: &[u8],
) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        _ => "OK",
    };
    let mut head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

// ── Request handling ────────────────────────────────────────────────
struct Req {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Option<Req> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).ok()? == 0 {
        return None;
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            break;
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).ok()?;
    }
    Some(Req { method, path, body })
}

fn handle(stream: &mut TcpStream, be: &Backend) {
    let req = match read_request(stream) {
        Some(r) => r,
        None => return,
    };
    let now = {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let (h, m, s) = ((t / 3600) % 24, (t / 60) % 60, t % 60);
        format!("{:02}:{:02}:{:02}", h, m, s)
    };

    if req.method == "POST" && req.path.contains("/chat/completions") {
        let body_bytes = if req.body.is_empty() { b"{}".to_vec() } else { req.body.clone() };
        let mut body = parse_json(&body_bytes).unwrap_or(Json::Obj(BTreeMap::new()));
        let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string();

        let target = if model == ROUTER_NAME || model == "auto" {
            if should_route_to_gpu(&body) { BIG_MODEL } else { SMALL_MODEL }
        } else if model == "npu" {
            SMALL_MODEL
        } else if model == "gpu" {
            BIG_MODEL
        } else {
            "" // pass model through unchanged
        };
        if !target.is_empty() {
            obj_set(&mut body, "model", Json::Str(target.to_string()));
        }
        let route = body.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string();

        let mut out = String::new();
        ser(&body, &mut out);
        let resp = forward(be, "POST", "/v1/chat/completions", out.as_bytes());
        println!("[{now}] POST {} -> {} [{}]", req.path, resp.status, route);
        write_response(stream, resp.status, &resp.content_type, &[("X-Route", &route)], &resp.body);
        return;
    }

    if req.method == "POST" && req.path.contains("/completions") {
        let body_bytes = if req.body.is_empty() { b"{}".to_vec() } else { req.body.clone() };
        let mut body = parse_json(&body_bytes).unwrap_or(Json::Obj(BTreeMap::new()));
        let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string();
        let mut send = req.body.clone();
        if model == ROUTER_NAME {
            let target = if should_route_to_gpu(&body) { BIG_MODEL } else { SMALL_MODEL };
            obj_set(&mut body, "model", Json::Str(target.to_string()));
            let mut out = String::new();
            ser(&body, &mut out);
            send = out.into_bytes();
        }
        let resp = forward(be, "POST", &req.path, &send);
        println!("[{now}] POST {} -> {}", req.path, resp.status);
        write_response(stream, resp.status, &resp.content_type, &[], &resp.body);
        return;
    }

    // Pass-through (GET or other POST)
    let resp = forward(be, &req.method, &req.path, &req.body);
    println!("[{now}] {} {} -> {}", req.method, req.path, resp.status);
    write_response(stream, resp.status, &resp.content_type, &[], &resp.body);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut port: u16 = 13305;
    let mut backend = "http://127.0.0.1:13305".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                if i < args.len() {
                    port = args[i].parse().unwrap_or(13305);
                }
            }
            "--backend" => {
                i += 1;
                if i < args.len() {
                    backend = args[i].clone();
                }
            }
            "-h" | "--help" => {
                println!("Unified NPU+GPU Router\n  --port <PORT>     listen port (default 13305)\n  --backend <URL>   lemond backend URL (default http://127.0.0.1:13305)");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    let be = Arc::new(parse_backend(&backend));

    println!("{}", "=".repeat(56));
    println!("  Unified NPU+GPU Router (Rust, std-only)");
    println!("  Listen:  http://0.0.0.0:{port}");
    println!("  Backend: http://{}:{}", be.host, be.port);
    println!("{}", "=".repeat(56));
    println!();
    println!("  Model routing:");
    println!("    user.Unified / auto  -> auto-route NPU <-> GPU");
    println!("    npu                  -> {SMALL_MODEL} (NPU)");
    println!("    gpu                  -> {BIG_MODEL} (GPU)");
    println!("    <any other>          -> pass-through to lemond");
    println!();

    let listener = TcpListener::bind(("0.0.0.0", port)).expect("bind failed");
    for stream in listener.incoming() {
        if let Ok(mut s) = stream {
            let be = Arc::clone(&be);
            thread::spawn(move || handle(&mut s, &be));
        }
    }
}
