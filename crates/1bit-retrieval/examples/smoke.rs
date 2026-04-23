//! Quick smoke-test against the real `docs/wiki` tree. Not a unit test —
//! run with `cargo run -p onebit-retrieval --example smoke -- "your query"`.

use std::env;
use std::path::PathBuf;

fn main() {
    let mut args = env::args().skip(1);
    let query = args
        .next()
        .unwrap_or_else(|| "amdgpu OPTC hang".to_string());
    let k: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    // Walk upward from CARGO_MANIFEST_DIR to find docs/wiki.
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !root.join("docs/wiki").exists() {
        if !root.pop() {
            eprintln!(
                "couldn't find docs/wiki from {:?}",
                env!("CARGO_MANIFEST_DIR")
            );
            std::process::exit(1);
        }
    }
    let wiki = root.join("docs/wiki");

    let idx = onebit_retrieval::WikiIndex::load(&wiki).expect("load");
    eprintln!("loaded {} chunks from {}", idx.len(), wiki.display());

    let hits = idx.top_k(&query, k);
    print!("{}", onebit_retrieval::format_for_system_prompt(&hits));
}
