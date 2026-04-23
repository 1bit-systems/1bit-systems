//! `1bit-ingest` — four-verb curator CLI.
//!
//! ```text
//!   1bit-ingest prepare  <src-dir> --out corpus.tar
//!   1bit-ingest pack     --model trained.gguf --manifest catalog.toml --out kevin.1bl
//!   1bit-ingest validate kevin.1bl
//!   1bit-ingest add-residual --in kevin.1bl \
//!                            --residual kevin.arith --index kevin-index.cbor \
//!                            --out kevin-premium.1bl
//! ```
//!
//! Output messages stay calm-technical. Help text is allowed a touch of
//! curator-flavoured prose — see the `#[command(about = ...)]` strings.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "1bit-ingest",
    version,
    about = "source-side packer for .1bl catalogs",
    long_about = "1bit-ingest is the curator's forge. Melt a FLAC directory \
                  down into a training corpus, then hammer the trained weights \
                  + metadata into a single .1bl catalog ready to ship."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Scan a FLAC directory, extract metadata, tar up a RunPod-ready training corpus.
    Prepare {
        /// Directory to walk. All `*.flac` files below this root are
        /// recorded in the manifest.
        src_dir: PathBuf,
        /// Output `.tar` archive. Holds every FLAC plus a
        /// `manifest.json` sidecar.
        #[arg(long)]
        out: PathBuf,
    },

    /// Assemble a `.1bl` from a trained GGUF, a catalog.toml, and optional sidecars.
    Pack {
        /// Trained ternary-LM `.gguf` produced on the pod.
        #[arg(long)]
        model: PathBuf,
        /// Hand-written `catalog.toml` — title, artist, license, tracks.
        #[arg(long)]
        manifest: PathBuf,
        /// Optional `cover.webp` / `.png` (≤ 512 KB per spec).
        #[arg(long)]
        cover: Option<PathBuf>,
        /// Optional lyrics bundle (UTF-8 text).
        #[arg(long)]
        lyrics: Option<PathBuf>,
        /// Output `.1bl` container path.
        #[arg(long)]
        out: PathBuf,
    },

    /// Verify footer hash, print the manifest, list TLV sections.
    Validate {
        /// Path to a `.1bl` file.
        catalog: PathBuf,
    },

    /// Append `RESIDUAL_BLOB` + `RESIDUAL_INDEX` to a lossy-only `.1bl` and rewrite the footer.
    AddResidual {
        /// Existing lossy-tier `.1bl`.
        #[arg(long = "in")]
        input: PathBuf,
        /// Arithmetic-coded residual blob (spec §"RESIDUAL_BLOB").
        #[arg(long)]
        residual: PathBuf,
        /// CBOR index: per-track byte offsets into the blob.
        #[arg(long)]
        index: PathBuf,
        /// Output path. A fresh file is written; the input is untouched.
        #[arg(long)]
        out: PathBuf,
    },
}

fn run(cli: Cli) -> Result<()> {
    match cli.cmd {
        Cmd::Prepare { src_dir, out } => {
            let summary = onebit_ingest::prepare::prepare(&src_dir, &out)?;
            eprintln!(
                "prepared corpus: {} FLAC file(s), {} bytes total → {}",
                summary.flac_count,
                summary.total_bytes,
                out.display()
            );
            Ok(())
        }
        Cmd::Pack { model, manifest, cover, lyrics, out } => {
            let summary = onebit_ingest::pack::pack(
                &model,
                &manifest,
                cover.as_deref(),
                lyrics.as_deref(),
                &out,
            )?;
            eprintln!(
                "packed {} section(s), {} bytes → {}",
                summary.section_count,
                summary.total_bytes,
                out.display()
            );
            Ok(())
        }
        Cmd::Validate { catalog } => {
            let report = onebit_ingest::validate::validate(&catalog)?;
            println!("{report}");
            Ok(())
        }
        Cmd::AddResidual { input, residual, index, out } => {
            let summary = onebit_ingest::residual::add_residual(
                &input, &residual, &index, &out,
            )?;
            eprintln!(
                "appended residual ({} bytes) + index ({} bytes) → {}",
                summary.residual_bytes,
                summary.index_bytes,
                out.display()
            );
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
