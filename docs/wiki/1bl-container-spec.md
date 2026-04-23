# `.1bl` container format — spec v0.1

Status: draft, 2026-04-23.

Short for **1-bit lossless** when it carries a residual, or **1-bit listen** in lossy-only mode. Same file extension, inner sections tell you which.

## Goals

- Ship one catalog = one file. No tar-ball, no outer zip, no directory traversal at install time.
- Lossy-only subset readable by any client that can load a GGUF + a Mimi codec.
- Lossless-capable subset decodes byte-exact to a source FLAC without re-encoding.
- Forward-compatible: unknown section tags are skipped cleanly.
- Signed + hashed so tamper is visible.

## Layout

```
[magic]         "1BL\x01"                (4 B, version byte is last)
[header]        CBOR-encoded manifest    (u32 LE length + CBOR blob)
[section ...]   tag-length-value         (u8 tag + u64 LE length + blob)
[footer]        SHA-256 over everything  (32 B, covers magic → last section byte)
```

Version byte `\x01` = format v0.1. Any future incompat bump goes to `\x02`+.

## Header (CBOR manifest)

```cbor
{
  "v":        "0.1",
  "catalog":  "kevin-macleod",           ; ASCII slug, ^[a-z0-9-]{1,64}$
  "title":    "Kevin MacLeod catalog",
  "artist":   "Kevin MacLeod",
  "license":  "CC-BY-4.0",               ; SPDX id
  "license_url": "https://creativecommons.org/licenses/by/4.0/",
  "attribution": "Kevin MacLeod (incompetech.com)",
  "source":   "https://incompetech.com/music/royalty-free/",
  "created":  "2026-04-23T00:00:00Z",
  "tier":     "lossy",                   ; "lossy" | "premium" | "both"
  "codec":    {
    "audio": "mimi-12hz",                ; or "wan22-ti2v-5b" for video
    "sample_rate": 24000,
    "channels": 2
  },
  "model":    {
    "arch":   "bitnet-1p58",
    "params": 10_485_760,
    "bpw":    1.58,
    "sha256": "<64 hex>"                 ; of the weights section
  },
  "tracks":   [                           ; ordered, index-based
    { "id": "01", "title": "Electric Sheep", "length_ms": 215_000, "sha256": "<..>" },
    { "id": "02", ... }
  ],
  "residual_present": true,              ; true if lossless section included
  "residual_sha256": "<64 hex>"           ; of the residual section
}
```

CBOR chosen over JSON because: deterministic binary encoding, smaller header, standard IANA tag for content-type, easy to stream-parse.

## Section tags

| Tag (u8) | Name | Required? | Content |
|---|---|---|---|
| `0x01` | `MODEL_GGUF` | yes | Full GGUF binary — ternary LM weights, tokenizer, codec metadata |
| `0x02` | `COVER` | no | WebP / PNG artwork, ≤ 512 KB |
| `0x03` | `TRACK_LYRICS` | no | UTF-8 lyrics bundle, optionally per-track sectioned |
| `0x04` | `ATTRIBUTION_TXT` | yes | Plain-text attribution block; copied into player credits on display |
| `0x05` | `LICENSE_TXT` | yes | Full license text (CC BY 4.0 etc) |
| `0x10` | `RESIDUAL_BLOB` | premium only | Arithmetic-coded residual stream for bit-exact reconstruction |
| `0x11` | `RESIDUAL_INDEX` | premium only | CBOR index: per-track byte offsets into `RESIDUAL_BLOB` |
| `0x80`–`0xEF` | reserved for future use | — | skip if unknown |
| `0xF0`–`0xFF` | vendor extensions | — | ignored by baseline readers |

Baseline reader MUST load `0x01`, `0x04`, `0x05` before playback. MUST NOT play a track whose per-track `sha256` in the manifest doesn't match the emitted bytes (for the lossless path) or the model-generator output hash (for the lossy path — optional enforcement).

## Section order

```
MODEL_GGUF         (always first — everything else depends on it)
ATTRIBUTION_TXT
LICENSE_TXT
COVER              (optional, skippable)
TRACK_LYRICS       (optional)
RESIDUAL_BLOB      (premium-only, last — appended without rewriting earlier sections)
RESIDUAL_INDEX     (premium-only, immediately after RESIDUAL_BLOB)
```

Lossless residual sits at the tail so a Premium upgrade can be append-only: seek to end of a lossy-only `.1bl`, append `0x10` + `0x11`, rewrite the footer SHA-256. No need to rebuild the lossy-tier file.

## Signing

Footer hash covers the entire file. If the catalog is cryptographically signed by the publisher (we sign ours), the signature sits in a sidecar `.1bl.sig` using the same ed25519 key the publisher uses for their GitHub releases. Detached signature — doesn't change the `.1bl` bytes.

## Reader behaviour

Pseudocode:

```rust
let mut r = BufReader::new(File::open("kevin-macleod.1bl")?);
assert_eq!(r.read_u32_le()?, MAGIC_1BL);
assert_eq!(r.read_u8()?, 0x01);                 // version

let header_len = r.read_u32_le()?;
let header: Manifest = ciborium::from_reader(r.take(header_len as u64))?;

while let Some((tag, len)) = read_tlv_header(&mut r)? {
    match tag {
        MODEL_GGUF        => load_gguf_into_runtime(take_n(&mut r, len)?),
        ATTRIBUTION_TXT   => display_credits_on_first_play(take_n(&mut r, len)?),
        LICENSE_TXT       => surface_in_about_dialog(take_n(&mut r, len)?),
        RESIDUAL_BLOB     => stream_to_disk(take_n(&mut r, len)?),
        RESIDUAL_INDEX    => parse_residual_index(take_n(&mut r, len)?),
        tag if tag >= 0x80 => skip_bytes(&mut r, len)?,    // forward compat
        _ => bail!("unknown required tag {tag:#x}"),
    }
}

let footer_hash = r.read_u8_n(32)?;
assert_eq!(footer_hash, sha256_of_preceding_bytes);
```

## Sizing reference

A Kevin MacLeod pack (1 800 tracks, ~10 GB FLAC source):

| Section | Size |
|---|---|
| MODEL_GGUF (10 M ternary, 4 k ctx) | 2.0 MB |
| ATTRIBUTION_TXT + LICENSE_TXT | 8 KB |
| COVER (optional WebP 512×512) | 60 KB |
| TRACK_LYRICS (optional) | n/a — instrumentals |
| RESIDUAL_BLOB (lossless) | ~6.8 GB |
| RESIDUAL_INDEX | 30 KB |

Free tier pack ≈ **2.1 MB**. Premium pack ≈ **6.8 GB** (~32% below FLAC).

## Reference implementation

Lives in `crates/1bit-stream` (see companion doc). Reader + writer in Rust, zero-copy where possible, streaming-friendly so a Premium client can start playback before the residual finishes downloading.

## Versioning

- `v0.1` — this document. No backward compatibility obligations; format is pre-release.
- `v1.0` — frozen once we publish the first production Catalog and issue the trademark filings. Breaks require new magic bytes or a `v` bump in the manifest.
