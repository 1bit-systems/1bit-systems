//! Markdown-aware chunker.
//!
//! Algorithm:
//! 1. Walk the file line by line.
//! 2. Lines starting with `#` reset the "current heading" and also force a
//!    chunk boundary (the accumulated section buffer is flushed first).
//! 3. Inside a section, accumulate lines into a buffer. When the buffer
//!    exceeds `SOFT_CAP`, emit it as a chunk (rolling back to the last
//!    word/line boundary) and restart with a ~`OVERLAP` tail for context.
//!
//! Heading is attached to *every* chunk produced from its section.
//!
//! Pure-string impl — no markdown parser dep. Good enough for our wiki.

const SOFT_CAP: usize = 500;
const OVERLAP: usize = 50;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub heading: String,
    pub text: String,
}

pub fn chunk_markdown(src: &str) -> Vec<Chunk> {
    let mut out: Vec<Chunk> = Vec::new();
    let mut cur_heading = String::new();
    let mut buf = String::new();

    let flush = |buf: &mut String, heading: &str, out: &mut Vec<Chunk>| {
        let trimmed = buf.trim();
        if !trimmed.is_empty() {
            out.push(Chunk {
                heading: heading.to_string(),
                text: trimmed.to_string(),
            });
        }
        buf.clear();
    };

    for line in src.lines() {
        let stripped = line.trim_start();
        if let Some(rest) = parse_heading(stripped) {
            // New heading -> flush whatever we had, start fresh.
            flush(&mut buf, &cur_heading, &mut out);
            cur_heading = rest.to_string();
            continue;
        }

        buf.push_str(line);
        buf.push('\n');

        // Hard cap — split long sections.
        if buf.len() >= SOFT_CAP {
            let (emit, tail) = split_at_boundary(&buf, SOFT_CAP);
            let emit_trimmed = emit.trim();
            if !emit_trimmed.is_empty() {
                out.push(Chunk {
                    heading: cur_heading.clone(),
                    text: emit_trimmed.to_string(),
                });
            }
            // Keep a small tail for overlap / context continuity.
            let overlap_start = tail.len().saturating_sub(OVERLAP);
            // Land the overlap start on a word boundary too so we don't
            // cut mid-word.
            let overlap_start = find_word_boundary(tail, overlap_start);
            buf = tail[overlap_start..].to_string();
        }
    }
    flush(&mut buf, &cur_heading, &mut out);
    out
}

/// Parse `# foo`, `## foo`, `### foo`, ... Returns `Some(heading_text)` when
/// `line` (already-left-trimmed) is a markdown heading, else `None`.
///
/// Rejects `#foo` (no space) and fenced-code-block-style lines.
fn parse_heading(line: &str) -> Option<&str> {
    let mut hashes = 0usize;
    for ch in line.chars() {
        if ch == '#' {
            hashes += 1;
            if hashes > 6 {
                return None;
            }
        } else {
            break;
        }
    }
    if hashes == 0 {
        return None;
    }
    let rest = &line[hashes..];
    // A heading needs at least one space after the hashes.
    if !rest.starts_with(' ') && !rest.starts_with('\t') {
        return None;
    }
    Some(rest.trim())
}

/// Given a buffer that has exceeded the cap, find the best place to split:
/// the last newline before the cap, else the last space, else the cap itself.
/// Returns `(emit, tail)` where `emit + tail == buf`.
fn split_at_boundary(buf: &str, cap: usize) -> (&str, &str) {
    // Snap `cap` down to a valid UTF-8 char boundary so slicing is safe
    // even in chunks that contain multi-byte glyphs (wiki has diagrams
    // with box-drawing characters).
    let mut safe_cap = cap.min(buf.len());
    while safe_cap > 0 && !buf.is_char_boundary(safe_cap) {
        safe_cap -= 1;
    }
    if let Some(nl) = buf[..safe_cap].rfind('\n')
        && nl > 0
    {
        return buf.split_at(nl + 1);
    }
    if let Some(sp) = buf[..safe_cap].rfind(' ')
        && sp > 0
    {
        return buf.split_at(sp + 1);
    }
    buf.split_at(safe_cap)
}

/// Walk backwards from `start` until we find a whitespace char; return the
/// index just after it so we don't slice into a word. Returns a safe
/// char-boundary if no whitespace is found within the first 64 bytes.
fn find_word_boundary(s: &str, start: usize) -> usize {
    let mut start = start.min(s.len());
    while start > 0 && !s.is_char_boundary(start) {
        start -= 1;
    }
    if start == 0 || start == s.len() {
        return start;
    }
    let bytes = s.as_bytes();
    let min = start.saturating_sub(64);
    let mut i = start;
    while i > min {
        // Step backward one full codepoint at a time.
        i -= 1;
        while i > 0 && !s.is_char_boundary(i) {
            i -= 1;
        }
        let b = bytes[i];
        if b == b' ' || b == b'\n' || b == b'\t' {
            // i+1 lands just after the whitespace byte, which is a
            // valid char boundary because whitespace bytes are 1 byte.
            return i + 1;
        }
    }
    start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heading_resets_chunk() {
        let src = "# One\n\nalpha alpha alpha\n\n# Two\n\nbeta beta beta\n";
        let chunks = chunk_markdown(src);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading, "One");
        assert!(chunks[0].text.contains("alpha"));
        assert_eq!(chunks[1].heading, "Two");
        assert!(chunks[1].text.contains("beta"));
    }

    #[test]
    fn long_section_splits_on_word_boundary() {
        let lorem = "word ".repeat(200); // ~1000 chars, well over cap
        let src = format!("# Big\n\n{}", lorem);
        let chunks = chunk_markdown(&src);
        assert!(chunks.len() >= 2, "expected split, got {}", chunks.len());
        for c in &chunks {
            // No chunk should start or end with a half-word (i.e. "wor" / "ord").
            let last_word = c.text.split_whitespace().last().unwrap_or("");
            assert!(
                last_word == "word" || last_word.is_empty(),
                "mid-word cut detected: tail {:?}",
                last_word
            );
        }
    }

    #[test]
    fn non_heading_hash_lines_are_content() {
        // `#foo` (no space) is not a heading.
        let src = "# Title\n\n#notaheading and some text\n";
        let chunks = chunk_markdown(src);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading, "Title");
        assert!(chunks[0].text.contains("#notaheading"));
    }

    #[test]
    fn nested_heading_levels_tracked() {
        let src = "# A\n\ntext A\n\n## A.1\n\ntext A.1\n\n### A.1.a\n\ntext deep\n";
        let chunks = chunk_markdown(src);
        let headings: Vec<&str> = chunks.iter().map(|c| c.heading.as_str()).collect();
        assert_eq!(headings, vec!["A", "A.1", "A.1.a"]);
    }

    #[test]
    fn multibyte_glyphs_do_not_panic_at_cap() {
        // Box-drawing characters are 3 bytes in UTF-8. A run that lands
        // the cap in the middle of one used to panic — regression test.
        let glyph = "─"; // U+2500, 3 bytes
        let line: String = glyph.repeat(400);
        let src = format!("# H\n\n{}\n", line);
        // Must not panic.
        let chunks = chunk_markdown(&src);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn content_before_first_heading_gets_empty_heading() {
        let src = "preamble text\n\nmore preamble\n\n# Real\n\nbody\n";
        let chunks = chunk_markdown(src);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading, "");
        assert!(chunks[0].text.contains("preamble"));
        assert_eq!(chunks[1].heading, "Real");
    }
}
