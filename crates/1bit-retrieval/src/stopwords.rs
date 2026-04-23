//! Standard English stopword list (≈60 words). Short on purpose — we want
//! to keep technical terms that look common in English but are discriminative
//! in our domain (e.g. "is", "can", "no" stay out; "build", "install", "test"
//! stay in).

use std::collections::HashSet;
use std::sync::OnceLock;

static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do",
    "does", "doing", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "just", "me", "my", "no", "nor", "not", "of",
    "off", "on", "or", "our", "out", "over", "own", "same", "she", "so", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "up", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
];

fn set() -> &'static HashSet<&'static str> {
    SET.get_or_init(|| STOPWORDS.iter().copied().collect())
}

pub fn is_stopword(s: &str) -> bool {
    set().contains(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_words_are_stops() {
        assert!(is_stopword("the"));
        assert!(is_stopword("a"));
        assert!(is_stopword("is"));
    }

    #[test]
    fn technical_words_are_kept() {
        assert!(!is_stopword("install"));
        assert!(!is_stopword("build"));
        assert!(!is_stopword("ternary"));
        assert!(!is_stopword("amdgpu"));
        assert!(!is_stopword("gfx1151"));
    }
}
