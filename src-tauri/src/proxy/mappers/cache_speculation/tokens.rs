/// Token counting utilities for cache speculation.
/// Uses tiktoken cl100k_base encoding (same as Claude) for accurate token counts.

use std::collections::HashMap;
use std::sync::Mutex;
use once_cell::sync::Lazy;
use tiktoken_rs::cl100k_base;

struct TokenCache {
    cache: HashMap<u64, u32>,
    max_size: usize,
}

impl TokenCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            max_size,
        }
    }

    fn get(&self, hash: u64) -> Option<u32> {
        self.cache.get(&hash).copied()
    }

    fn set(&mut self, hash: u64, tokens: u32) {
        if self.cache.len() >= self.max_size {
            self.cache.clear();
        }
        self.cache.insert(hash, tokens);
    }
}

static TOKEN_CACHE: Lazy<Mutex<TokenCache>> = Lazy::new(|| {
    Mutex::new(TokenCache::new(10_000))
});

static BPE: Lazy<tiktoken_rs::CoreBPE> = Lazy::new(|| {
    cl100k_base().expect("Failed to initialize cl100k_base tokenizer")
});

fn hash_text(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

pub fn count_tokens(text: &str) -> u32 {
    if text.is_empty() {
        return 0;
    }

    let h = hash_text(text);

    if let Ok(cache) = TOKEN_CACHE.lock() {
        if let Some(cached) = cache.get(h) {
            return cached;
        }
    }

    let tokens = BPE.encode_with_special_tokens(text).len() as u32;

    if let Ok(mut cache) = TOKEN_CACHE.lock() {
        cache.set(h, tokens);
    }

    tokens
}

pub fn count_tokens_json(value: &serde_json::Value) -> u32 {
    let text = serde_json::to_string(value).unwrap_or_default();
    count_tokens(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(count_tokens(""), 0);
    }

    #[test]
    fn test_hello_world() {
        let tokens = count_tokens("hello world");
        assert_eq!(tokens, 2); // cl100k_base: "hello" + " world"
    }

    #[test]
    fn test_chinese() {
        let tokens = count_tokens("你好世界");
        assert!(tokens > 0);
        assert!(tokens <= 8); // CJK chars typically 1-2 tokens each
    }

    #[test]
    fn test_code() {
        let tokens = count_tokens("fn main() { println!(\"Hello\"); }");
        assert!(tokens > 5 && tokens < 20);
    }

    #[test]
    fn test_cache_hit() {
        let text = "This is a test string for cache verification.";
        let first = count_tokens(text);
        let second = count_tokens(text);
        assert_eq!(first, second);
    }

    #[test]
    fn test_long_text() {
        let text = "You are a helpful AI assistant. ".repeat(200);
        let tokens = count_tokens(&text);
        // ~1400 tokens for 200 repetitions of 7-token sentence
        assert!(tokens > 1000, "Expected >1000 tokens, got {}", tokens);
    }
}
