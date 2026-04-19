/// Core cache speculation engine.
/// Mirrors kiro-go internal/cache/estimator.go algorithm.

use sha2::{Digest, Sha256};
use serde_json::Value;
use std::time::Duration;
use super::lru::LruCache;
use super::tokens::count_tokens;

const MAX_LOOKBACK_BLOCKS: usize = 20;

const IMAGE_TOKEN_ESTIMATE: u32 = 1600;

fn model_min_cache_tokens(model: &str) -> u32 {
    if model.contains("opus") {
        4096
    } else if model.contains("haiku") {
        4096
    } else {
        1024
    }
}

#[derive(Debug, Clone)]
pub enum BlockType {
    ThinkingPrefix,
    Tools,
    System,
    Message,
}

impl std::fmt::Display for BlockType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockType::ThinkingPrefix => write!(f, "thinking_prefix"),
            BlockType::Tools => write!(f, "tools"),
            BlockType::System => write!(f, "system"),
            BlockType::Message => write!(f, "message"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheBlock {
    pub block_type: BlockType,
    pub index: i32,
    pub content: String,
    pub tokens: u32,
    pub has_cache_control: bool,
    pub cumulative_hash: String,
    pub cumulative_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct Estimation {
    pub cache_read_tokens: u32,
    pub cache_creation_tokens: u32,
    pub uncached_tokens: u32,
    pub estimated: bool,
    pub source: String,
}

pub struct EstimatorConfig {
    pub cache_ttl: Duration,
    pub max_cache_size: usize,
    pub enabled: bool,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            cache_ttl: Duration::from_secs(300),
            max_cache_size: 200,
            enabled: true,
        }
    }
}

pub struct CacheEstimator {
    block_cache: LruCache,
    config: EstimatorConfig,
}

impl CacheEstimator {
    pub fn new(config: EstimatorConfig) -> Self {
        let block_cache = LruCache::new(config.max_cache_size, config.cache_ttl);
        Self {
            block_cache,
            config,
        }
    }

    pub fn estimate(&mut self, request: &Value, model: &str) -> Estimation {
        if !self.config.enabled {
            return Estimation {
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                uncached_tokens: 0,
                estimated: false,
                source: "disabled".to_string(),
            };
        }

        let mut blocks = self.build_block_list(request);
        if blocks.is_empty() {
            return Estimation {
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                uncached_tokens: 0,
                estimated: false,
                source: "empty_request".to_string(),
            };
        }

        compute_cumulative_hashes(&mut blocks);
        let breakpoints = find_breakpoints(&blocks);

        if breakpoints.is_empty() {
            return Estimation {
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                uncached_tokens: blocks.last().map(|b| b.cumulative_tokens).unwrap_or(0),
                estimated: true,
                source: "no_cache_control".to_string(),
            };
        }

        let total_cacheable_tokens = blocks[*breakpoints.last().unwrap()].cumulative_tokens;
        let min_tokens = model_min_cache_tokens(model);
        if total_cacheable_tokens < min_tokens {
            return Estimation {
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                uncached_tokens: blocks.last().map(|b| b.cumulative_tokens).unwrap_or(0),
                estimated: true,
                source: "below_threshold".to_string(),
            };
        }

        let (cache_read, last_matched) = self.check_cache_hits(&blocks, &breakpoints);

        self.store_blocks(&blocks, &breakpoints);

        let total_tokens = blocks.last().map(|b| b.cumulative_tokens).unwrap_or(0);
        let last_bp_tokens = blocks[*breakpoints.last().unwrap()].cumulative_tokens;

        let (cache_creation, source) = if cache_read > 0 {
            let creation = last_bp_tokens.saturating_sub(cache_read);
            let src = if creation > 0 {
                "cache_hit_partial"
            } else {
                "cache_hit"
            };
            (creation, src.to_string())
        } else {
            (last_bp_tokens, "cache_creation".to_string())
        };

        let uncached = total_tokens.saturating_sub(cache_read).saturating_sub(cache_creation);

        tracing::debug!(
            "[CacheSpeculation] source={}, read={}, creation={}, uncached={}, matched_block={:?}",
            source, cache_read, cache_creation, uncached, last_matched
        );

        Estimation {
            cache_read_tokens: cache_read,
            cache_creation_tokens: cache_creation,
            uncached_tokens: uncached,
            estimated: true,
            source,
        }
    }

    fn build_block_list(&self, request: &Value) -> Vec<CacheBlock> {
        let mut blocks = Vec::new();

        // 1. Thinking prefix (if thinking is enabled)
        if let Some(thinking) = request.get("thinking") {
            if thinking.get("type").and_then(|t| t.as_str()) == Some("enabled")
                || thinking.get("type").and_then(|t| t.as_str()) == Some("adaptive")
            {
                let budget = thinking
                    .get("budget_tokens")
                    .or_else(|| thinking.get("budgetTokens"))
                    .and_then(|b| b.as_u64())
                    .unwrap_or(0);
                let content = format!("thinking:enabled:budget:{}", budget);
                let tokens = count_tokens(&content);
                blocks.push(CacheBlock {
                    block_type: BlockType::ThinkingPrefix,
                    index: -1,
                    content,
                    tokens,
                    has_cache_control: false,
                    cumulative_hash: String::new(),
                    cumulative_tokens: 0,
                });
            }
        }

        // 2. Tools
        if let Some(tools) = request.get("tools").and_then(|t| t.as_array()) {
            if !tools.is_empty() {
                let content = serde_json::to_string(tools).unwrap_or_default();
                let tokens = count_tokens(&content);
                let has_cc = tools.iter().any(|t| t.get("cache_control").is_some());
                blocks.push(CacheBlock {
                    block_type: BlockType::Tools,
                    index: -1,
                    content,
                    tokens,
                    has_cache_control: has_cc,
                    cumulative_hash: String::new(),
                    cumulative_tokens: 0,
                });
            }
        }

        // 3. System
        if let Some(system) = request.get("system") {
            let (content, has_cc) = extract_system_content(system);
            if !content.is_empty() {
                let tokens = count_tokens(&content);
                blocks.push(CacheBlock {
                    block_type: BlockType::System,
                    index: -1,
                    content,
                    tokens,
                    has_cache_control: has_cc,
                    cumulative_hash: String::new(),
                    cumulative_tokens: 0,
                });
            }
        }

        // 4. Messages
        if let Some(messages) = request.get("messages").and_then(|m| m.as_array()) {
            for (i, msg) in messages.iter().enumerate() {
                let (content, has_cc) = extract_message_content(msg);
                let tokens = count_tokens(&content);
                blocks.push(CacheBlock {
                    block_type: BlockType::Message,
                    index: i as i32,
                    content,
                    tokens,
                    has_cache_control: has_cc,
                    cumulative_hash: String::new(),
                    cumulative_tokens: 0,
                });
            }
        }

        blocks
    }

    fn check_cache_hits(
        &mut self,
        blocks: &[CacheBlock],
        breakpoints: &[usize],
    ) -> (u32, Option<i32>) {
        if breakpoints.is_empty() || blocks.is_empty() {
            return (0, None);
        }

        let last_bp = *breakpoints.last().unwrap();
        let lookback_start = if last_bp >= MAX_LOOKBACK_BLOCKS {
            last_bp - MAX_LOOKBACK_BLOCKS
        } else {
            0
        };

        for i in (lookback_start..=last_bp).rev() {
            let hash = &blocks[i].cumulative_hash;
            if let Some(info) = self.block_cache.get(hash) {
                let cache_read = info.cumulative_tokens;
                return (cache_read, Some(i as i32));
            }
        }

        (0, None)
    }

    fn store_blocks(&mut self, blocks: &[CacheBlock], breakpoints: &[usize]) {
        if breakpoints.is_empty() {
            return;
        }

        let last_bp = *breakpoints.last().unwrap();
        for i in 0..=last_bp {
            if i < blocks.len() {
                self.block_cache.set(
                    blocks[i].cumulative_hash.clone(),
                    blocks[i].cumulative_tokens,
                );
            }
        }
    }
}

fn compute_cumulative_hashes(blocks: &mut [CacheBlock]) {
    let mut hasher_state = String::new();
    let mut cumulative_tokens = 0u32;

    for block in blocks.iter_mut() {
        let block_repr = format!("{}:{}", block.block_type, block.content);
        if !hasher_state.is_empty() {
            hasher_state.push('|');
        }
        hasher_state.push_str(&block_repr);

        let mut hasher = Sha256::new();
        hasher.update(hasher_state.as_bytes());
        let hash_bytes = hasher.finalize();
        block.cumulative_hash = format!("{:x}", hash_bytes);

        cumulative_tokens += block.tokens;
        block.cumulative_tokens = cumulative_tokens;
    }
}

fn find_breakpoints(blocks: &[CacheBlock]) -> Vec<usize> {
    blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| b.has_cache_control)
        .map(|(i, _)| i)
        .collect()
}

fn extract_system_content(system: &Value) -> (String, bool) {
    match system {
        Value::String(s) => (s.clone(), false),
        Value::Array(arr) => {
            let mut content = String::new();
            let mut has_cc = false;
            for block in arr {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(text);
                }
                if block.get("cache_control").is_some() {
                    has_cc = true;
                }
            }
            (content, has_cc)
        }
        _ => (String::new(), false),
    }
}

fn extract_message_content(msg: &Value) -> (String, bool) {
    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
    let mut parts = Vec::new();
    let mut has_cc = false;

    parts.push(format!("role:{}", role));

    match msg.get("content") {
        Some(Value::String(s)) => {
            parts.push(s.clone());
        }
        Some(Value::Array(arr)) => {
            for block in arr {
                if block.get("cache_control").is_some() {
                    has_cc = true;
                }
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match block_type {
                    "text" => {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            parts.push(text.to_string());
                        }
                    }
                    "thinking" => {
                        if let Some(thinking) = block.get("thinking").and_then(|t| t.as_str()) {
                            parts.push(format!("thinking:{}", thinking));
                        }
                    }
                    "tool_use" => {
                        if let Some(name) = block.get("name").and_then(|n| n.as_str()) {
                            parts.push(format!("tool_use:{}", name));
                        }
                        if let Some(input) = block.get("input") {
                            parts.push(serde_json::to_string(input).unwrap_or_default());
                        }
                    }
                    "tool_result" => {
                        if let Some(tool_use_id) = block.get("tool_use_id").and_then(|t| t.as_str()) {
                            parts.push(format!("tool_result:{}", tool_use_id));
                        }
                        if let Some(content) = block.get("content") {
                            match content {
                                Value::String(s) => parts.push(s.clone()),
                                Value::Array(arr) => {
                                    for item in arr {
                                        if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                            parts.push(text.to_string());
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    "image" => {
                        parts.push(format!("image:{}", IMAGE_TOKEN_ESTIMATE));
                    }
                    _ => {
                        parts.push(serde_json::to_string(block).unwrap_or_default());
                    }
                }
            }
        }
        _ => {}
    }

    (parts.join("|"), has_cc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_estimator() -> CacheEstimator {
        CacheEstimator::new(EstimatorConfig::default())
    }

    #[test]
    fn test_no_cache_control() {
        let mut est = make_estimator();
        let req = json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        });
        let result = est.estimate(&req, "claude-sonnet-4-20250514");
        assert_eq!(result.source, "no_cache_control");
        assert_eq!(result.cache_read_tokens, 0);
        assert_eq!(result.cache_creation_tokens, 0);
    }

    fn long_system_prompt() -> String {
        // Need >4096 chars to produce >1024 tokens (chars/4 heuristic)
        let base = "You are a highly capable AI assistant with expertise in software engineering, \
        mathematics, science, and general knowledge. You provide detailed, accurate, \
        and helpful responses. You follow instructions carefully and think step by step. ";
        base.repeat(25)
    }

    #[test]
    fn test_first_request_cache_creation() {
        let mut est = make_estimator();
        let req = json!({
            "model": "claude-sonnet-4-20250514",
            "system": [
                {"type": "text", "text": long_system_prompt(), "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        });
        let result = est.estimate(&req, "claude-sonnet-4-20250514");
        assert_eq!(result.source, "cache_creation");
        assert!(result.cache_creation_tokens > 0);
        assert_eq!(result.cache_read_tokens, 0);
    }

    #[test]
    fn test_cache_hit_on_second_request() {
        let mut est = make_estimator();
        let req = json!({
            "model": "claude-sonnet-4-20250514",
            "system": [
                {"type": "text", "text": long_system_prompt(), "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        });

        let _first = est.estimate(&req, "claude-sonnet-4-20250514");

        let req2 = json!({
            "model": "claude-sonnet-4-20250514",
            "system": [
                {"type": "text", "text": long_system_prompt(), "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "how are you?"}
            ]
        });

        let second = est.estimate(&req2, "claude-sonnet-4-20250514");
        assert!(
            second.source == "cache_hit" || second.source == "cache_hit_partial",
            "expected cache hit, got: {}",
            second.source
        );
        assert!(second.cache_read_tokens > 0);
    }

    #[test]
    fn test_empty_request() {
        let mut est = make_estimator();
        let req = json!({});
        let result = est.estimate(&req, "claude-sonnet-4-20250514");
        assert_eq!(result.source, "empty_request");
        assert!(!result.estimated);
    }

    #[test]
    fn test_below_threshold() {
        let mut est = make_estimator();
        // Very short system with cache_control — below 1024 token threshold
        let req = json!({
            "model": "claude-sonnet-4-20250514",
            "system": [
                {"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": "x"}
            ]
        });
        let result = est.estimate(&req, "claude-sonnet-4-20250514");
        assert_eq!(result.source, "below_threshold");
    }

    #[test]
    fn test_block_ordering() {
        let est = CacheEstimator::new(EstimatorConfig::default());
        let req = json!({
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}],
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        });
        let blocks = est.build_block_list(&req);
        assert!(blocks.len() >= 4);
        assert!(matches!(blocks[0].block_type, BlockType::ThinkingPrefix));
        assert!(matches!(blocks[1].block_type, BlockType::Tools));
        assert!(matches!(blocks[2].block_type, BlockType::System));
        assert!(matches!(blocks[3].block_type, BlockType::Message));
    }
}
