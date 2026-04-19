/// LRU cache with sliding TTL for cache speculation.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct CachedBlockInfo {
    pub cumulative_tokens: u32,
    pub hit_count: u32,
}

struct LruEntry {
    value: CachedBlockInfo,
    last_accessed: Instant,
}

pub struct LruCache {
    entries: HashMap<String, LruEntry>,
    order: VecDeque<String>,
    max_size: usize,
    ttl: Duration,
}

impl LruCache {
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            entries: HashMap::with_capacity(max_size),
            order: VecDeque::with_capacity(max_size),
            max_size,
            ttl,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&CachedBlockInfo> {
        let now = Instant::now();
        if let Some(entry) = self.entries.get_mut(key) {
            if now.duration_since(entry.last_accessed) > self.ttl {
                self.remove(key);
                return None;
            }
            entry.last_accessed = now;
            entry.value.hit_count += 1;
            self.move_to_front(key);
            return self.entries.get(key).map(|e| &e.value);
        }
        None
    }

    pub fn has(&self, key: &str) -> bool {
        if let Some(entry) = self.entries.get(key) {
            Instant::now().duration_since(entry.last_accessed) <= self.ttl
        } else {
            false
        }
    }

    pub fn set(&mut self, key: String, cumulative_tokens: u32) {
        if self.entries.contains_key(&key) {
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.value.cumulative_tokens = cumulative_tokens;
                entry.last_accessed = Instant::now();
            }
            self.move_to_front(&key);
            return;
        }

        while self.entries.len() >= self.max_size {
            self.evict_oldest();
        }

        self.entries.insert(
            key.clone(),
            LruEntry {
                value: CachedBlockInfo {
                    cumulative_tokens,
                    hit_count: 0,
                },
                last_accessed: Instant::now(),
            },
        );
        self.order.push_front(key);
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }

    pub fn cleanup_expired(&mut self) {
        let now = Instant::now();
        let expired_keys: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, entry)| now.duration_since(entry.last_accessed) > self.ttl)
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            self.remove(&key);
        }
    }

    fn remove(&mut self, key: &str) {
        self.entries.remove(key);
        self.order.retain(|k| k != key);
    }

    fn move_to_front(&mut self, key: &str) {
        self.order.retain(|k| k != key);
        self.order.push_front(key.to_string());
    }

    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self.order.pop_back() {
            self.entries.remove(&oldest_key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get() {
        let mut cache = LruCache::new(10, Duration::from_secs(300));
        cache.set("key1".to_string(), 100);
        let info = cache.get("key1");
        assert!(info.is_some());
        assert_eq!(info.unwrap().cumulative_tokens, 100);
    }

    #[test]
    fn test_eviction() {
        let mut cache = LruCache::new(2, Duration::from_secs(300));
        cache.set("a".to_string(), 10);
        cache.set("b".to_string(), 20);
        cache.set("c".to_string(), 30);
        assert_eq!(cache.size(), 2);
        assert!(!cache.has("a"));
        assert!(cache.has("b"));
        assert!(cache.has("c"));
    }

    #[test]
    fn test_has() {
        let mut cache = LruCache::new(10, Duration::from_secs(300));
        assert!(!cache.has("missing"));
        cache.set("key".to_string(), 50);
        assert!(cache.has("key"));
    }

    #[test]
    fn test_update_existing() {
        let mut cache = LruCache::new(10, Duration::from_secs(300));
        cache.set("key".to_string(), 100);
        cache.set("key".to_string(), 200);
        assert_eq!(cache.size(), 1);
        let info = cache.get("key").unwrap();
        assert_eq!(info.cumulative_tokens, 200);
    }
}
