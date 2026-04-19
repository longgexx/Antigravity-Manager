/// Multi-session cache estimator manager.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use super::estimator::{CacheEstimator, EstimatorConfig};

struct EstimatorEntry {
    estimator: CacheEstimator,
    last_used: Instant,
}

pub struct ManagerConfig {
    pub max_estimators: usize,
    pub estimator_ttl: Duration,
    pub estimator_config: EstimatorConfig,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            max_estimators: 200,
            estimator_ttl: Duration::from_secs(600),
            estimator_config: EstimatorConfig::default(),
        }
    }
}

pub struct CacheManager {
    estimators: HashMap<String, EstimatorEntry>,
    config: ManagerConfig,
}

impl CacheManager {
    pub fn new(config: ManagerConfig) -> Self {
        Self {
            estimators: HashMap::with_capacity(config.max_estimators),
            config,
        }
    }

    pub fn get_estimator(&mut self, model: &str, session_id: &str) -> &mut CacheEstimator {
        let key = format!("{}:{}", model, session_id);

        if !self.estimators.contains_key(&key) {
            if self.estimators.len() >= self.config.max_estimators {
                self.evict_oldest();
            }

            self.estimators.insert(
                key.clone(),
                EstimatorEntry {
                    estimator: CacheEstimator::new(EstimatorConfig {
                        cache_ttl: self.config.estimator_config.cache_ttl,
                        max_cache_size: self.config.estimator_config.max_cache_size,
                        enabled: self.config.estimator_config.enabled,
                    }),
                    last_used: Instant::now(),
                },
            );
        }

        let entry = self.estimators.get_mut(&key).unwrap();
        entry.last_used = Instant::now();
        &mut entry.estimator
    }

    pub fn cleanup(&mut self) {
        let now = Instant::now();
        self.estimators.retain(|_, entry| {
            now.duration_since(entry.last_used) < self.config.estimator_ttl
        });
    }

    pub fn estimator_count(&self) -> usize {
        self.estimators.len()
    }

    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .estimators
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone())
        {
            self.estimators.remove(&oldest_key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_or_create() {
        let mut mgr = CacheManager::new(ManagerConfig::default());
        let _est = mgr.get_estimator("claude-sonnet", "session-1");
        assert_eq!(mgr.estimator_count(), 1);

        let _est = mgr.get_estimator("claude-sonnet", "session-1");
        assert_eq!(mgr.estimator_count(), 1);

        let _est = mgr.get_estimator("claude-opus", "session-2");
        assert_eq!(mgr.estimator_count(), 2);
    }

    #[test]
    fn test_max_estimators() {
        let mut mgr = CacheManager::new(ManagerConfig {
            max_estimators: 2,
            ..Default::default()
        });
        mgr.get_estimator("m1", "s1");
        mgr.get_estimator("m2", "s2");
        mgr.get_estimator("m3", "s3");
        assert_eq!(mgr.estimator_count(), 2);
    }
}
