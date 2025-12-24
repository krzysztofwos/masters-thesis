//! Belief state over opponent policies for Active Inference analysis.
//!
//! Stores per-canonical-state Dirichlet counts for O-turn actions and exposes
//! predictive distributions, information gain, and streaming updates. The
//! container stays lightweight so it can be threaded through planners without
//! additional allocations when no O-moves have been observed yet.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::efe::dirichlet_categorical_mi;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Beliefs {
    opponent_alpha: BTreeMap<String, Vec<f64>>,
    default_alpha: f64,
    decay: f64,
    version: usize,
}

impl Beliefs {
    /// Create symmetric Dirichlet beliefs for all unseen O-states.
    pub fn symmetric(default_alpha: f64) -> Self {
        assert!(
            default_alpha.is_finite() && default_alpha > 0.0,
            "Dirichlet α must be positive"
        );
        Self {
            opponent_alpha: BTreeMap::new(),
            default_alpha,
            decay: 1.0,
            version: 0,
        }
    }

    /// Apply exponential forgetting when updating counts; 1.0 disables decay.
    pub fn with_decay(mut self, decay: f64) -> Self {
        assert!(
            decay.is_finite() && decay > 0.0 && decay <= 1.0,
            "Decay must lie in (0, 1]"
        );
        self.decay = decay;
        self
    }

    fn ensure_alpha(&self, state_label: &str, k: usize) -> Vec<f64> {
        if let Some(alpha) = self.opponent_alpha.get(state_label)
            && alpha.len() == k
        {
            return alpha.clone();
        }
        vec![self.default_alpha; k]
    }

    /// Return the Dirichlet α vector for this O-state, initialising if needed.
    pub fn alpha_for(&self, state_label: &str, k: usize) -> Vec<f64> {
        self.ensure_alpha(state_label, k)
    }

    /// Predictive distribution implied by the Dirichlet counts at this state.
    pub fn predictive(&self, state_label: &str, k: usize) -> Vec<f64> {
        let alpha = self.ensure_alpha(state_label, k);
        crate::utils::normalize_weights(alpha).unwrap_or_else(|| vec![1.0 / k as f64; k])
    }

    /// One-step expected information gain about the opponent policy.
    pub fn opponent_eig(&self, state_label: &str, k: usize) -> f64 {
        let alpha = self.ensure_alpha(state_label, k);
        dirichlet_categorical_mi(&alpha)
    }

    /// Observe an opponent action (in canonical coordinates) and update Dirichlet counts.
    pub fn observe_opponent_action(&mut self, state_label: &str, k: usize, action: usize) {
        let entry = self
            .opponent_alpha
            .entry(state_label.to_string())
            .or_insert_with(|| vec![self.default_alpha; k]);
        if entry.len() != k {
            *entry = vec![self.default_alpha; k];
        }
        if (self.decay - 1.0).abs() > f64::EPSILON {
            for value in entry.iter_mut() {
                *value *= self.decay;
            }
        }
        if action < k {
            entry[action] += 1.0;
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Monotone counter that changes whenever beliefs are updated.
    pub fn version(&self) -> usize {
        self.version
    }

    /// Number of canonical states with explicit opponent belief counts tracked so far.
    pub fn tracked_states(&self) -> usize {
        self.opponent_alpha.len()
    }
}

#[cfg(test)]
mod tests {
    use super::Beliefs;

    #[test]
    fn symmetric_predictive_updates_after_observation() {
        let mut beliefs = Beliefs::symmetric(1.0);
        let state = "state";
        let predictive_before = beliefs.predictive(state, 3);
        assert_eq!(predictive_before, vec![1.0 / 3.0; 3]);
        let version_before = beliefs.version();

        beliefs.observe_opponent_action(state, 3, 1);

        let predictive_after = beliefs.predictive(state, 3);
        assert!(predictive_after[1] > predictive_before[1]);
        assert_eq!(
            predictive_after.iter().sum::<f64>(),
            1.0,
            "Predictive distribution must remain normalized"
        );
        assert_ne!(version_before, beliefs.version());
    }
}
