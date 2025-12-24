//! Q-table implementation for temporal difference learning

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::CanonicalLabel;

/// Q-table mapping (state, action) pairs to Q-values
///
/// Uses canonical board states and move positions as keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QTable {
    /// Q-values: (canonical_state, action_position) -> Q-value
    q_values: HashMap<(String, usize), f64>,
    /// Learning rate α
    learning_rate: f64,
    /// Discount factor γ
    discount_factor: f64,
    /// Initial Q-value for unseen state-action pairs
    q_init: f64,
}

impl QTable {
    /// Create a new Q-table
    pub fn new(learning_rate: f64, discount_factor: f64, q_init: f64) -> Self {
        Self {
            q_values: HashMap::new(),
            learning_rate,
            discount_factor,
            q_init,
        }
    }

    /// Get Q-value for a state-action pair
    pub fn get(&self, state: &CanonicalLabel, action: usize) -> f64 {
        *self
            .q_values
            .get(&(state.as_str().to_string(), action))
            .unwrap_or(&self.q_init)
    }

    /// Set Q-value for a state-action pair
    pub fn set(&mut self, state: CanonicalLabel, action: usize, value: f64) {
        self.q_values.insert((state.into_string(), action), value);
    }

    /// Get maximum Q-value over legal actions in a state
    pub fn max_q(&self, state: &CanonicalLabel, legal_actions: &[usize]) -> f64 {
        legal_actions
            .iter()
            .map(|&action| self.get(state, action))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Select greedy action (highest Q-value) from legal actions
    pub fn greedy_action(&self, state: &CanonicalLabel, legal_actions: &[usize]) -> usize {
        legal_actions
            .iter()
            .map(|&action| (action, self.get(state, action)))
            .max_by(|(_, q1), (_, q2)| q1.partial_cmp(q2).unwrap())
            .map(|(action, _)| action)
            .unwrap_or(legal_actions[0])
    }

    /// Q-learning update: off-policy TD control
    ///
    /// Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    pub fn q_learning_update(
        &mut self,
        state: CanonicalLabel,
        action: usize,
        reward: f64,
        next_state: &CanonicalLabel,
        next_legal_actions: &[usize],
        done: bool,
    ) {
        let current_q = self.get(&state, action);
        let max_next_q = if done || next_legal_actions.is_empty() {
            0.0
        } else {
            self.max_q(next_state, next_legal_actions)
        };
        let td_target = reward + self.discount_factor * max_next_q;
        let td_error = td_target - current_q;
        let new_q = current_q + self.learning_rate * td_error;
        self.set(state, action, new_q);
    }

    /// SARSA update: on-policy TD control
    ///
    /// Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
    pub fn sarsa_update(
        &mut self,
        state: CanonicalLabel,
        action: usize,
        reward: f64,
        next_state: &CanonicalLabel,
        next_action: usize,
        done: bool,
    ) {
        let current_q = self.get(&state, action);
        let next_q = if done {
            0.0
        } else {
            self.get(next_state, next_action)
        };
        let td_target = reward + self.discount_factor * next_q;
        let td_error = td_target - current_q;
        let new_q = current_q + self.learning_rate * td_error;
        self.set(state, action, new_q);
    }

    /// Reset all Q-values (for episodic learning)
    pub fn reset(&mut self) {
        self.q_values.clear();
    }

    /// Get total number of Q-values stored
    pub fn size(&self) -> usize {
        self.q_values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qtable_initialization() {
        let qtable = QTable::new(0.5, 0.99, 0.0);
        let state = CanonicalLabel::parse("........._X").unwrap();
        assert_eq!(qtable.get(&state, 0), 0.0);
    }

    #[test]
    fn test_qtable_set_get() {
        let mut qtable = QTable::new(0.5, 0.99, 0.0);
        let state = CanonicalLabel::parse("........._X").unwrap();
        qtable.set(state.clone(), 4, 1.5);
        assert_eq!(qtable.get(&state, 4), 1.5);
    }

    #[test]
    fn test_max_q() {
        let mut qtable = QTable::new(0.5, 0.99, 0.0);
        let state = CanonicalLabel::parse("........._X").unwrap();
        qtable.set(state.clone(), 0, 0.5);
        qtable.set(state.clone(), 1, 1.5);
        qtable.set(state.clone(), 2, 0.8);

        let legal_actions = vec![0, 1, 2];
        assert_eq!(qtable.max_q(&state, &legal_actions), 1.5);
    }

    #[test]
    fn test_greedy_action() {
        let mut qtable = QTable::new(0.5, 0.99, 0.0);
        let state = CanonicalLabel::parse("........._X").unwrap();
        qtable.set(state.clone(), 0, 0.5);
        qtable.set(state.clone(), 1, 1.5);
        qtable.set(state.clone(), 2, 0.8);

        let legal_actions = vec![0, 1, 2];
        assert_eq!(qtable.greedy_action(&state, &legal_actions), 1);
    }

    #[test]
    fn test_q_learning_update() {
        let mut qtable = QTable::new(0.5, 0.99, 0.0);
        let state = CanonicalLabel::parse("........._X").unwrap();
        let next_state = CanonicalLabel::parse("X........_O").unwrap();

        // Set next state values
        qtable.set(next_state.clone(), 1, 1.0);
        qtable.set(next_state.clone(), 2, 2.0);

        // Q-learning update
        let next_legal = vec![1, 2];
        qtable.q_learning_update(state.clone(), 4, 0.0, &next_state, &next_legal, false);

        // Q(s,4) = 0.0 + 0.5 * (0.0 + 0.99 * 2.0 - 0.0) = 0.99
        let updated_q = qtable.get(&state, 4);
        assert!((updated_q - 0.99).abs() < 0.01);
    }

    #[test]
    fn test_sarsa_update() {
        let mut qtable = QTable::new(0.5, 0.99, 0.0);
        let state = CanonicalLabel::parse("........._X").unwrap();
        let next_state = CanonicalLabel::parse("X........_O").unwrap();

        // Set next state value for actual action taken
        qtable.set(next_state.clone(), 1, 1.5);

        // SARSA update
        qtable.sarsa_update(state.clone(), 4, 0.0, &next_state, 1, false);

        // Q(s,4) = 0.0 + 0.5 * (0.0 + 0.99 * 1.5 - 0.0) = 0.7425
        let updated_q = qtable.get(&state, 4);
        assert!((updated_q - 0.7425).abs() < 0.01);
    }
}
