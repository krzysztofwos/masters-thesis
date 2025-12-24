use std::collections::HashMap;

use crate::{
    menace::agent::MenaceAgent,
    tictactoe::{BoardState, Player, game_tree::build_reduced_game_tree},
};

#[derive(Debug, Clone)]
pub struct OptimalPolicy {
    pub value: i32,
    pub optimal_moves: Vec<usize>,
}

fn solve(state: BoardState, memo: &mut HashMap<String, OptimalPolicy>) -> OptimalPolicy {
    let key = state.encode();
    if let Some(policy) = memo.get(&key) {
        return policy.clone();
    }

    if state.is_terminal() {
        let value = match state.winner() {
            Some(Player::X) => 1,
            Some(Player::O) => -1,
            None => 0,
        };
        let policy = OptimalPolicy {
            value,
            optimal_moves: Vec::new(),
        };
        memo.insert(key, policy.clone());
        return policy;
    }

    let mut best_value = match state.to_move {
        Player::X => i32::MIN,
        Player::O => i32::MAX,
    };
    let mut best_moves: Vec<usize> = Vec::new();

    for mv in state.legal_moves() {
        let next_state = state
            .make_move(mv)
            .expect("legal move generation should not fail");
        let canonical_next = next_state.canonical();
        let child_policy = solve(canonical_next, memo);
        let child_value = child_policy.value;

        match state.to_move {
            Player::X => {
                if child_value > best_value {
                    best_value = child_value;
                    best_moves.clear();
                    best_moves.push(mv);
                } else if child_value == best_value {
                    best_moves.push(mv);
                }
            }
            Player::O => {
                if child_value < best_value {
                    best_value = child_value;
                    best_moves.clear();
                    best_moves.push(mv);
                } else if child_value == best_value {
                    best_moves.push(mv);
                }
            }
        }
    }

    let policy = OptimalPolicy {
        value: best_value,
        optimal_moves: best_moves,
    };
    memo.insert(key, policy.clone());
    policy
}

pub fn compute_optimal_policy() -> HashMap<String, OptimalPolicy> {
    let tree = build_reduced_game_tree(false, false);
    let mut memo: HashMap<String, OptimalPolicy> = HashMap::new();

    for state in tree.canonical_states.values() {
        let canonical_state = *state;
        solve(canonical_state, &mut memo);
    }

    memo
}

pub fn optimal_move_distribution() -> HashMap<String, HashMap<usize, f64>> {
    let policies = compute_optimal_policy();
    let tree = build_reduced_game_tree(false, false);

    let mut distributions = HashMap::new();

    for (key, state) in tree.canonical_states {
        if state.is_terminal() || state.to_move != Player::X {
            continue;
        }

        if state.occupied_count() % 2 == 1 {
            continue;
        }
        if let Some(policy) = policies.get(&key) {
            if policy.optimal_moves.is_empty() {
                continue;
            }
            let prob = 1.0 / policy.optimal_moves.len() as f64;
            let mut distribution = HashMap::new();
            for mv in &policy.optimal_moves {
                distribution.insert(*mv, prob);
            }
            distributions.insert(key.clone(), distribution);
        }
    }

    distributions
}

fn state_kl(
    agent: &MenaceAgent,
    state_key: &crate::types::CanonicalLabel,
    optimal_dist: &HashMap<usize, f64>,
) -> Option<f64> {
    let distribution = agent.canonical_distribution(state_key)?;
    let total: f64 = distribution.values().sum();
    if total <= 0.0 {
        return None;
    }

    let mut sum = 0.0;
    let mut moves: Vec<_> = optimal_dist.keys().copied().collect();
    moves.sort_unstable();
    for mv in moves {
        let opt_prob = *optimal_dist.get(&mv).unwrap_or(&0.0);
        if opt_prob == 0.0 {
            continue;
        }

        let agent_prob = distribution.get(&mv).copied().unwrap_or(0.0);
        if agent_prob <= 0.0 {
            return None;
        }

        sum += opt_prob * (opt_prob / agent_prob).ln();
    }

    Some(sum)
}

pub fn kl_divergence(agent: &MenaceAgent, optimal: &HashMap<String, HashMap<usize, f64>>) -> f64 {
    let mut total = 0.0;

    let mut states: Vec<_> = optimal.keys().collect();
    states.sort();
    for state_key_str in states {
        let optimal_dist = optimal.get(state_key_str).expect("state key should exist");
        let state_key = match crate::types::CanonicalLabel::parse(state_key_str) {
            Ok(label) => label,
            Err(_) => return f64::INFINITY,
        };
        match state_kl(agent, &state_key, optimal_dist) {
            Some(value) => total += value,
            None => return f64::INFINITY,
        }
    }

    total
}

pub fn kl_divergence_weighted(
    agent: &MenaceAgent,
    optimal: &HashMap<String, HashMap<usize, f64>>,
    weights: &HashMap<String, f64>,
) -> f64 {
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    let mut states: Vec<_> = weights.keys().collect();
    states.sort();
    for state_key_str in states {
        let weight = *weights
            .get(state_key_str)
            .expect("state key should exist in weights");
        if weight <= 0.0 {
            continue;
        }
        let optimal_dist = match optimal.get(state_key_str) {
            Some(dist) => dist,
            None => return f64::INFINITY,
        };

        let state_key = match crate::types::CanonicalLabel::parse(state_key_str) {
            Ok(label) => label,
            Err(_) => return f64::INFINITY,
        };

        match state_kl(agent, &state_key, optimal_dist) {
            Some(value) => {
                weighted_sum += value * weight;
                total_weight += weight;
            }
            None => return f64::INFINITY,
        }
    }

    if total_weight == 0.0 {
        0.0
    } else {
        weighted_sum / total_weight
    }
}
