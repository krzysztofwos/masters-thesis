//! Generative model for Active Inference game tree evaluation
//!
//! This module implements the core GenerativeModel that performs game tree evaluation
//! using Active Inference principles.

use std::{cell::RefCell, collections::HashMap};

use super::{
    evaluation::{
        ActionEvaluation, ExactStateSummary, OpponentActionEvaluation, OpponentStateSummary,
        OutcomeDistribution, StateValue,
    },
    opponents::Opponent,
    preferences::PreferenceModel,
    state::{ActionEdge, StateNode},
    types::{OpponentKind, TieBreak},
};
use crate::{
    beliefs::Beliefs,
    efe::exact_policy_from_risk_ep,
    menace::optimal::{OptimalPolicy, compute_optimal_policy},
    tictactoe::{BoardState, Player, build_reduced_game_tree},
};

/// Generative model for Active Inference in Tic-Tac-Toe
///
/// # Thread Safety
///
/// This struct uses `RefCell<HashMap>` for internal caching (`minimax_cache`),
/// which provides interior mutability but is **not thread-safe**. The generative
/// model is designed for single-threaded use.
///
/// If you need to parallelize game tree evaluation across multiple threads:
/// - Replace `RefCell<HashMap>` with `RwLock<HashMap>` or `Mutex<HashMap>`
/// - Or create separate `GenerativeModel` instances per thread (recommended)
///
/// Creating separate instances is often preferable because:
/// 1. Avoids lock contention
/// 2. Allows parallel cache warming
/// 3. Simpler error handling (no poisoning)
#[derive(Debug, Clone)]
pub struct GenerativeModel {
    pub(crate) nodes: HashMap<String, StateNode>,
    root: String,
    optimal: HashMap<String, OptimalPolicy>,
    /// Cache for minimax outcome distributions. Uses RefCell for interior mutability.
    /// Not thread-safe - see struct-level docs for parallelization strategies.
    minimax_cache: RefCell<HashMap<String, OutcomeDistribution>>,
    tie_break: TieBreak,
}

impl Default for GenerativeModel {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerativeModel {
    /// Creates a new generative model with default tie-breaking
    pub fn new() -> Self {
        Self::with_tie_break(TieBreak::Uniform)
    }

    /// Creates a generative model with specified tie-breaking strategy
    pub fn with_tie_break(tie_break: TieBreak) -> Self {
        let tree = build_reduced_game_tree(false, false);
        let mut nodes = HashMap::new();

        for label in tree.states.iter() {
            let label_str = label.to_string();
            let Some(state) = tree.canonical_states.get(&label_str) else {
                continue;
            };

            let outcome = if state.is_terminal() {
                state.winner()
            } else {
                None
            };

            let legal_moves = state.legal_moves();
            let mut actions = Vec::with_capacity(legal_moves.len());
            // Only add actions for non-terminal states
            if !state.is_terminal() {
                for mv in &legal_moves {
                    let next_state = state
                        .make_move(*mv)
                        .expect("legal moves should always succeed");
                    let next_label = next_state.canonical().encode();
                    actions.push(ActionEdge {
                        action: *mv,
                        next_label,
                    });
                }
            }

            nodes.insert(
                label_str,
                StateNode {
                    state: *state,
                    outcome,
                    actions,
                },
            );
        }

        let root = BoardState::new().canonical().encode();
        let optimal = compute_optimal_policy();

        Self {
            nodes,
            root,
            optimal,
            minimax_cache: RefCell::new(HashMap::new()),
            tie_break,
        }
    }

    /// Returns the root state label
    pub fn root(&self) -> &str {
        &self.root
    }

    /// Returns the number of legal actions from a state
    pub fn legal_action_count(&self, state_label: &str) -> usize {
        self.nodes
            .get(state_label)
            .map(|node| node.actions.len())
            .unwrap_or(0)
    }

    /// Evaluates all actions from a state
    pub fn evaluate_actions(
        &self,
        state_label: &str,
        preferences: &PreferenceModel,
        opponent: &dyn Opponent,
        beliefs: &Beliefs,
        agent_player: Player,
    ) -> Vec<ActionEvaluation> {
        let node = self.nodes.get(state_label).unwrap_or_else(|| {
            panic!("Internal error: state label '{state_label}' not found in generative model")
        });

        assert!(
            !node.state.is_terminal(),
            "Internal error: cannot evaluate terminal state"
        );
        let agent_preferences = preferences.for_player(agent_player);
        let prior_weights = agent_preferences.policy_prior_weights(node);

        if node.state.to_move == agent_player {
            // Agent to move: compute evaluations in their perspective
            let mut evaluations = self.evaluate_actions_for_agent(
                node,
                &agent_preferences,
                opponent,
                beliefs,
                &prior_weights,
                agent_player,
            );
            evaluations.sort_by(|a, b| {
                a.free_energy
                    .partial_cmp(&b.free_energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            evaluations
        } else {
            // Opponent to move: mirror results to agent perspective
            let mut mapped = self.evaluate_actions_for_opponent(
                node,
                &agent_preferences,
                opponent,
                beliefs,
                &prior_weights,
                agent_player,
            );
            mapped.sort_by(|a, b| {
                a.free_energy
                    .partial_cmp(&b.free_energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            mapped
        }
    }

    fn evaluate_actions_for_agent(
        &self,
        node: &StateNode,
        preferences: &PreferenceModel,
        opponent: &dyn Opponent,
        beliefs: &Beliefs,
        prior_weights: &[f64],
        agent_player: Player,
    ) -> Vec<ActionEvaluation> {
        let opponent_kind = opponent.kind();
        let mut evaluations = Vec::with_capacity(node.actions.len());
        let mut cache = HashMap::new();

        for (idx, edge) in node.actions.iter().enumerate() {
            let child = self.evaluate_state_internal(
                &edge.next_label,
                preferences,
                opponent,
                beliefs,
                &mut cache,
                agent_player,
            );
            let distribution = child.distribution;
            let risk = distribution.expected_risk(preferences);
            let epistemic = child.epistemic;
            let ambiguity = child.ambiguity;
            let free_energy = child
                .expected_free_energy(preferences.epistemic_weight, preferences.ambiguity_weight);

            let opponent_eig = if opponent_kind == OpponentKind::Uniform {
                self.compute_opponent_eig(&edge.next_label, beliefs, agent_player)
            } else {
                0.0
            };

            evaluations.push(ActionEvaluation {
                action: edge.action,
                next_state: edge.next_label.clone(),
                free_energy,
                risk,
                epistemic,
                ambiguity,
                outcome_distribution: distribution,
                opponent_eig,
                policy_prior: prior_weights[idx],
            });
        }

        evaluations
    }

    fn evaluate_actions_for_opponent(
        &self,
        node: &StateNode,
        preferences: &PreferenceModel,
        opponent: &dyn Opponent,
        beliefs: &Beliefs,
        prior_weights: &[f64],
        agent_player: Player,
    ) -> Vec<ActionEvaluation> {
        let opponent_kind = opponent.kind();
        let mut evaluations = Vec::with_capacity(node.actions.len());
        let mut cache = HashMap::new();

        for (idx, edge) in node.actions.iter().enumerate() {
            let child = self.evaluate_state_internal(
                &edge.next_label,
                preferences,
                opponent,
                beliefs,
                &mut cache,
                agent_player,
            );

            let distribution = child.distribution;
            let risk = distribution.expected_risk(preferences);
            let epistemic = child.epistemic;
            let free_energy = child
                .expected_free_energy(preferences.epistemic_weight, preferences.ambiguity_weight);
            let opponent_eig = if opponent_kind == OpponentKind::Uniform {
                self.compute_opponent_eig(&edge.next_label, beliefs, agent_player)
            } else {
                0.0
            };

            evaluations.push(ActionEvaluation {
                action: edge.action,
                next_state: edge.next_label.clone(),
                free_energy,
                risk,
                epistemic,
                ambiguity: child.ambiguity,
                outcome_distribution: distribution,
                opponent_eig,
                policy_prior: prior_weights[idx],
            });
        }

        evaluations
    }

    /// Computes exact state summary with optimal policy
    pub fn exact_state_summary(
        &self,
        state_label: &str,
        preferences: &PreferenceModel,
        opponent: &dyn Opponent,
        beliefs: &Beliefs,
        agent_player: Player,
    ) -> ExactStateSummary {
        let agent_preferences = preferences.for_player(agent_player);
        let actions =
            self.evaluate_actions(state_label, preferences, opponent, beliefs, agent_player);
        let risk: Vec<f64> = actions.iter().map(|a| a.risk).collect();
        let epistemic: Vec<f64> = actions.iter().map(|a| a.epistemic).collect();
        let prior: Vec<f64> = actions.iter().map(|a| a.policy_prior).collect();

        let policy = exact_policy_from_risk_ep(
            &risk,
            &epistemic,
            agent_preferences.epistemic_weight,
            agent_preferences.policy_lambda,
            &prior,
        );

        ExactStateSummary { actions, policy }
    }

    /// Computes minimax distribution for a state
    pub fn minimax_distribution(&self, state_label: &str) -> OutcomeDistribution {
        if let Some(cached) = self.minimax_cache.borrow().get(state_label) {
            return *cached;
        }

        let node = self
            .nodes
            .get(state_label)
            .unwrap_or_else(|| panic!("Unknown state label: {state_label}"));

        // Check if state is terminal (handles both wins and draws)
        if node.state.is_terminal() {
            return OutcomeDistribution::terminal(node.outcome);
        }

        let optimal_policy = self.optimal.get(state_label);
        let moves = optimal_policy
            .map(|p| &p.optimal_moves)
            .filter(|moves| !moves.is_empty());

        let move_list = if let Some(moves) = moves {
            moves.clone()
        } else {
            node.actions.iter().map(|edge| edge.action).collect()
        };

        assert!(
            !move_list.is_empty(),
            "Non-terminal state must have actions"
        );

        let weight = match self.tie_break {
            TieBreak::Uniform => 1.0 / move_list.len() as f64,
        };

        let mut distribution = OutcomeDistribution::zero();
        for mv in move_list {
            let edge = node
                .actions
                .iter()
                .find(|action| action.action == mv)
                .unwrap_or_else(|| panic!("Action {mv} not found"));
            let child = self.minimax_distribution(&edge.next_label);
            distribution.add_weighted(&child, weight);
        }

        self.minimax_cache
            .borrow_mut()
            .insert(state_label.to_string(), distribution);
        distribution
    }

    /// Evaluates opponent state
    pub fn opponent_state_summary(
        &self,
        state_label: &str,
        preferences: &PreferenceModel,
        opponent: &dyn Opponent,
        beliefs: &Beliefs,
        agent_player: Player,
    ) -> OpponentStateSummary {
        let node = self
            .nodes
            .get(state_label)
            .unwrap_or_else(|| panic!("Unknown state label: {state_label}"));

        assert!(!node.state.is_terminal(), "Cannot evaluate terminal state");
        let agent_preferences = preferences.for_player(agent_player);
        let opponent_player = agent_player.opponent();
        assert_eq!(
            node.state.to_move, opponent_player,
            "Cannot summarize opponent state when agent is to move"
        );

        let mut cache = HashMap::new();

        match opponent.kind() {
            OpponentKind::Uniform => {
                let (predictive, eig) = opponent
                    .predictive_and_eig(self, state_label, beliefs)
                    .expect("Uniform opponent must provide predictive weights");

                let actions = self.evaluate_opponent_actions_with_weights(
                    node,
                    &agent_preferences,
                    opponent,
                    beliefs,
                    &mut cache,
                    agent_player,
                    |idx, _edge| predictive[idx],
                );

                OpponentStateSummary {
                    information_gain: eig,
                    actions,
                }
            }
            OpponentKind::Adversarial => {
                // First pass: evaluate all children and find the best EFE
                let mut best = f64::NEG_INFINITY;
                let mut efes = Vec::with_capacity(node.actions.len());

                for edge in &node.actions {
                    let child = self.evaluate_state_internal(
                        &edge.next_label,
                        &agent_preferences,
                        opponent,
                        beliefs,
                        &mut cache,
                        agent_player,
                    );
                    let efe = child.expected_free_energy(
                        preferences.epistemic_weight,
                        preferences.ambiguity_weight,
                    );

                    if efe > best {
                        best = efe;
                    }
                    efes.push(efe);
                }

                // Calculate uniform weight for best actions
                let winner_count = efes
                    .iter()
                    .filter(|efe| (**efe - best).abs() <= 1e-9)
                    .count();
                let weight = if winner_count == 0 {
                    0.0
                } else {
                    1.0 / winner_count as f64
                };

                // Second pass: assign weights based on best EFE
                let actions = self.evaluate_opponent_actions_with_weights(
                    node,
                    &agent_preferences,
                    opponent,
                    beliefs,
                    &mut cache,
                    agent_player,
                    |idx, _edge| {
                        if (efes[idx] - best).abs() <= 1e-9 {
                            weight
                        } else {
                            0.0
                        }
                    },
                );

                OpponentStateSummary {
                    information_gain: 0.0,
                    actions,
                }
            }
            OpponentKind::Minimax => {
                let optimal_moves = self
                    .optimal
                    .get(state_label)
                    .map(|policy| {
                        if policy.optimal_moves.is_empty() {
                            node.actions.iter().map(|edge| edge.action).collect()
                        } else {
                            policy.optimal_moves.clone()
                        }
                    })
                    .unwrap_or_else(|| node.actions.iter().map(|edge| edge.action).collect());

                let weight = if optimal_moves.is_empty() {
                    0.0
                } else {
                    1.0 / optimal_moves.len() as f64
                };

                let actions = node
                    .actions
                    .iter()
                    .map(|edge| {
                        if optimal_moves.contains(&edge.action) {
                            let distribution = self.minimax_distribution(&edge.next_label);
                            let risk = distribution.expected_risk(&agent_preferences);
                            OpponentActionEvaluation::from_edge(edge).with_distribution(
                                distribution,
                                weight,
                                risk,
                            )
                        } else {
                            OpponentActionEvaluation::from_edge(edge)
                        }
                    })
                    .collect();

                OpponentStateSummary {
                    information_gain: 0.0,
                    actions,
                }
            }
        }
    }

    /// Computes opponent's expected information gain
    fn compute_opponent_eig(
        &self,
        state_label: &str,
        beliefs: &Beliefs,
        agent_player: Player,
    ) -> f64 {
        let Some(node) = self.nodes.get(state_label) else {
            return 0.0;
        };
        let opponent_player = agent_player.opponent();
        if node.outcome.is_some() || node.state.to_move != opponent_player {
            return 0.0;
        }
        let legal = node.actions.len();
        if legal < 2 {
            return 0.0;
        }
        beliefs.opponent_eig(state_label, legal)
    }

    /// Helper to cache a value and return it
    fn cache_and_return(
        cache: &mut HashMap<String, StateValue>,
        state_label: &str,
        value: StateValue,
    ) -> StateValue {
        cache.insert(state_label.to_string(), value.clone());
        value
    }

    /// Evaluates opponent actions with provided weight function
    #[allow(clippy::too_many_arguments)]
    fn evaluate_opponent_actions_with_weights<F>(
        &self,
        node: &StateNode,
        preferences: &PreferenceModel,
        opponent: &dyn Opponent,
        beliefs: &Beliefs,
        cache: &mut HashMap<String, StateValue>,
        agent_player: Player,
        weight_fn: F,
    ) -> Vec<OpponentActionEvaluation>
    where
        F: Fn(usize, &ActionEdge) -> f64,
    {
        node.actions
            .iter()
            .enumerate()
            .map(|(idx, edge)| {
                let child = self.evaluate_state_internal(
                    &edge.next_label,
                    preferences,
                    opponent,
                    beliefs,
                    cache,
                    agent_player,
                );
                let weight = weight_fn(idx, edge);
                OpponentActionEvaluation::from_edge(edge).with_state_value(
                    child,
                    weight,
                    preferences.epistemic_weight,
                    preferences.ambiguity_weight,
                )
            })
            .collect()
    }

    /// Finds the best child state based on a comparison function
    #[allow(clippy::too_many_arguments)]
    fn find_best_child<F>(
        &self,
        node: &StateNode,
        preferences: &PreferenceModel,
        opponent: &dyn Opponent,
        beliefs: &Beliefs,
        cache: &mut HashMap<String, StateValue>,
        agent_player: Player,
        compare_fn: F,
    ) -> StateValue
    where
        F: Fn(f64, f64) -> bool,
    {
        let mut best: Option<(StateValue, f64)> = None;

        for edge in &node.actions {
            let child = self.evaluate_state_internal(
                &edge.next_label,
                preferences,
                opponent,
                beliefs,
                cache,
                agent_player,
            );
            let efe = child
                .expected_free_energy(preferences.epistemic_weight, preferences.ambiguity_weight);

            best = match best {
                None => Some((child, efe)),
                Some((current, current_efe)) => {
                    if compare_fn(efe, current_efe) {
                        Some((child, efe))
                    } else {
                        Some((current, current_efe))
                    }
                }
            };
        }

        best.expect("Non-terminal state must have actions").0
    }

    /// Internal recursive state evaluation
    fn evaluate_state_internal(
        &self,
        state_label: &str,
        preferences: &PreferenceModel,
        opponent: &dyn Opponent,
        beliefs: &Beliefs,
        cache: &mut HashMap<String, StateValue>,
        agent_player: Player,
    ) -> StateValue {
        if let Some(cached) = cache.get(state_label) {
            return cached.clone();
        }

        let node = self
            .nodes
            .get(state_label)
            .unwrap_or_else(|| panic!("Unknown state label: {state_label}"));

        if let Some(winner) = node.outcome {
            let value = StateValue::terminal(Some(winner), preferences);
            return Self::cache_and_return(cache, state_label, value);
        }

        if node.outcome.is_none() && node.actions.is_empty() {
            let value = StateValue::terminal(None, preferences);
            return Self::cache_and_return(cache, state_label, value);
        }

        let value = if node.state.to_move == agent_player {
            // Minimize expected free energy
            self.find_best_child(
                node,
                preferences,
                opponent,
                beliefs,
                cache,
                agent_player,
                |child_efe, current_efe| child_efe < current_efe,
            )
        } else {
            match opponent.kind() {
                OpponentKind::Uniform => {
                    let (predictive, eig) = opponent
                        .predictive_and_eig(self, state_label, beliefs)
                        .expect("Uniform opponent must provide predictive weights");

                    let mut risk = 0.0;
                    let mut epistemic = 0.0;
                    let mut distribution = OutcomeDistribution::zero();

                    for (idx, edge) in node.actions.iter().enumerate() {
                        let child = self.evaluate_state_internal(
                            &edge.next_label,
                            preferences,
                            opponent,
                            beliefs,
                            cache,
                            agent_player,
                        );
                        let weight = predictive[idx];
                        risk += weight * child.risk;
                        epistemic += weight * child.epistemic;
                        distribution.add_weighted(&child.distribution, weight);
                    }

                    epistemic += eig;

                    // Compute ambiguity as entropy of the weighted outcome distribution
                    let ambiguity = distribution.ambiguity();

                    StateValue {
                        risk,
                        epistemic,
                        ambiguity,
                        distribution,
                    }
                }
                OpponentKind::Adversarial => {
                    // Maximize expected free energy (worst case for X)
                    self.find_best_child(
                        node,
                        preferences,
                        opponent,
                        beliefs,
                        cache,
                        agent_player,
                        |child_efe, current_efe| child_efe > current_efe,
                    )
                }
                OpponentKind::Minimax => {
                    let distribution = self.minimax_distribution(state_label);
                    let ambiguity = distribution.ambiguity();
                    StateValue {
                        risk: distribution.expected_risk(preferences),
                        epistemic: 0.0,
                        ambiguity,
                        distribution,
                    }
                }
            }
        };

        Self::cache_and_return(cache, state_label, value)
    }
}
