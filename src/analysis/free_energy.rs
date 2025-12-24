//! Free Energy analysis for MENACE policies
//!
//! This module implements tools to compute the Free Energy of MENACE's learned
//! policies and track how it evolves during training. This validates the hypothesis
//! that reinforcement learning minimizes Free Energy according to the Free Energy
//! Principle.
//!
//! # Free Energy Definition
//!
//! For a policy π over actions at state s:
//!
//! ```text
//! F(π) = -E_π[ln P(o)] + KL[q(a|s) || p(a)]
//! ```
//!
//! Where:
//! - `E_π[ln P(o)]` = Expected log probability of outcomes (negative surprise)
//! - `KL[q||p]` = Divergence from prior policy (complexity cost)
//! - `q(a|s)` = Current policy (MENACE's normalized bead counts)
//! - `p(a)` = Prior policy (initial uniform distribution)
//!
//! # Opponent Models
//!
//! The expected surprise term depends on the opponent's policy. This module
//! supports multiple opponent models:
//! - Uniform: opponent plays randomly (original implementation)
//! - Workspace: opponent follows a learned policy from a MenaceWorkspace
//! - Optimal: opponent plays minimax optimal (via workspace)
//!
//! This enables analyzing Free Energy dynamics in co-evolutionary scenarios,
//! including "perfect vs perfect" analysis where both agents use optimal policies.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    Result,
    active_inference::PreferenceModel,
    tictactoe::{BoardState, GameOutcome, Player},
    workspace::MenaceWorkspace,
};

/// Components of Free Energy computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyComponents {
    /// Expected surprise: -E[ln P(o|π)]
    pub expected_surprise: f64,

    /// KL divergence from prior: KL[q(a|s) || p(a)]
    pub kl_divergence: f64,

    /// Total Free Energy: surprise + KL
    pub total: f64,

    /// Number of states analyzed
    pub num_states: usize,
}

impl FreeEnergyComponents {
    /// Create new Free Energy components
    pub fn new(expected_surprise: f64, kl_divergence: f64, num_states: usize) -> Self {
        Self {
            expected_surprise,
            kl_divergence,
            total: expected_surprise + kl_divergence,
            num_states,
        }
    }

    /// Normalize by number of states (average per-state Free Energy)
    pub fn normalized(&self) -> Self {
        if self.num_states == 0 {
            return self.clone();
        }
        let n = self.num_states as f64;
        Self {
            expected_surprise: self.expected_surprise / n,
            kl_divergence: self.kl_divergence / n,
            total: self.total / n,
            num_states: self.num_states,
        }
    }
}

/// Trait for opponent models used in Free Energy computation
///
/// This abstraction allows Free Energy analysis to work with different
/// assumptions about opponent behavior:
/// - Uniform random play (baseline)
/// - Learned policies from training
/// - Optimal minimax policies
/// - Co-evolving Active Inference agents
pub trait OpponentModel: Send + Sync {
    /// Get the probability distribution over moves for the given state
    ///
    /// Returns None if:
    /// - The state is terminal
    /// - No policy is available for this state
    /// - The opponent has no preference (fall back to uniform)
    fn move_distribution(&self, state: &BoardState) -> Option<Vec<(usize, f64)>>;
}

/// Uniform random opponent model
///
/// This is the default opponent assumption where the opponent plays
/// each legal move with equal probability.
#[derive(Debug, Clone, Copy)]
pub struct UniformOpponent;

impl OpponentModel for UniformOpponent {
    fn move_distribution(&self, state: &BoardState) -> Option<Vec<(usize, f64)>> {
        if state.is_terminal() {
            return None;
        }

        let legal_moves = state.legal_moves();
        let prob = 1.0 / legal_moves.len() as f64;
        Some(legal_moves.iter().map(|&m| (m, prob)).collect())
    }
}

/// Workspace-based opponent model
///
/// Uses a learned policy from a MenaceWorkspace. This enables analyzing
/// Free Energy when the opponent follows a specific learned strategy,
/// including optimal policies or partially-trained policies.
pub struct WorkspaceOpponent {
    workspace: MenaceWorkspace,
}

impl WorkspaceOpponent {
    /// Create a new workspace-based opponent model
    pub fn new(workspace: MenaceWorkspace) -> Self {
        Self { workspace }
    }
}

impl OpponentModel for WorkspaceOpponent {
    fn move_distribution(&self, state: &BoardState) -> Option<Vec<(usize, f64)>> {
        if state.is_terminal() {
            return None;
        }

        let (ctx, label) = state.canonical_context_and_label();

        // Get the workspace policy for canonical state
        let canonical_dist = self.workspace.move_distribution(&label)?;

        // Transform moves from canonical coordinates to original state coordinates
        let original_dist: Vec<(usize, f64)> = canonical_dist
            .into_iter()
            .map(|(canonical_move, prob)| {
                // Map canonical move back to original coordinates
                let original_move = ctx.map_canonical_to_original(canonical_move);
                (original_move, prob)
            })
            .collect();

        Some(original_dist)
    }
}

/// Free Energy analysis for MENACE policies
pub struct FreeEnergyAnalysis<O: OpponentModel = UniformOpponent> {
    preferences: PreferenceModel,
    player: Player,
    opponent_model: O,
}

impl FreeEnergyAnalysis<UniformOpponent> {
    /// Create new Free Energy analysis with uniform opponent model
    pub fn new(mut preferences: PreferenceModel, player: Player) -> Self {
        if player == Player::O {
            preferences = preferences.for_player(Player::O);
        }
        Self {
            preferences,
            player,
            opponent_model: UniformOpponent,
        }
    }
}

impl<O: OpponentModel> FreeEnergyAnalysis<O> {
    /// Create new Free Energy analysis with custom opponent model
    pub fn with_opponent_model(
        mut preferences: PreferenceModel,
        player: Player,
        opponent_model: O,
    ) -> Self {
        if player == Player::O {
            preferences = preferences.for_player(Player::O);
        }
        Self {
            preferences,
            player,
            opponent_model,
        }
    }

    /// Compute Free Energy for a workspace relative to a prior
    pub fn compute_free_energy(
        &self,
        workspace: &MenaceWorkspace,
        prior_workspace: &MenaceWorkspace,
    ) -> Result<FreeEnergyComponents> {
        let mut expected_surprise = 0.0;
        let mut kl_divergence = 0.0;
        let mut num_states = 0;

        // Get all states that have policies
        let states = self.collect_decision_states(workspace)?;

        for state in states {
            let (ctx, label) = state.canonical_context_and_label();

            // Get current policy q(a|s)
            if let Some(current_dist) = workspace.move_distribution(&label) {
                // Get prior policy p(a)
                let prior_dist = prior_workspace
                    .move_distribution(&label)
                    .unwrap_or_else(|| {
                        // If no prior exists, assume uniform
                        let legal_moves = ctx.state.legal_moves();
                        let uniform_prob = 1.0 / legal_moves.len() as f64;
                        legal_moves.iter().map(|&m| (m, uniform_prob)).collect()
                    });

                // Compute expected surprise for this state
                // Use canonical state since move distribution is in canonical coordinates
                let surprise =
                    self.compute_expected_surprise(workspace, &ctx.state, &current_dist)?;
                expected_surprise += surprise;

                // Compute KL divergence
                let kl = self.compute_kl_divergence(&current_dist, &prior_dist);
                kl_divergence += kl;

                num_states += 1;
            }
        }

        Ok(FreeEnergyComponents::new(
            expected_surprise,
            kl_divergence,
            num_states,
        ))
    }

    /// Collect all decision states from initial position
    fn collect_decision_states(&self, workspace: &MenaceWorkspace) -> Result<Vec<BoardState>> {
        let mut states = Vec::new();
        let mut stack = vec![BoardState::new()];
        let mut visited = std::collections::HashSet::new();

        while let Some(state) = stack.pop() {
            if state.is_terminal() {
                continue;
            }

            let (_, label) = state.canonical_context_and_label();
            if visited.contains(&label) {
                continue;
            }
            visited.insert(label.clone());

            // Only include states where this player moves
            if state.to_move == self.player {
                // Check if workspace has this state
                if workspace.move_distribution(&label).is_some() {
                    states.push(state);
                }
            }

            // Explore successors
            for mv in state.legal_moves() {
                if let Ok(next_state) = state.make_move(mv) {
                    stack.push(next_state);
                }
            }
        }

        if states.is_empty() {
            return Err(crate::Error::InvalidConfiguration {
                message: format!(
                    "free energy analysis found no decision states for player {:?}; ensure the workspace includes matchboxes for this perspective",
                    self.player
                ),
            });
        }

        Ok(states)
    }

    /// Compute expected surprise for a state under current policy
    fn compute_expected_surprise(
        &self,
        workspace: &MenaceWorkspace,
        state: &BoardState,
        policy: &[(usize, f64)],
    ) -> Result<f64> {
        let mut expected_surprise = 0.0;

        for &(action, prob) in policy {
            if prob == 0.0 {
                continue;
            }

            // Compute outcome probabilities for this action
            let outcome_probs = self.compute_outcome_probabilities(workspace, state, action)?;

            // Compute surprise for this action: -sum(P(o) * ln P(o))
            let mut action_surprise = 0.0;
            for (outcome, outcome_prob) in outcome_probs {
                if outcome_prob <= 0.0 {
                    continue;
                }

                // Weight by preference
                let pref_prob = self.preference_weight_for_outcome(&outcome);

                // Surprise is -ln(preferred probability weighted by actual probability)
                if pref_prob > 0.0 {
                    action_surprise -= outcome_prob * pref_prob.ln();
                }
            }

            // Weight by policy probability
            expected_surprise += prob * action_surprise;
        }

        Ok(expected_surprise)
    }

    fn preference_weight_for_outcome(&self, outcome: &GameOutcome) -> f64 {
        match outcome {
            GameOutcome::Win(Player::X) => self.preferences.preferred_dist_x_win,
            GameOutcome::Win(Player::O) => self.preferences.preferred_dist_o_win,
            GameOutcome::Draw => self.preferences.preferred_dist_draw,
        }
    }

    #[cfg(test)]
    pub(crate) fn preference_weight_for_outcome_for_test(&self, outcome: &GameOutcome) -> f64 {
        self.preference_weight_for_outcome(outcome)
    }

    /// Compute outcome probabilities for an action (using game tree)
    fn compute_outcome_probabilities(
        &self,
        workspace: &MenaceWorkspace,
        state: &BoardState,
        action: usize,
    ) -> Result<Vec<(GameOutcome, f64)>> {
        let next_state = state.make_move(action)?;

        // If terminal, return certain outcome
        if next_state.is_terminal() {
            let outcome = if let Some(winner) = next_state.winner() {
                GameOutcome::Win(winner)
            } else {
                GameOutcome::Draw
            };
            return Ok(vec![(outcome, 1.0)]);
        }

        // Otherwise, enumerate possible game continuations
        // Enumerate continuations under the configured opponent model (default: uniform fallback)
        let mut outcomes = HashMap::new();
        self.enumerate_outcomes(workspace, &next_state, &mut outcomes, 1.0)?;

        Ok(outcomes.into_iter().collect())
    }

    /// Recursively enumerate outcomes from a state
    fn enumerate_outcomes(
        &self,
        workspace: &MenaceWorkspace,
        state: &BoardState,
        outcomes: &mut HashMap<GameOutcome, f64>,
        probability: f64,
    ) -> Result<()> {
        if state.is_terminal() {
            let outcome = if let Some(winner) = state.winner() {
                GameOutcome::Win(winner)
            } else {
                GameOutcome::Draw
            };
            *outcomes.entry(outcome).or_insert(0.0) += probability;
            return Ok(());
        }

        // Use opponent model to get move distribution
        // Falls back to uniform if opponent model returns None
        let move_dist = if state.to_move == self.player {
            self.agent_move_distribution(workspace, state)
        } else {
            self.opponent_model.move_distribution(state)
        }
        .unwrap_or_else(|| {
            let legal_moves = state.legal_moves();
            let uniform_prob = 1.0 / legal_moves.len() as f64;
            legal_moves.iter().map(|&m| (m, uniform_prob)).collect()
        });

        for (mv, move_prob) in move_dist {
            let next_state = state.make_move(mv)?;
            self.enumerate_outcomes(workspace, &next_state, outcomes, probability * move_prob)?;
        }

        Ok(())
    }

    fn agent_move_distribution(
        &self,
        workspace: &MenaceWorkspace,
        state: &BoardState,
    ) -> Option<Vec<(usize, f64)>> {
        if state.is_terminal() {
            return None;
        }

        let (ctx, label) = state.canonical_context_and_label();
        let canonical_dist = workspace.move_distribution(&label)?;

        Some(
            canonical_dist
                .into_iter()
                .map(|(canonical_move, prob)| {
                    let original_move = ctx.map_canonical_to_original(canonical_move);
                    (original_move, prob)
                })
                .collect(),
        )
    }

    /// Compute KL divergence between two distributions
    fn compute_kl_divergence(&self, q: &[(usize, f64)], p: &[(usize, f64)]) -> f64 {
        let q_map: HashMap<usize, f64> = q.iter().copied().collect();
        let p_map: HashMap<usize, f64> = p.iter().copied().collect();

        let mut kl = 0.0;
        for (&action, &q_prob) in &q_map {
            if q_prob == 0.0 {
                continue;
            }

            let p_prob = p_map.get(&action).copied().unwrap_or(1e-10);
            if p_prob > 0.0 {
                kl += q_prob * (q_prob / p_prob).ln();
            }
        }

        kl
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::workspace::{InitialBeadSchedule, RestockMode, StateFilter};

    #[test]
    fn test_free_energy_computation() {
        let preferences = PreferenceModel::from_probabilities(0.9, 0.5, 0.1);
        let analysis = FreeEnergyAnalysis::new(preferences, Player::X);

        // Create initial and trained workspaces
        let initial_workspace = MenaceWorkspace::with_config(
            StateFilter::Michie,
            RestockMode::default(),
            InitialBeadSchedule::default(),
        )
        .expect("workspace creation");

        let trained_workspace = initial_workspace.clone();
        // Simulate some training by modifying weights (would normally happen during learning)

        let fe = analysis
            .compute_free_energy(&trained_workspace, &initial_workspace)
            .expect("FE computation");

        assert!(fe.num_states > 0, "Should analyze some states");
        assert!(fe.total >= 0.0, "Free Energy should be non-negative");
    }

    #[test]
    fn test_kl_divergence() {
        let preferences = PreferenceModel::from_probabilities(0.9, 0.5, 0.1);
        let analysis = FreeEnergyAnalysis::new(preferences, Player::X);

        // Identical distributions should have zero KL
        let dist1 = vec![(0, 0.5), (1, 0.5)];
        let dist2 = vec![(0, 0.5), (1, 0.5)];
        let kl = analysis.compute_kl_divergence(&dist1, &dist2);
        assert!(kl.abs() < 1e-6, "KL of identical dists should be ~0");

        // Different distributions should have positive KL
        let dist3 = vec![(0, 0.9), (1, 0.1)];
        let kl2 = analysis.compute_kl_divergence(&dist3, &dist2);
        assert!(kl2 > 0.0, "KL of different dists should be > 0");
    }

    #[test]
    fn collect_decision_states_for_o_includes_valid_positions() {
        let preferences = PreferenceModel::from_probabilities(0.6, 0.3, 0.1);
        let analysis = FreeEnergyAnalysis::new(preferences, Player::O);

        let workspace = MenaceWorkspace::with_config(
            StateFilter::Both,
            RestockMode::default(),
            InitialBeadSchedule::default(),
        )
        .expect("workspace");

        let states = analysis
            .collect_decision_states(&workspace)
            .expect("collect decision states");

        assert!(
            states.iter().any(|s| s.to_move == Player::O),
            "expected at least one O-to-move state"
        );

        assert!(
            !states
                .iter()
                .any(|s| s.to_move == Player::O && s.occupied_count() == 0),
            "should not include empty board with O to move"
        );
    }

    #[test]
    fn agent_move_distribution_respects_board_orientation() {
        let preferences = PreferenceModel::from_probabilities(0.7, 0.2, 0.1);
        let analysis = FreeEnergyAnalysis::new(preferences, Player::X);

        let mut workspace = MenaceWorkspace::with_config(
            StateFilter::All,
            RestockMode::default(),
            InitialBeadSchedule::default(),
        )
        .expect("workspace");

        // Construct a position where X is to move but the board is not in canonical orientation.
        let mut state = BoardState::new();
        state = state.make_move(4).expect("x plays center");
        state = state.make_move(0).expect("o plays corner");
        assert_eq!(state.to_move, Player::X);

        let (ctx, label) = state.canonical_context_and_label();
        let actual_move = 8;
        let canonical_move = ctx.map_move_to_canonical(actual_move);

        let mut weights = HashMap::new();
        weights.insert(canonical_move, 5.0);
        workspace.set_move_weights(&label, &weights);

        let distribution = analysis
            .agent_move_distribution(&workspace, &state)
            .expect("agent distribution");

        let mut found_target = false;
        for (mv, prob) in &distribution {
            if *mv == actual_move {
                assert!(
                    (prob - 1.0).abs() < 1e-6,
                    "target move should have probability 1"
                );
                found_target = true;
            } else {
                assert!(
                    *prob <= 1e-12,
                    "non-target moves should have zero probability"
                );
            }
        }
        assert!(
            found_target,
            "expected target move to be present in distribution"
        );
    }

    #[test]
    fn preference_weights_align_for_o_agent() {
        let preferences = PreferenceModel::from_probabilities(0.9, 0.05, 0.05);
        let analysis = FreeEnergyAnalysis::new(preferences, Player::O);

        let agent_win =
            analysis.preference_weight_for_outcome_for_test(&GameOutcome::Win(Player::O));
        let agent_loss =
            analysis.preference_weight_for_outcome_for_test(&GameOutcome::Win(Player::X));

        assert!(
            agent_win > agent_loss,
            "O agent should prefer O wins; win={agent_win}, loss={agent_loss}"
        );
    }

    #[test]
    fn preference_weights_align_for_x_agent() {
        let preferences = PreferenceModel::from_probabilities(0.9, 0.05, 0.05);
        let analysis = FreeEnergyAnalysis::new(preferences, Player::X);

        let agent_win =
            analysis.preference_weight_for_outcome_for_test(&GameOutcome::Win(Player::X));
        let agent_loss =
            analysis.preference_weight_for_outcome_for_test(&GameOutcome::Win(Player::O));

        assert!(
            agent_win > agent_loss,
            "X agent should prefer X wins; win={agent_win}, loss={agent_loss}"
        );
    }
}
