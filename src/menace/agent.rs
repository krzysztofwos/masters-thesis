//! MENACE agent that learns to play Tic-Tac-Toe using categorical abstractions.

use std::collections::HashMap;

use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use super::{
    active::PureActiveInference,
    classic::ReinforcementValues,
    learning::{LearningAlgorithm, LearningStrategy},
};
use crate::{
    tictactoe::{BoardState, GameOutcome, Player},
    workspace::{InitialBeadSchedule, MenaceWorkspace, RestockMode, StateFilter},
};

/// MENACE learning agent
pub struct MenaceAgent {
    pub workspace: MenaceWorkspace,
    pub state_filter: StateFilter,
    /// Random number generator
    pub rng: Option<StdRng>,
    pub(crate) restock_mode: RestockMode,
    pub(crate) initial_beads: InitialBeadSchedule,
    /// Learning algorithm strategy
    pub(crate) algorithm: LearningStrategy,
    pub agent_player: Player,
}

impl std::fmt::Debug for MenaceAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MenaceAgent")
            .field("state_filter", &self.state_filter)
            .field("restock_mode", &self.restock_mode)
            .field("algorithm", &self.algorithm.name())
            .field("agent_player", &self.agent_player)
            .finish()
    }
}

impl MenaceAgent {
    /// Create a new builder for constructing a MENACE agent.
    ///
    /// # Example
    /// ```
    /// use menace::menace::agent::MenaceAgent;
    /// use menace::StateFilter;
    ///
    /// let agent = MenaceAgent::builder()
    ///     .seed(42)
    ///     .filter(StateFilter::Michie)
    ///     .build();
    /// ```
    pub fn builder() -> crate::menace::builder::MenaceAgentBuilder {
        crate::menace::builder::MenaceAgentBuilder::new()
    }

    /// Create a new MENACE agent with default configuration (Classic MENACE).
    ///
    /// For more control over configuration, use `MenaceAgent::builder()`.
    ///
    /// # Errors
    /// Returns an error if workspace construction fails.
    pub fn new(seed: Option<u64>) -> crate::Result<Self> {
        let mut builder = Self::builder();
        if let Some(s) = seed {
            builder = builder.seed(s);
        }
        builder.build()
    }

    /// Set or reset the agent's RNG seed
    pub fn reseed(&mut self, seed: Option<u64>) {
        let rng = match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => StdRng::seed_from_u64(rand::random::<u64>()),
        };
        self.rng = Some(rng);
    }

    pub fn restock_mode(&self) -> RestockMode {
        self.restock_mode
    }

    pub fn initial_beads(&self) -> &InitialBeadSchedule {
        &self.initial_beads
    }

    pub fn set_restock_mode(&mut self, restock_mode: RestockMode) {
        if self.restock_mode == restock_mode {
            return;
        }
        self.restock_mode = restock_mode;
        self.workspace.set_restock_mode(restock_mode);
    }

    /// Get the current reinforcement values (if using a reinforcement-based algorithm)
    ///
    /// Returns `Some(values)` for ClassicMenace, `None` for other algorithms like ActiveInference.
    pub fn reinforcement_values(&self) -> Option<ReinforcementValues> {
        self.algorithm.reinforcement_values()
    }

    /// Set reinforcement values (if using a reinforcement-based algorithm)
    ///
    /// Returns `true` if the values were set (algorithm supports reinforcement),
    /// `false` otherwise.
    pub fn set_reinforcement_values(&mut self, values: ReinforcementValues) -> bool {
        self.algorithm.set_reinforcement_values(values)
    }

    /// Access the Pure Active Inference algorithm, if configured.
    pub fn pure_active_inference(&self) -> Option<&PureActiveInference> {
        match &self.algorithm {
            LearningStrategy::PureActiveInference(algo) => Some(algo.as_ref()),
            _ => None,
        }
    }

    /// Helper to rebuild workspace with current configuration
    fn rebuild_workspace(&mut self) -> crate::Result<()> {
        self.workspace =
            MenaceWorkspace::with_config(self.state_filter, self.restock_mode, self.initial_beads)?;
        Ok(())
    }

    pub fn set_initial_beads(&mut self, schedule: InitialBeadSchedule) -> crate::Result<()> {
        if self.initial_beads == schedule {
            return Ok(());
        }
        self.initial_beads = schedule;
        self.rebuild_workspace()
    }

    /// Access the underlying categorical workspace.
    pub fn workspace(&self) -> &MenaceWorkspace {
        &self.workspace
    }

    pub fn workspace_mut(&mut self) -> &mut MenaceWorkspace {
        &mut self.workspace
    }

    /// Select a move for the given board state, playing as the specified player.
    ///
    /// When the workspace is built with `StateFilter::Both`, both X-to-move and O-to-move
    /// states have matchboxes, so this method works for either player.
    ///
    /// When using X-only filters (All, DecisionOnly, Michie), this only works for X.
    pub fn select_move_as(&mut self, state: &BoardState, player: Player) -> crate::Result<usize> {
        self.ensure_agent_player(player);
        self.select_move_internal(state)
    }

    /// Select a move for the given board state (assumes playing as X)
    pub fn select_move(&mut self, state: &BoardState) -> crate::Result<usize> {
        self.select_move_as(state, Player::X)
    }

    fn select_move_internal(&mut self, state: &BoardState) -> crate::Result<usize> {
        let (ctx, label) = state.canonical_context_and_label();

        if self.rng.is_none() {
            self.reseed(None);
        }

        self.workspace.ensure_box_weighted(&label);

        let rng = self
            .rng
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidConfiguration {
                message: "RNG was not initialised for MENACE agent".to_string(),
            })?;
        let sampled_move = self.workspace.sample_move(&label, rng)?;
        let canonical_move = sampled_move.position.value();

        Ok(ctx.map_canonical_to_original(canonical_move))
    }

    fn ensure_agent_player(&mut self, player: Player) {
        if self.agent_player != player {
            self.agent_player = player;
            self.algorithm.set_agent_player(player);
        }
    }

    /// Train from a game history
    pub fn train_from_game(
        &mut self,
        states: Vec<BoardState>,
        moves: Vec<usize>,
        outcome: GameOutcome,
        player: Player,
    ) -> crate::Result<()> {
        self.ensure_agent_player(player);
        self.algorithm
            .train_from_game(&mut self.workspace, &states, &moves, outcome, player)?;
        Ok(())
    }

    /// Train from a minimal game trace consisting of the first player and move sequence.
    pub fn train_from_moves(
        &mut self,
        first_player: Player,
        moves: &[usize],
        outcome: GameOutcome,
        player: Player,
    ) -> crate::Result<()> {
        self.ensure_agent_player(player);
        let mut states = Vec::with_capacity(moves.len());
        let mut state = BoardState::new_with_player(first_player);
        for (index, &mv) in moves.iter().enumerate() {
            states.push(state);
            state = state
                .make_move(mv)
                .map_err(|err| crate::Error::InvalidConfiguration {
                    message: format!(
                        "invalid move {mv} at step {index} while replaying training trajectory: {err}"
                    ),
                })?;
        }
        self.train_from_game(states, moves.to_vec(), outcome, player)?;
        Ok(())
    }

    /// Get the name of the current learning algorithm
    pub fn algorithm_name(&self) -> &str {
        self.algorithm.name()
    }

    /// Get algorithm-specific statistics
    pub fn algorithm_stats(&self) -> HashMap<String, f64> {
        self.algorithm.stats()
    }

    /// Get statistics about the agent
    pub fn stats(&self) -> AgentStats {
        let mut total_beads = 0.0;
        let matchbox_count = self.workspace.decision_labels().count();
        let mut entropies = Vec::with_capacity(matchbox_count);
        let mut total_matchboxes = 0usize;

        for label_str in self.workspace.decision_labels() {
            total_matchboxes += 1;
            // Parse the label string as a CanonicalLabel
            if let Ok(label) = crate::types::CanonicalLabel::parse(label_str)
                && let Some(weights) = self.workspace.move_weights(&label)
            {
                let total: f64 = weights.iter().map(|(_, w)| *w).sum();
                if total > 0.0 {
                    total_beads += total;
                    let entropy =
                        crate::utils::entropy_from_weights(weights.iter().map(|(_, w)| *w));
                    entropies.push(entropy);
                }
            }
        }

        let avg_entropy = if entropies.is_empty() {
            0.0
        } else {
            entropies.iter().sum::<f64>() / entropies.len() as f64
        };

        AgentStats {
            total_matchboxes,
            total_beads,
            avg_entropy,
        }
    }

    /// Reset all matchboxes to initial state
    ///
    /// # Errors
    /// Returns an error if workspace reconstruction fails.
    pub fn reset(&mut self) -> crate::Result<()> {
        self.rebuild_workspace()?;
        self.algorithm.reset();
        Ok(())
    }

    pub fn decision_labels(&self) -> impl Iterator<Item = &String> {
        self.workspace.decision_labels()
    }

    /// Get the probability distribution for moves from a canonical state.
    pub fn canonical_distribution(
        &self,
        label: &crate::types::CanonicalLabel,
    ) -> Option<HashMap<usize, f64>> {
        let distribution = self.workspace.move_distribution(label)?;
        Some(distribution.into_iter().collect())
    }

    /// Set the weights for moves from a canonical state.
    pub fn set_canonical_move_weights(
        &mut self,
        label: &crate::types::CanonicalLabel,
        weights: &HashMap<usize, f64>,
    ) {
        self.workspace.set_move_weights(label, weights);
    }
}

/// Statistics about a MENACE agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    pub total_matchboxes: usize,
    pub total_beads: f64,
    pub avg_entropy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{menace::learning::LearningStrategy, workspace::StateFilter};

    #[test]
    fn reset_clears_classic_progress() {
        let mut agent = MenaceAgent::builder().seed(123).build().unwrap();
        agent
            .train_from_moves(Player::X, &[0, 4, 8], GameOutcome::Draw, Player::X)
            .expect("training from moves should succeed");

        let before = agent
            .algorithm_stats()
            .get("games_trained")
            .copied()
            .unwrap_or_default();
        assert!(before > 0.0, "expected training to increase games_trained");

        agent.reset().unwrap();

        let after = agent
            .algorithm_stats()
            .get("games_trained")
            .copied()
            .unwrap_or_default();
        assert_eq!(after, 0.0, "reset should clear classic MENACE progress");
    }

    #[test]
    fn reset_clears_active_inference_beliefs() {
        let mut agent = MenaceAgent::builder()
            .filter(StateFilter::Both)
            .seed(7)
            .active_inference_uniform(0.4)
            .build()
            .unwrap();

        agent
            .train_from_moves(Player::X, &[0, 4], GameOutcome::Draw, Player::X)
            .expect("training from moves should succeed");

        let tracked_before = match agent.algorithm {
            LearningStrategy::ActiveInference(ref boxed) => boxed.beliefs().tracked_states(),
            _ => panic!("expected Active Inference strategy"),
        };
        assert!(
            tracked_before > 0,
            "beliefs should track at least one opponent state after training"
        );

        agent.reset().unwrap();

        let tracked_after = match agent.algorithm {
            LearningStrategy::ActiveInference(ref boxed) => boxed.beliefs().tracked_states(),
            _ => panic!("expected Active Inference strategy"),
        };
        assert_eq!(
            tracked_after, 0,
            "reset should restore Active Inference beliefs to prior state"
        );

        let games_after = agent
            .algorithm_stats()
            .get("games_played")
            .copied()
            .unwrap_or_default();
        assert_eq!(games_after, 0.0, "games_played counter should reset to 0");
    }
}
