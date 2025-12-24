//! Q-learning and SARSA agents
//!
//! This module implements temporal difference learning agents that use
//! Q-tables to learn optimal policies through experience.

use rand::{Rng, SeedableRng, rngs::StdRng, seq::IndexedRandom};
use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    ports::Learner,
    q_learning::q_table::QTable,
    tictactoe::{BoardState, GameOutcome, Player},
    types::CanonicalLabel,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TdAgentState {
    pub q_table: QTable,
    pub epsilon: f64,
    pub initial_epsilon: f64,
    pub epsilon_decay: f64,
    pub min_epsilon: f64,
    pub rng_seed: Option<u64>,
}

fn build_rng(seed: Option<u64>) -> StdRng {
    if let Some(seed) = seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::rng())
    }
}

/// Q-learning agent (off-policy TD control)
///
/// Learns optimal Q* function by always updating toward the maximum
/// next-state value, regardless of the action actually taken.
#[derive(Debug, Clone)]
pub struct QLearningAgent {
    q_table: QTable,
    epsilon: f64,
    initial_epsilon: f64,
    epsilon_decay: f64,
    min_epsilon: f64,
    rng: StdRng,
    rng_seed: Option<u64>,
}

impl QLearningAgent {
    /// Create a new Q-learning agent
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - α parameter (0.0 to 1.0)
    /// * `discount_factor` - γ parameter (0.0 to 1.0)
    /// * `epsilon` - Initial exploration rate
    /// * `epsilon_decay` - Multiplicative decay per episode
    /// * `min_epsilon` - Minimum exploration rate
    /// * `q_init` - Initial Q-value for unseen states
    pub fn new(
        learning_rate: f64,
        discount_factor: f64,
        epsilon: f64,
        epsilon_decay: f64,
        min_epsilon: f64,
        q_init: f64,
    ) -> Self {
        Self {
            q_table: QTable::new(learning_rate, discount_factor, q_init),
            epsilon,
            initial_epsilon: epsilon,
            epsilon_decay,
            min_epsilon,
            rng: build_rng(None),
            rng_seed: None,
        }
    }

    /// ε-greedy action selection
    fn select_action_epsilon_greedy(
        &mut self,
        state: &CanonicalLabel,
        legal_moves: &[usize],
    ) -> usize {
        if self.rng.random::<f64>() < self.epsilon {
            // Explore: random action
            *legal_moves.choose(&mut self.rng).unwrap()
        } else {
            // Exploit: greedy action based on Q-values
            self.q_table.greedy_action(state, legal_moves)
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self.rng_seed = Some(seed);
        self
    }

    pub(crate) fn export_state(&self) -> TdAgentState {
        TdAgentState {
            q_table: self.q_table.clone(),
            epsilon: self.epsilon,
            initial_epsilon: self.initial_epsilon,
            epsilon_decay: self.epsilon_decay,
            min_epsilon: self.min_epsilon,
            rng_seed: self.rng_seed,
        }
    }

    pub(crate) fn from_state(state: TdAgentState) -> Self {
        Self {
            q_table: state.q_table,
            epsilon: state.epsilon,
            initial_epsilon: state.initial_epsilon,
            epsilon_decay: state.epsilon_decay,
            min_epsilon: state.min_epsilon,
            rng: build_rng(state.rng_seed),
            rng_seed: state.rng_seed,
        }
    }

    pub(crate) fn q_table_size(&self) -> usize {
        self.q_table.size()
    }

    /// Decay epsilon after episode
    fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.min_epsilon);
    }

    fn reset_rng(&mut self) {
        if let Some(seed) = self.rng_seed {
            self.rng = StdRng::seed_from_u64(seed);
        } else {
            self.rng = build_rng(None);
        }
    }
}

impl Learner for QLearningAgent {
    fn select_move(&mut self, state: &BoardState) -> Result<usize> {
        // Get canonical representation
        let ctx = state.canonical_context();
        let canonical_label = CanonicalLabel::from(&ctx);

        // Get legal moves in original coordinates
        let legal_moves = state.legal_moves();
        if legal_moves.is_empty() {
            return Err(crate::error::Error::NoValidMoves);
        }

        // Transform legal moves to canonical coordinates
        let legal_moves_canonical: Vec<usize> = legal_moves
            .iter()
            .map(|&m| ctx.map_move_to_canonical(m))
            .collect();

        // Select action using ε-greedy (in canonical coordinates)
        let canonical_position =
            self.select_action_epsilon_greedy(&canonical_label, &legal_moves_canonical);

        // Map back to original coordinates for execution
        let original_position = ctx.map_canonical_to_original(canonical_position);

        Ok(original_position)
    }

    fn learn(
        &mut self,
        _first_player: Player,
        moves: &[usize],
        outcome: GameOutcome,
        role: Player,
    ) -> Result<()> {
        // Convert outcome to reward
        let reward = match outcome {
            GameOutcome::Win(winner) if winner == role => 1.0,
            GameOutcome::Win(_) => -1.0,
            GameOutcome::Draw => 0.5,
        };

        // Replay trajectory and apply Q-learning updates
        let mut current_state = BoardState::new();

        for (i, &position) in moves.iter().enumerate() {
            let player = if i.is_multiple_of(2) {
                Player::X
            } else {
                Player::O
            };

            // Only update for our moves
            if player == role {
                let ctx = current_state.canonical_context();
                let canonical_label = CanonicalLabel::from(&ctx);

                // Transform position to canonical coordinates
                let canonical_position = ctx.map_move_to_canonical(position);

                // Apply our move
                let next_state = current_state.make_move(position)?;

                // For two-player games, we need to look ahead to OUR next turn
                // Find the next state where it's our turn (after opponent moves)
                let mut next_our_turn_state = next_state;
                let mut done = next_state.is_terminal();

                // If game not over and there's an opponent move, apply it
                if !done && i + 1 < moves.len() {
                    let opponent_move = moves[i + 1];
                    next_our_turn_state = next_state.make_move(opponent_move)?;
                    done = next_our_turn_state.is_terminal();
                }

                // Get next state where it's our turn
                let next_ctx = next_our_turn_state.canonical_context();
                let canonical_next_label = CanonicalLabel::from(&next_ctx);

                // Get legal moves in next state where it's our turn
                let next_legal = next_our_turn_state.legal_moves();

                // Transform next legal moves to canonical coordinates
                let next_legal_canonical: Vec<usize> = next_legal
                    .iter()
                    .map(|&m| next_ctx.map_move_to_canonical(m))
                    .collect();

                // Apply immediate reward only on terminal states
                let step_reward = if done { reward } else { 0.0 };

                // Q-learning update using canonical positions
                self.q_table.q_learning_update(
                    canonical_label,
                    canonical_position,
                    step_reward,
                    &canonical_next_label,
                    &next_legal_canonical,
                    done,
                );
            }

            // Apply move to current state for next iteration
            current_state = current_state.make_move(position)?;
        }

        // Decay epsilon after episode
        self.decay_epsilon();

        Ok(())
    }

    fn name(&self) -> &str {
        "Q-Learning"
    }

    fn reset(&mut self) -> Result<()> {
        self.q_table.reset();
        self.epsilon = self.initial_epsilon;
        self.reset_rng();
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// SARSA agent (on-policy TD control)
///
/// Learns Q^π for the policy it follows (including exploration).
/// More conservative than Q-learning, better for stochastic environments.
#[derive(Debug, Clone)]
pub struct SarsaAgent {
    q_table: QTable,
    epsilon: f64,
    initial_epsilon: f64,
    epsilon_decay: f64,
    min_epsilon: f64,
    rng: StdRng,
    rng_seed: Option<u64>,
}

impl SarsaAgent {
    /// Create a new SARSA agent
    pub fn new(
        learning_rate: f64,
        discount_factor: f64,
        epsilon: f64,
        epsilon_decay: f64,
        min_epsilon: f64,
        q_init: f64,
    ) -> Self {
        Self {
            q_table: QTable::new(learning_rate, discount_factor, q_init),
            epsilon,
            initial_epsilon: epsilon,
            epsilon_decay,
            min_epsilon,
            rng: build_rng(None),
            rng_seed: None,
        }
    }

    fn select_action_epsilon_greedy(
        &mut self,
        state: &CanonicalLabel,
        legal_moves: &[usize],
    ) -> usize {
        if self.rng.random::<f64>() < self.epsilon {
            *legal_moves.choose(&mut self.rng).unwrap()
        } else {
            self.q_table.greedy_action(state, legal_moves)
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self.rng_seed = Some(seed);
        self
    }

    pub(crate) fn export_state(&self) -> TdAgentState {
        TdAgentState {
            q_table: self.q_table.clone(),
            epsilon: self.epsilon,
            initial_epsilon: self.initial_epsilon,
            epsilon_decay: self.epsilon_decay,
            min_epsilon: self.min_epsilon,
            rng_seed: self.rng_seed,
        }
    }

    pub(crate) fn from_state(state: TdAgentState) -> Self {
        Self {
            q_table: state.q_table,
            epsilon: state.epsilon,
            initial_epsilon: state.initial_epsilon,
            epsilon_decay: state.epsilon_decay,
            min_epsilon: state.min_epsilon,
            rng: build_rng(state.rng_seed),
            rng_seed: state.rng_seed,
        }
    }

    pub(crate) fn q_table_size(&self) -> usize {
        self.q_table.size()
    }

    fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.min_epsilon);
    }

    fn reset_rng(&mut self) {
        if let Some(seed) = self.rng_seed {
            self.rng = StdRng::seed_from_u64(seed);
        } else {
            self.rng = build_rng(None);
        }
    }
}

impl Learner for SarsaAgent {
    fn select_move(&mut self, state: &BoardState) -> Result<usize> {
        let ctx = state.canonical_context();
        let canonical_label = CanonicalLabel::from(&ctx);

        let legal_moves = state.legal_moves();
        if legal_moves.is_empty() {
            return Err(crate::error::Error::NoValidMoves);
        }

        // Transform legal moves to canonical coordinates
        let legal_moves_canonical: Vec<usize> = legal_moves
            .iter()
            .map(|&m| ctx.map_move_to_canonical(m))
            .collect();

        // Select action using ε-greedy (in canonical coordinates)
        let canonical_position =
            self.select_action_epsilon_greedy(&canonical_label, &legal_moves_canonical);

        // Map back to original coordinates for execution
        let original_position = ctx.map_canonical_to_original(canonical_position);

        Ok(original_position)
    }

    fn learn(
        &mut self,
        _first_player: Player,
        moves: &[usize],
        outcome: GameOutcome,
        role: Player,
    ) -> Result<()> {
        let reward = match outcome {
            GameOutcome::Win(winner) if winner == role => 1.0,
            GameOutcome::Win(_) => -1.0,
            GameOutcome::Draw => 0.5,
        };

        let mut current_state = BoardState::new();
        let mut prev_state_action: Option<(CanonicalLabel, usize)> = None;

        for (i, &position) in moves.iter().enumerate() {
            let player = if i.is_multiple_of(2) {
                Player::X
            } else {
                Player::O
            };

            if player == role {
                let ctx = current_state.canonical_context();
                let canonical_label = CanonicalLabel::from(&ctx);

                // Transform position to canonical coordinates
                let canonical_position = ctx.map_move_to_canonical(position);

                // If we have a previous state-action, update it now that we know next action
                if let Some((prev_state, prev_action)) = prev_state_action {
                    let next_state = current_state.make_move(position)?;
                    let done = next_state.is_terminal();
                    let step_reward = if done { reward } else { 0.0 };

                    // SARSA update uses actual next action (current action in canonical coordinates)
                    self.q_table.sarsa_update(
                        prev_state,
                        prev_action,
                        step_reward,
                        &canonical_label,
                        canonical_position,
                        done,
                    );
                }

                prev_state_action = Some((canonical_label, canonical_position));
            }

            current_state = current_state.make_move(position)?;
        }

        // Final update for last action (terminal state)
        if let Some((prev_state, prev_action)) = prev_state_action {
            let done = current_state.is_terminal();
            let terminal_ctx = current_state.canonical_context();
            let terminal_label = CanonicalLabel::from(&terminal_ctx);

            // Use dummy action (won't be used since done=true)
            self.q_table.sarsa_update(
                prev_state,
                prev_action,
                reward,
                &terminal_label,
                0, // dummy action (not used when done=true)
                done,
            );
        }

        self.decay_epsilon();
        Ok(())
    }

    fn name(&self) -> &str {
        "SARSA"
    }

    fn reset(&mut self) -> Result<()> {
        self.q_table.reset();
        self.epsilon = self.initial_epsilon;
        self.reset_rng();
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
