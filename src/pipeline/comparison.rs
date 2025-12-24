//! Comparison framework for evaluating multiple learners
//!
//! This module provides a unified interface for comparing different learning approaches:
//! - MENACE agents
//! - Active Inference policies
//! - Optimal policies
//! - Random baselines

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use rand::{Rng, SeedableRng, random, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::{
    Error, Result,
    menace::MenaceAgent,
    ports::Learner,
    tictactoe::{BoardState, GameOutcome, Player},
};

/// MENACE agent wrapper
pub struct MenaceLearner {
    agent: MenaceAgent,
    name: String,
}

impl MenaceLearner {
    /// Create from an existing agent
    pub fn new(agent: MenaceAgent, name: String) -> Self {
        Self { agent, name }
    }

    /// Get reference to underlying agent
    pub fn agent(&self) -> &MenaceAgent {
        &self.agent
    }

    /// Get mutable reference to underlying agent
    pub fn agent_mut(&mut self) -> &mut MenaceAgent {
        &mut self.agent
    }
}

impl Learner for MenaceLearner {
    fn select_move(&mut self, state: &BoardState) -> Result<usize> {
        self.agent.select_move_as(state, state.to_move)
    }

    fn learn(
        &mut self,
        first_player: Player,
        moves: &[usize],
        outcome: GameOutcome,
        player: Player,
    ) -> Result<()> {
        self.agent
            .train_from_moves(first_player, moves, outcome, player)?;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn move_weights(&self, state: &BoardState) -> Option<Vec<(usize, f64)>> {
        use crate::types::CanonicalLabel;
        let ctx = state.canonical_context();
        let label = CanonicalLabel::from(&ctx);
        self.agent.workspace.move_weights(&label).map(|weights| {
            weights
                .into_iter()
                .map(|(canonical_move, weight)| {
                    (ctx.map_canonical_to_original(canonical_move), weight)
                })
                .collect()
        })
    }

    fn set_rng_seed(&mut self, seed: u64) -> Result<()> {
        self.agent.reseed(Some(seed));
        Ok(())
    }

    fn workspace_snapshot(&self) -> Option<&dyn std::any::Any> {
        Some(self.agent.workspace() as &dyn std::any::Any)
    }
}

/// Shared MENACE agent wrapper for self-play
///
/// This wrapper allows multiple learners to share the same underlying MenaceAgent,
/// enabling self-play where a single agent plays against itself and learns from
/// both perspectives (X and O).
pub struct SharedMenaceLearner {
    agent: Arc<Mutex<MenaceAgent>>,
    name: String,
    player: Player,
}

impl SharedMenaceLearner {
    /// Create from a shared agent reference
    pub fn new(agent: Arc<Mutex<MenaceAgent>>, name: String, player: Player) -> Self {
        Self {
            agent,
            name,
            player,
        }
    }

    /// Get a clone of the shared agent reference
    pub fn clone_ref(&self) -> Arc<Mutex<MenaceAgent>> {
        Arc::clone(&self.agent)
    }
}

impl Learner for SharedMenaceLearner {
    fn select_move(&mut self, state: &BoardState) -> Result<usize> {
        debug_assert_eq!(
            state.to_move, self.player,
            "SharedMenaceLearner '{}' expected turn {:?} but received {:?}",
            self.name, self.player, state.to_move
        );
        let mut guard = self.agent.lock().map_err(|_| Error::InvalidConfiguration {
            message: format!(
                "shared MENACE learner '{}' failed to lock agent for move selection",
                self.name
            ),
        })?;
        guard.select_move_as(state, state.to_move)
    }

    fn learn(
        &mut self,
        first_player: Player,
        moves: &[usize],
        outcome: GameOutcome,
        player: Player,
    ) -> Result<()> {
        debug_assert_eq!(
            player, self.player,
            "SharedMenaceLearner '{}' received training data for {:?} but configured for {:?}",
            self.name, player, self.player
        );
        let mut guard = self.agent.lock().map_err(|_| Error::InvalidConfiguration {
            message: format!(
                "shared MENACE learner '{}' failed to lock agent for training",
                self.name
            ),
        })?;
        guard.train_from_moves(first_player, moves, outcome, player)?;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn move_weights(&self, state: &BoardState) -> Option<Vec<(usize, f64)>> {
        use crate::types::CanonicalLabel;
        debug_assert_eq!(
            state.to_move, self.player,
            "SharedMenaceLearner '{}' move_weights mismatch {:?} vs {:?}",
            self.name, state.to_move, self.player
        );
        let guard = self.agent.lock().unwrap();
        let ctx = state.canonical_context();
        let label = CanonicalLabel::from(&ctx);
        if let Some(weights) = guard.workspace.move_weights(&label) {
            return Some(
                weights
                    .into_iter()
                    .map(|(canonical_move, weight)| {
                        (ctx.map_canonical_to_original(canonical_move), weight)
                    })
                    .collect(),
            );
        }

        let swapped = state.swap_players();
        let ctx_swapped = swapped.canonical_context();
        let label_swapped = CanonicalLabel::from(&ctx_swapped);
        guard.workspace.move_weights(&label_swapped).map(|weights| {
            weights
                .into_iter()
                .map(|(canonical_move, weight)| {
                    (
                        ctx_swapped.map_canonical_to_original(canonical_move),
                        weight,
                    )
                })
                .collect()
        })
    }

    fn set_rng_seed(&mut self, seed: u64) -> Result<()> {
        let mut guard = self.agent.lock().map_err(|_| Error::InvalidConfiguration {
            message: format!(
                "shared MENACE learner '{}' failed to lock agent for reseeding",
                self.name
            ),
        })?;
        guard.reseed(Some(seed));
        Ok(())
    }
}

/// Optimal policy learner (minimax)
pub struct OptimalLearner {
    name: String,
    cache: HashMap<String, i32>,
}

impl OptimalLearner {
    /// Create a new optimal learner
    pub fn new(name: String) -> Self {
        Self {
            name,
            cache: HashMap::new(),
        }
    }

    fn minimax(&mut self, state: &BoardState, is_maximizing: bool) -> i32 {
        let key = state.encode();
        if let Some(&value) = self.cache.get(&key) {
            return value;
        }

        if state.is_terminal() {
            let value = match state.winner() {
                Some(Player::X) => 1,
                Some(Player::O) => -1,
                None => 0,
            };
            self.cache.insert(key, value);
            return value;
        }

        let mut best = if is_maximizing { i32::MIN } else { i32::MAX };

        for pos in state.empty_positions() {
            if let Ok(next_state) = state.make_move(pos) {
                let value = self.minimax(&next_state, !is_maximizing);
                best = if is_maximizing {
                    best.max(value)
                } else {
                    best.min(value)
                };
            }
        }

        self.cache.insert(key, best);
        best
    }

    /// Evaluate every legal move in the given state and return its minimax value.
    pub fn evaluate_moves(&mut self, state: &BoardState) -> Vec<(usize, i32)> {
        let is_x = state.to_move == Player::X;
        let mut moves_with_values = Vec::new();
        for pos in state.empty_positions() {
            if let Ok(next_state) = state.make_move(pos) {
                let value = self.minimax(&next_state, !is_x);
                moves_with_values.push((pos, value));
            }
        }
        moves_with_values
    }
}

impl Learner for OptimalLearner {
    fn select_move(&mut self, state: &BoardState) -> Result<usize> {
        let is_x = state.to_move == Player::X;
        let moves_with_values = self.evaluate_moves(state);
        if moves_with_values.is_empty() {
            return Err(crate::Error::NoValidMoves);
        }

        let mut best_move = None;
        let mut best_value = if is_x { i32::MIN } else { i32::MAX };

        for (mv, value) in moves_with_values {
            if (is_x && value > best_value) || (!is_x && value < best_value) {
                best_value = value;
                best_move = Some(mv);
            }
        }

        best_move.ok_or(crate::Error::NoValidMoves)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Random policy learner (baseline)
pub struct RandomLearner {
    name: String,
    rng: StdRng,
}

impl RandomLearner {
    /// Create a new random learner
    pub fn new(name: String) -> Self {
        Self {
            name,
            rng: StdRng::seed_from_u64(random()),
        }
    }

    /// Create a new random learner with a deterministic seed
    pub fn with_seed(name: String, seed: u64) -> Self {
        Self {
            name,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Learner for RandomLearner {
    fn select_move(&mut self, state: &BoardState) -> Result<usize> {
        let moves = state.empty_positions();
        if moves.is_empty() {
            return Err(crate::Error::NoValidMoves);
        }
        let index = self.rng.random_range(0..moves.len());
        Ok(moves[index])
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn move_weights(&self, state: &BoardState) -> Option<Vec<(usize, f64)>> {
        let moves = state.empty_positions();
        if moves.is_empty() {
            return None;
        }
        let weight = 1.0 / moves.len() as f64;
        Some(moves.into_iter().map(|mv| (mv, weight)).collect())
    }

    fn set_rng_seed(&mut self, seed: u64) -> Result<()> {
        self.rng = StdRng::seed_from_u64(seed);
        Ok(())
    }
}

/// Defensive policy learner (blocks winning moves)
///
/// This learner will:
/// 1. Check if the opponent can win on their next move
/// 2. Block that winning move if found
/// 3. Otherwise, play randomly
///
/// Note: This does NOT try to win itself, only to block opponent wins.
pub struct DefensiveLearner {
    name: String,
    rng: StdRng,
}

impl DefensiveLearner {
    /// Create a new defensive learner
    pub fn new(name: String) -> Self {
        Self {
            name,
            rng: StdRng::seed_from_u64(random()),
        }
    }

    /// Create a defensive learner with a deterministic seed
    pub fn with_seed(name: String, seed: u64) -> Self {
        Self {
            name,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Check if a player can win with a given move
    fn is_winning_move(state: &BoardState, pos: usize, player: Player) -> bool {
        if let Ok(next_state) = state.make_move(pos)
            && let Some(winner) = next_state.winner()
        {
            return winner == player;
        }
        false
    }

    /// Find opponent's winning move (if any)
    fn find_opponent_winning_move(state: &BoardState) -> Option<usize> {
        let opponent = state.to_move.opponent();

        for pos in state.empty_positions() {
            // Simulate opponent making this move
            let mut test_state = *state;
            test_state.to_move = opponent; // Switch to opponent's turn temporarily

            if Self::is_winning_move(&test_state, pos, opponent) {
                return Some(pos);
            }
        }
        None
    }
}

impl Learner for DefensiveLearner {
    fn select_move(&mut self, state: &BoardState) -> Result<usize> {
        // First, check if opponent has a winning move
        if let Some(blocking_move) = Self::find_opponent_winning_move(state) {
            return Ok(blocking_move);
        }

        // Otherwise, play randomly
        let moves = state.empty_positions();
        if moves.is_empty() {
            return Err(crate::Error::NoValidMoves);
        }
        let index = self.rng.random_range(0..moves.len());
        Ok(moves[index])
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn set_rng_seed(&mut self, seed: u64) -> Result<()> {
        self.rng = StdRng::seed_from_u64(seed);
        Ok(())
    }
}

/// Frozen learner wrapper - prevents learning during validation
///
/// This wrapper delegates move selection and other operations to the inner learner,
/// but blocks all learning updates. This is useful for validation phases where you
/// want to test the learned policy without modifying it.
///
/// # Use Case
///
/// Active Inference agents that recompute their workspace after each game can
/// have their validation results affected by belief updates during validation.
/// Freezing learning ensures the validation measures the policy as it existed
/// after training, not as it evolves during validation.
///
/// # Example
///
/// ```ignore
/// use menace::pipeline::FrozenLearner;
///
/// // Train agent normally
/// let mut agent = /* ... */;
/// training_pipeline.run(&mut agent, &mut opponent)?;
///
/// // Validate with frozen learning
/// let mut frozen = FrozenLearner::new(&mut agent);
/// validation_pipeline.run(&mut frozen, &mut test_opponent)?;
/// ```
pub struct FrozenLearner<'a> {
    inner: &'a mut dyn Learner,
}

impl<'a> FrozenLearner<'a> {
    /// Create a new frozen learner wrapping the given learner
    pub fn new(inner: &'a mut dyn Learner) -> Self {
        Self { inner }
    }
}

impl Learner for FrozenLearner<'_> {
    fn select_move(&mut self, state: &BoardState) -> Result<usize> {
        // Delegate to inner learner
        self.inner.select_move(state)
    }

    fn learn(
        &mut self,
        _first_player: Player,
        _moves: &[usize],
        _outcome: GameOutcome,
        _role: Player,
    ) -> Result<()> {
        // NO-OP - learning is frozen during validation!
        Ok(())
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn reset(&mut self) -> Result<()> {
        // Allow resets (doesn't affect learned policy)
        self.inner.reset()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        // Return inner's as_any to avoid lifetime issues
        self.inner.as_any()
    }

    fn move_weights(&self, state: &BoardState) -> Option<Vec<(usize, f64)>> {
        self.inner.move_weights(state)
    }

    fn set_rng_seed(&mut self, seed: u64) -> Result<()> {
        self.inner.set_rng_seed(seed)
    }

    fn workspace_snapshot(&self) -> Option<&dyn std::any::Any> {
        let learner: &dyn Learner = &*self.inner;
        learner.workspace_snapshot()
    }
}

/// Result of comparing multiple learners
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Learner names
    pub learners: Vec<String>,

    /// Head-to-head results: (learner1_index, learner2_index) -> (wins, draws, losses)
    pub head_to_head: HashMap<(usize, usize), (usize, usize, usize)>,

    /// Total games played
    pub total_games: usize,
}

impl ComparisonResult {
    /// Create a new comparison result
    pub fn new(learners: Vec<String>) -> Self {
        Self {
            learners,
            head_to_head: HashMap::new(),
            total_games: 0,
        }
    }

    /// Record a game result (from player1's perspective)
    pub fn record_game(&mut self, player1_idx: usize, player2_idx: usize, outcome: GameOutcome) {
        let key = (player1_idx, player2_idx);
        let entry = self.head_to_head.entry(key).or_insert((0, 0, 0));

        match outcome {
            GameOutcome::Win(Player::X) => {
                // Player 1 (X) won
                entry.0 += 1;
            }
            GameOutcome::Win(Player::O) => {
                // Player 2 (O) won, so player 1 lost
                entry.2 += 1;
            }
            GameOutcome::Draw => {
                entry.1 += 1;
            }
        }

        self.total_games += 1;
    }

    /// Get win rate for a learner
    pub fn win_rate(&self, learner_idx: usize) -> f64 {
        let mut wins = 0;
        let mut total = 0;

        for ((p1, p2), (w, d, l)) in &self.head_to_head {
            if *p1 == learner_idx {
                wins += w;
                total += w + d + l;
            } else if *p2 == learner_idx {
                wins += l; // Their losses are our wins
                total += w + d + l;
            }
        }

        if total == 0 {
            0.0
        } else {
            wins as f64 / total as f64
        }
    }
}

/// Framework for comparing multiple learners
pub struct ComparisonFramework {
    learners: Vec<Box<dyn Learner>>,
    first_player: Player,
}

impl ComparisonFramework {
    /// Create a new comparison framework
    pub fn new(learners: Vec<Box<dyn Learner>>) -> Self {
        Self {
            learners,
            first_player: Player::X,
        }
    }

    /// Configure which player makes the first move in simulated games
    pub fn with_first_player(mut self, player: Player) -> Self {
        self.first_player = player;
        self
    }

    /// Run a round-robin comparison
    pub fn compare_round_robin(&mut self, games_per_matchup: usize) -> Result<ComparisonResult> {
        let names: Vec<String> = self.learners.iter().map(|l| l.name().to_string()).collect();
        let mut result = ComparisonResult::new(names);

        // Play each pair against each other
        for i in 0..self.learners.len() {
            for j in (i + 1)..self.learners.len() {
                // Play games_per_matchup games
                for _ in 0..games_per_matchup {
                    let outcome = self.play_game(i, j)?;
                    result.record_game(i, j, outcome);
                }
            }
        }

        Ok(result)
    }

    fn play_game(&mut self, player1_idx: usize, player2_idx: usize) -> Result<GameOutcome> {
        let mut state = BoardState::new_with_player(self.first_player);
        let mut moves = Vec::new();

        while !state.is_terminal() {
            let current_player_idx = if state.to_move == self.first_player {
                player1_idx
            } else {
                player2_idx
            };

            let move_pos = self.learners[current_player_idx].select_move(&state)?;
            moves.push(move_pos);
            state = state.make_move(move_pos)?;
        }

        let outcome = if let Some(winner) = state.winner() {
            GameOutcome::Win(winner)
        } else {
            GameOutcome::Draw
        };

        // Let learners learn from the game
        self.learners[player1_idx].learn(self.first_player, &moves, outcome, self.first_player)?;
        // Flip outcome for player 2
        let outcome_p2 = outcome.swap_players();
        self.learners[player2_idx].learn(
            self.first_player,
            &moves,
            outcome_p2,
            self.first_player.opponent(),
        )?;

        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_learner() {
        let mut learner = RandomLearner::new("Random".to_string());
        let state = BoardState::new();
        let move_pos = learner
            .select_move(&state)
            .expect("random learner should supply a move");
        assert!(move_pos < 9);
    }

    #[test]
    fn test_optimal_learner() {
        let mut learner = OptimalLearner::new("Optimal".to_string());
        let state = BoardState::new();
        let move_pos = learner
            .select_move(&state)
            .expect("optimal learner should supply a move");
        // Optimal first move should be center or corner
        assert!(move_pos == 4 || [0, 2, 6, 8].contains(&move_pos));
    }

    #[test]
    fn comparison_respects_first_player() {
        use std::sync::{Arc, Mutex};

        struct RecordingLearner {
            name: String,
            role_log: Arc<Mutex<Vec<Player>>>,
        }

        impl RecordingLearner {
            fn new(name: &str, log: Arc<Mutex<Vec<Player>>>) -> Self {
                Self {
                    name: name.to_string(),
                    role_log: log,
                }
            }
        }

        impl Learner for RecordingLearner {
            fn select_move(&mut self, state: &BoardState) -> Result<usize> {
                state
                    .empty_positions()
                    .first()
                    .copied()
                    .ok_or(crate::Error::NoValidMoves)
            }

            fn learn(
                &mut self,
                _first_player: Player,
                _moves: &[usize],
                _outcome: GameOutcome,
                role: Player,
            ) -> Result<()> {
                self.role_log.lock().unwrap().push(role);
                Ok(())
            }

            fn name(&self) -> &str {
                &self.name
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        let log_a = Arc::new(Mutex::new(Vec::new()));
        let log_b = Arc::new(Mutex::new(Vec::new()));

        let learners: Vec<Box<dyn Learner>> = vec![
            Box::new(RecordingLearner::new("A", Arc::clone(&log_a))),
            Box::new(RecordingLearner::new("B", Arc::clone(&log_b))),
        ];

        let mut framework = ComparisonFramework::new(learners).with_first_player(Player::O);
        let result = framework.compare_round_robin(1).unwrap();

        assert_eq!(result.total_games, 1);
        assert_eq!(log_a.lock().unwrap().as_slice(), &[Player::O]);
        assert_eq!(log_b.lock().unwrap().as_slice(), &[Player::X]);
    }
}
