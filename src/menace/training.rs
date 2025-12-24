//! Training utilities for MENACE

use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};

use rand::{SeedableRng, prelude::IndexedRandom, rngs::StdRng};

use super::{
    agent::MenaceAgent,
    optimal::{OptimalPolicy, compute_optimal_policy, kl_divergence, optimal_move_distribution},
};
use crate::{
    tictactoe::{BoardState, Game, GameOutcome, Player},
    workspace::RestockMode,
};

/// Configuration for training session
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub num_games: usize,
    pub opponent: OpponentType,
    pub logging: bool,
    pub seed: Option<u64>,
    pub restock: Option<RestockMode>,
    pub curriculum: Option<Vec<TrainingBlockConfig>>,
    /// Which player the agent plays as (X or O). Defaults to X if not specified.
    pub agent_player: Player,
    /// Which token makes the opening move when replaying games.
    pub first_player: Player,
}

/// Sequential block of games played against a specific opponent configuration.
#[derive(Debug, Clone)]
pub struct TrainingBlockConfig {
    pub games: usize,
    pub opponent: OpponentType,
}

impl TrainingBlockConfig {
    pub fn new(opponent: OpponentType, games: usize) -> Self {
        Self { games, opponent }
    }
}

/// Type of opponent for training
#[derive(Debug, Clone)]
pub enum OpponentType {
    Random,
    AnotherMenace(Rc<RefCell<MenaceAgent>>),
    Minimax,
}

/// A training session
pub struct TrainingSession {
    pub agent: MenaceAgent,
    pub config: TrainingConfig,
    pub games_played: usize,
    pub results: TrainingResults,
    rng: StdRng,
}

/// Results from training
#[derive(Debug, Clone, Default)]
pub struct TrainingResults {
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    pub win_rate_history: Vec<f64>,
    pub kl_history: Vec<f64>,
}

struct TrainingBlockRuntime {
    games: usize,
    opponent: OpponentRuntime,
}

impl TrainingBlockRuntime {
    fn new(config: TrainingBlockConfig) -> Self {
        Self {
            games: config.games,
            opponent: OpponentRuntime::from_type(config.opponent),
        }
    }
}

enum OpponentRuntime {
    Random,
    Menace(Rc<RefCell<MenaceAgent>>),
    Minimax(MinimaxOpponent),
}

impl OpponentRuntime {
    fn from_type(opponent: OpponentType) -> Self {
        match opponent {
            OpponentType::Random => OpponentRuntime::Random,
            OpponentType::AnotherMenace(agent) => OpponentRuntime::Menace(agent),
            OpponentType::Minimax => OpponentRuntime::Minimax(MinimaxOpponent::new()),
        }
    }

    fn select_move(&mut self, rng: &mut StdRng, state: &BoardState) -> Result<usize, crate::Error> {
        match self {
            OpponentRuntime::Random => Self::random_move(rng, state),
            OpponentRuntime::Menace(agent) => Self::menace_move(agent, rng, state),
            OpponentRuntime::Minimax(opponent) => opponent.best_response(state),
        }
    }

    /// Select a random move from available empty positions
    fn random_move(rng: &mut StdRng, state: &BoardState) -> Result<usize, crate::Error> {
        let moves = state.empty_positions();
        moves.choose(rng).copied().ok_or(crate::Error::NoValidMoves)
    }

    /// Select a move using a MENACE agent
    fn menace_move(
        agent: &Rc<RefCell<MenaceAgent>>,
        _rng: &mut StdRng,
        state: &BoardState,
    ) -> Result<usize, crate::Error> {
        // Agent can play as either X or O
        let opponent_player = state.to_move;
        agent.borrow_mut().select_move_as(state, opponent_player)
    }
}

struct MinimaxOpponent {
    policy: Arc<HashMap<String, OptimalPolicy>>,
}

impl MinimaxOpponent {
    fn new() -> Self {
        Self {
            policy: Arc::new(compute_optimal_policy()),
        }
    }

    fn best_response(&self, state: &BoardState) -> Result<usize, crate::Error> {
        let moves = state.empty_positions();
        if moves.is_empty() {
            return Err(crate::Error::NoValidMoves);
        }

        let ctx = state.canonical_context();

        if let Some(policy) = self.policy.get(&ctx.encoding)
            && let Some(&best_canonical_move) = policy.optimal_moves.iter().min()
        {
            let inverse = ctx.transform.inverse();
            let actual_move = inverse.transform_position(best_canonical_move);
            return Ok(actual_move);
        }

        Ok(moves[0])
    }
}

impl TrainingSession {
    /// Create a new training session
    pub fn new(mut agent: MenaceAgent, config: TrainingConfig) -> Self {
        if let Some(restock) = config.restock {
            agent.set_restock_mode(restock);
        }
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(rand::random::<u64>()),
        };

        TrainingSession {
            agent,
            config,
            games_played: 0,
            results: TrainingResults::default(),
            rng,
        }
    }

    /// Run training for configured number of games
    pub fn train(&mut self) -> Result<(), crate::Error> {
        let optimal_distribution = if self.config.logging {
            Some(optimal_move_distribution())
        } else {
            None
        };

        let mut schedule = self.build_schedule();

        for block in &mut schedule {
            if block.games == 0 {
                continue;
            }

            for _ in 0..block.games {
                self.play_training_game(&mut block.opponent)?;
                self.games_played += 1;

                #[allow(clippy::manual_is_multiple_of)]
                if self.games_played % 100 == 0 {
                    let win_rate = self.results.wins as f64 / self.games_played as f64;
                    self.results.win_rate_history.push(win_rate);

                    if let Some(optimal) = optimal_distribution.as_ref() {
                        let kl = kl_divergence(&self.agent, optimal);
                        self.results.kl_history.push(kl);
                    }
                }
            }
        }
        Ok(())
    }

    fn play_training_game(&mut self, opponent: &mut OpponentRuntime) -> Result<(), crate::Error> {
        let mut game = Game::new();
        game.initial = BoardState::new_with_player(self.config.first_player);
        let mut moves = Vec::new();

        let agent_player = self.config.agent_player;

        while game.outcome.is_none() {
            let current = game.current_state()?;
            let to_move = current.to_move;

            let position = if to_move == agent_player {
                // Agent's turn
                self.agent.select_move_as(&current, agent_player)?
            } else {
                // Opponent's turn
                opponent.select_move(&mut self.rng, &current)?
            };

            moves.push(position);
            game.play(position)?;
        }

        // Train agent based on game outcome
        if let Some(outcome) = game.outcome {
            self.agent
                .train_from_moves(self.config.first_player, &moves, outcome, agent_player)?;

            // Update results
            match outcome {
                GameOutcome::Win(winner) if winner == agent_player => {
                    self.results.wins += 1;
                }
                GameOutcome::Draw => {
                    self.results.draws += 1;
                }
                _ => {
                    self.results.losses += 1;
                }
            }
        }

        Ok(())
    }

    fn build_schedule(&self) -> Vec<TrainingBlockRuntime> {
        if let Some(blocks) = &self.config.curriculum {
            let schedule: Vec<_> = blocks
                .iter()
                .filter(|block| block.games > 0)
                .cloned()
                .map(TrainingBlockRuntime::new)
                .collect();
            if !schedule.is_empty() {
                return schedule;
            }
        }

        if self.config.num_games == 0 {
            return Vec::new();
        }

        vec![TrainingBlockRuntime::new(TrainingBlockConfig::new(
            self.config.opponent.clone(),
            self.config.num_games,
        ))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_opponent_blocks_immediate_win() {
        let opponent = MinimaxOpponent::new();
        let state = BoardState::new()
            .make_move_force(0)
            .unwrap() // X
            .make_move_force(4)
            .unwrap() // O
            .make_move_force(1)
            .unwrap(); // X threatens top row
        assert_eq!(state.to_move, Player::O);

        let response = opponent
            .best_response(&state)
            .expect("minimax opponent should find legal move");
        assert_eq!(response, 2, "expected minimax to block at position 2");
    }
}
