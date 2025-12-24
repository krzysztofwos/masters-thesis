//! Statistical analysis of games

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::tictactoe::{BoardState, Player};

/// Complete game analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameAnalysis {
    pub total_states: usize,
    pub valid_states: usize,
    pub canonical_states: usize,
    pub terminal_states: usize,
    pub total_games: usize,
    pub canonical_trajectories: usize,
    pub outcome_distribution: OutcomeDistribution,
    pub length_histogram: HashMap<usize, usize>,
    pub average_game_length: f64,
}

/// Distribution of game outcomes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutcomeDistribution {
    pub x_wins: usize,
    pub o_wins: usize,
    pub draws: usize,
}

/// Statistics for a set of games
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GameStats {
    pub games_analyzed: usize,
    pub unique_positions: usize,
    pub average_branching_factor: f64,
    pub state_visit_frequency: HashMap<String, usize>,
}

impl GameAnalysis {
    /// Run complete analysis of Tic-Tac-Toe
    ///
    /// # Errors
    ///
    /// Returns error if game canonicalization fails during analysis.
    pub fn analyze() -> Result<Self, crate::Error> {
        let mut analysis = GameAnalysis {
            total_states: 0,
            valid_states: 0,
            canonical_states: 0,
            terminal_states: 0,
            total_games: 0,
            canonical_trajectories: 0,
            outcome_distribution: OutcomeDistribution::default(),
            length_histogram: HashMap::new(),
            average_game_length: 0.0,
        };

        // Count all valid states
        analysis.count_states();

        // Analyze all possible games
        analysis.analyze_games()?;

        Ok(analysis)
    }

    fn count_states(&mut self) {
        let mut canonical_set = std::collections::HashSet::new();
        let mut terminal_count = 0;

        // Generate all valid states via BFS
        let mut stack = vec![BoardState::new()];
        let mut seen = std::collections::HashSet::new();

        while let Some(state) = stack.pop() {
            let key = state.encode();
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);

            if state.is_valid() {
                self.valid_states += 1;

                let canonical_key = state.canonical().encode();
                canonical_set.insert(canonical_key);

                if state.is_terminal() {
                    terminal_count += 1;
                }

                if !state.is_terminal() {
                    for pos in state.empty_positions() {
                        if let Ok(next) = state.make_move(pos) {
                            stack.push(next);
                        }
                    }
                }
            }
        }

        self.canonical_states = canonical_set.len();
        self.terminal_states = terminal_count;
        self.total_states = 19_683; // 3^9
    }

    fn analyze_games(&mut self) -> Result<(), crate::Error> {
        let mut canonical_trajectories = std::collections::HashSet::new();
        let mut total_length = 0;

        self.analyze_games_from(
            BoardState::new(),
            Vec::new(),
            &mut canonical_trajectories,
            &mut total_length,
        )?;

        self.canonical_trajectories = canonical_trajectories.len();

        if self.total_games > 0 {
            self.average_game_length = total_length as f64 / self.total_games as f64;
        }

        Ok(())
    }

    fn analyze_games_from(
        &mut self,
        state: BoardState,
        history: Vec<usize>,
        canonical_trajectories: &mut std::collections::HashSet<Vec<usize>>,
        total_length: &mut usize,
    ) -> Result<(), crate::Error> {
        if state.is_terminal() {
            self.total_games += 1;
            let length = history.len();
            *self.length_histogram.entry(length).or_insert(0) += 1;
            *total_length += length;

            // Record outcome
            if let Some(winner) = state.winner() {
                match winner {
                    Player::X => self.outcome_distribution.x_wins += 1,
                    Player::O => self.outcome_distribution.o_wins += 1,
                }
            } else {
                self.outcome_distribution.draws += 1;
            }

            // Add canonical trajectory
            let game = crate::tictactoe::Game {
                initial: BoardState::new(),
                moves: history
                    .iter()
                    .enumerate()
                    .map(|(i, &pos)| crate::tictactoe::game::Move {
                        position: pos,
                        player: if i.is_multiple_of(2) {
                            Player::X
                        } else {
                            Player::O
                        },
                    })
                    .collect(),
                outcome: None,
            };
            let canonical = game.canonical()?;
            canonical_trajectories.insert(canonical);

            return Ok(());
        }

        for pos in state.empty_positions() {
            if let Ok(next) = state.make_move(pos) {
                let mut new_history = history.clone();
                new_history.push(pos);
                self.analyze_games_from(next, new_history, canonical_trajectories, total_length)?;
            }
        }

        Ok(())
    }
}
